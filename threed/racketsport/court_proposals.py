"""Fail-closed court proposal artifact helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping


Point2D = tuple[float, float]


@dataclass(frozen=True)
class CourtProposal:
    proposal_id: str
    source: str
    court_keypoints: dict[str, Point2D]
    scores: dict[str, float | int | None]
    homography_image_from_court: list[list[float]] | None = None
    gate: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        gate = {
            "auto_usable": False,
            "review_usable": True,
            "failed": ["not_verified"],
            "warnings": [],
        }
        gate.update(self.gate)
        gate["auto_usable"] = False
        failed = [str(item) for item in gate.get("failed", [])]
        if "not_verified" not in failed:
            failed.append("not_verified")
        gate["failed"] = failed
        gate["warnings"] = [str(item) for item in gate.get("warnings", [])]
        return {
            "proposal_id": self.proposal_id,
            "source": self.source,
            "verified": False,
            "not_cal3_verified": True,
            "court_keypoints": {
                name: [float(x), float(y)]
                for name, (x, y) in sorted(self.court_keypoints.items())
            },
            "homography_image_from_court": self.homography_image_from_court,
            "scores": dict(sorted(self.scores.items())),
            "gate": gate,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class CourtProposalReport:
    clip: str
    image_size: tuple[int, int]
    frame_indices: list[int]
    proposals: list[CourtProposal]
    video: str | None = None
    motion_mode: str = "unknown"
    assist: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        selected = self.proposals[0].proposal_id if self.proposals else None
        return {
            "artifact_type": "racketsport_court_proposals",
            "schema_version": 1,
            "clip": self.clip,
            "status": "ranked_not_verified",
            "verified": False,
            "not_cal3_verified": True,
            "input": {
                "video": self.video,
                "frame_indices": [int(index) for index in self.frame_indices],
                "image_size": [int(self.image_size[0]), int(self.image_size[1])],
                "motion_mode": self.motion_mode,
            },
            "assist": self.assist or {"mode": "none", "tap_points": [], "line_label": None},
            "ranking": {
                "selected_proposal_id": selected,
                "selection_reason": "best_score_but_review_required" if selected else "no_proposals",
                "abstain": True,
                "abstain_reasons": ["not_cal3_verified"],
            },
            "proposals": [proposal.to_json_dict() for proposal in self.proposals],
        }


def write_court_proposal_report(path: str | Path, report: CourtProposalReport) -> None:
    Path(path).write_text(
        json.dumps(report.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Real Stage 0 (frame selection) + Stage 4 (temporal consensus) + the stable
# detect-court-from-video entry point (CAL-GEO 2026-07-05).
# ---------------------------------------------------------------------------


# ROUND-2 FIX 6: selector fallback safety net. PREDECLARED (before any
# round-2 benchmark measurement, 2026-07-05): when the detector_v2 internal
# support score -- the fraction of observable expected template lines
# supported by the FULL persistent line bank (a label-free, evidence-only
# metric) -- is below this threshold, the proven hough_keypoints-family
# proposal is emitted as SELECTED instead, so the aggregate can never
# regress below the old-system baseline on clips where the homography
# search self-reports weak support. Fallback usage is recorded per clip.
FALLBACK_INTERNAL_SUPPORT_THRESHOLD = 0.65

# GEO r3 (2026-07-08) predeclared in:
# runs/lanes/calv1_geor3_20260708/PREDECLARED.md
GEO_R3_TRIGGER_TEMPORAL_MEDIAN_PX = 24.0
GEO_R3_TRIGGER_MIN_FRAMES = 3
GEO_R3_VOTE_TOP_K = 3
GEO_R3_IDENTITY_LINK_MEDIAN_PX = 96.0
GEO_R3_IDENTITY_MIN_SHARED_KEYPOINTS = 8


def _hough_keypoints_fallback_proposal(
    frame: Any,
    *,
    reason: str,
    internal_support_score: float | None,
) -> CourtProposal | None:
    try:
        from .court_line_keypoints import detect_court_keypoints_from_image

        detected = detect_court_keypoints_from_image(frame)
    except Exception:
        return None
    court_keypoints: dict[str, Point2D] = {}
    for name, item in detected.keypoints.items():
        xy = item.get("xy") if isinstance(item, Mapping) else None
        if isinstance(xy, (list, tuple)) and len(xy) == 2:
            court_keypoints[str(name)] = (float(xy[0]), float(xy[1]))
    if len(court_keypoints) < 4:
        return None
    return CourtProposal(
        proposal_id="proposal_hough_keypoints_fallback_0001",
        source="hough_keypoints_fallback",
        court_keypoints=court_keypoints,
        scores={"overall": round(max(0.0, min(1.0, float(detected.confidence))), 4)},
        gate={
            "auto_usable": False,
            "review_usable": True,
            "failed": ["fallback_selected_low_detector_support", "not_verified"],
            "warnings": [],
        },
        evidence={
            "fallback_used": True,
            "fallback_family": "hough_keypoints",
            "fallback_reason": reason,
            "internal_support_score": internal_support_score,
            "fallback_threshold": FALLBACK_INTERNAL_SUPPORT_THRESHOLD,
        },
    )


def _linspace_indices(start: int, stop: int, count: int) -> list[int]:
    if count <= 1:
        return [int(start)]
    step = (float(stop) - float(start)) / float(count - 1)
    return [int(round(float(start) + step * index)) for index in range(count)]


def select_frames_for_proposals(
    video_path: str | Path,
    *,
    max_frames: int = 24,
    top_k: int = 8,
) -> dict[str, Any]:
    """Stage 0: sample frames, score sharpness, keep top-K sharp/spread frames.

    Sharpness is the variance of the Laplacian (a standard blur metric): a
    motion-blurred frame loses high-frequency edge content and scores low.
    Candidates are bucketed into `top_k` roughly-equal temporal segments and
    the sharpest frame per bucket is kept, so the selection is both blur-
    resistant (rejects frames like a pure-motion-blur frame) and temporally
    spread across the clip rather than clustering around one sharp moment.
    Accepts a video file, a single image, OR a directory of numbered `.jpg`
    frames (the shape benchmark eval-clip samples use). Dependency-light: cv2
    + numpy only.
    """

    import cv2  # type: ignore[import-not-found]
    import numpy as np

    path = Path(video_path)
    candidates: list[dict[str, Any]] = []
    if path.is_dir():
        frame_paths = sorted(path.glob("*.jpg"))
        count = max(1, min(int(max_frames), len(frame_paths)))
        chosen_paths = [frame_paths[index] for index in _linspace_indices(0, max(0, len(frame_paths) - 1), count)] if frame_paths else []
        for frame_index, frame_path in enumerate(chosen_paths):
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            candidates.append({"frame_index": int(frame_index), "frame": frame, "sharpness": sharpness})
        if not candidates:
            raise ValueError(f"no readable .jpg frames in directory: {path}")
    else:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            candidates.append({"frame_index": 0, "frame": image, "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var())})
        else:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise ValueError(f"cannot open input as image/video/frame-directory: {path}")
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total > 0:
                count = max(1, min(int(max_frames), total))
                indices = [int(round(value)) for value in np.linspace(0, max(0, total - 1), count)]
            else:
                indices = list(range(max(1, int(max_frames))))
            for frame_index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                candidates.append({"frame_index": int(frame_index), "frame": frame, "sharpness": sharpness})
            cap.release()
            if not candidates:
                raise ValueError(f"no frames decoded from video: {path}")

    max_sharpness = max(item["sharpness"] for item in candidates)
    k = max(1, min(int(top_k), len(candidates)))
    bucket_size = max(1, len(candidates) // k)
    selected: list[dict[str, Any]] = []
    for bucket_index in range(k):
        start = bucket_index * bucket_size
        end = (bucket_index + 1) * bucket_size if bucket_index < k - 1 else len(candidates)
        bucket = candidates[start:end]
        if not bucket:
            continue
        selected.append(max(bucket, key=lambda item: item["sharpness"]))
    selected.sort(key=lambda item: item["frame_index"])
    rejected_blur = [
        item["frame_index"] for item in candidates if item["sharpness"] < 0.12 * max_sharpness and item not in selected
    ]
    return {
        "selected": selected,
        "candidate_count": len(candidates),
        "candidate_frame_indices": [item["frame_index"] for item in candidates],
        "rejected_blur_frame_indices": rejected_blur,
        "max_sharpness": round(float(max_sharpness), 3),
    }


_CONSENSUS_ANCHOR_NAMES = ("near_left_corner", "near_right_corner", "far_left_corner", "far_right_corner")


def _persistent_frame_cluster(
    per_frame_keypoints: list[dict[str, tuple[float, float]]],
    *,
    distance_threshold_px: float = 90.0,
) -> list[int]:
    """Stage 4 persistence: pick the largest set of MUTUALLY agreeing frames.

    Different frames can each find an internally self-consistent but
    DIFFERENT court (e.g. one frame locks onto the true court, another onto
    an adjacent one or a partial mis-read). A naive per-keypoint median over
    every frame blends these into a "franken-court" that matches neither.
    This adapts the benchmark's `_persistent_temporal_segments` idea: cluster
    frames by mutual corner-position agreement and keep only the majority
    cluster, so isolated outlier-court frames cannot drag the consensus.
    Returns the indices (into `per_frame_keypoints`) of the retained frames.
    """

    import math

    anchors: list[tuple[float, float] | None] = []
    for keypoints in per_frame_keypoints:
        points = [keypoints[name] for name in _CONSENSUS_ANCHOR_NAMES if name in keypoints]
        if not points:
            anchors.append(None)
            continue
        anchors.append((sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)))

    n = len(anchors)
    if n <= 2:
        return list(range(n))

    neighbor_counts = []
    for i in range(n):
        if anchors[i] is None:
            neighbor_counts.append(-1)
            continue
        count = 0
        for j in range(n):
            if i == j or anchors[j] is None:
                continue
            if math.hypot(anchors[i][0] - anchors[j][0], anchors[i][1] - anchors[j][1]) <= distance_threshold_px:
                count += 1
        neighbor_counts.append(count)

    seed = max(range(n), key=lambda index: neighbor_counts[index])
    if neighbor_counts[seed] < 0:
        return list(range(n))
    cluster = [
        index
        for index in range(n)
        if anchors[index] is not None
        and math.hypot(anchors[index][0] - anchors[seed][0], anchors[index][1] - anchors[seed][1]) <= distance_threshold_px
    ]
    return cluster if cluster else list(range(n))


def _temporal_consensus(
    per_frame_keypoints: list[dict[str, tuple[float, float]]],
) -> tuple[dict[str, tuple[float, float]], dict[str, Any], list[int]]:
    """Stage 4: persistence-clustered, robust per-keypoint median consensus.

    Returns (consensus_keypoints, temporal_stability, retained_frame_indices)
    so callers can pick a representative frame from the SAME persistent
    cluster the consensus was built from, instead of an arbitrary best-scored
    frame that might itself be one of the discarded outliers.
    """

    import math
    import statistics

    cluster_indices = _persistent_frame_cluster(per_frame_keypoints)
    retained = [per_frame_keypoints[index] for index in cluster_indices]

    names: set[str] = set()
    for keypoints in retained:
        names |= set(keypoints.keys())

    consensus: dict[str, tuple[float, float]] = {}
    spreads: list[float] = []
    per_keypoint_spread: dict[str, float] = {}
    for name in sorted(names):
        xs = [keypoints[name][0] for keypoints in retained if name in keypoints]
        ys = [keypoints[name][1] for keypoints in retained if name in keypoints]
        if not xs:
            continue
        cx, cy = statistics.median(xs), statistics.median(ys)
        consensus[name] = (float(cx), float(cy))
        if len(xs) > 1:
            distances = [math.hypot(x - cx, y - cy) for x, y in zip(xs, ys)]
            spread = statistics.median(distances)
            spreads.append(spread)
            per_keypoint_spread[name] = round(float(spread), 3)

    temporal_stability = {
        "median": round(float(statistics.median(spreads)), 3) if spreads else 0.0,
        "p95": round(_percentile95(spreads), 3) if spreads else 0.0,
        "frame_count": len(retained),
        "total_frame_count": len(per_frame_keypoints),
        "outlier_frame_count": len(per_frame_keypoints) - len(retained),
        "per_keypoint_spread_px": per_keypoint_spread,
    }
    return consensus, temporal_stability, cluster_indices


def _percentile95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = 0.95 * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _geor3_temporal_trigger_fires(
    temporal_stability_px: Mapping[str, Any],
    *,
    enabled: bool = True,
    temporal_median_threshold_px: float = GEO_R3_TRIGGER_TEMPORAL_MEDIAN_PX,
    min_frames: int = GEO_R3_TRIGGER_MIN_FRAMES,
) -> bool:
    if not enabled:
        return False
    median = _as_float(temporal_stability_px.get("median"))
    frame_count = _as_int(temporal_stability_px.get("frame_count"))
    if median is None or frame_count is None:
        return False
    return frame_count >= int(min_frames) and median > float(temporal_median_threshold_px)


def _geor3_ranked_top_hypotheses(
    pickleball_hypotheses: list[dict[str, Any]],
    best: dict[str, Any] | None,
    *,
    top_k: int = GEO_R3_VOTE_TOP_K,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    seen: set[int] = set()
    if best is not None:
        ranked.append(best)
        seen.add(id(best))
    for hypothesis in pickleball_hypotheses:
        if id(hypothesis) in seen:
            continue
        ranked.append(hypothesis)
        seen.add(id(hypothesis))
        if len(ranked) >= max(1, int(top_k)):
            break
    return ranked[: max(1, int(top_k))]


def _geor3_select_identity_vote(
    frames_with_hypothesis: list[dict[str, Any]],
    *,
    top_k: int = GEO_R3_VOTE_TOP_K,
    identity_link_median_px: float = GEO_R3_IDENTITY_LINK_MEDIAN_PX,
    min_shared_keypoints: int = GEO_R3_IDENTITY_MIN_SHARED_KEYPOINTS,
) -> dict[str, Any]:
    """Vote for a persistent court identity across each frame's top-3 hypotheses.

    The vote is label-free: it clusters projected court keypoints by pixel
    agreement and counts distinct frame support. The returned `selected_frames`
    is intentionally private/in-memory because frame records contain image
    arrays; use `_geor3_json_safe_vote` for artifact metadata.
    """

    import math

    frame_count = len(frames_with_hypothesis)
    candidates: list[dict[str, Any]] = []
    for frame_pos, item in enumerate(frames_with_hypothesis):
        hypotheses = list(item.get("top_pickleball_hypotheses") or [])
        if not hypotheses and item.get("best") is not None:
            hypotheses = [item["best"]]
        for rank, hypothesis in enumerate(hypotheses[: max(1, int(top_k))], start=1):
            keypoints = _coerce_keypoints(hypothesis.get("keypoints") or {})
            if len(keypoints) < int(min_shared_keypoints):
                continue
            candidates.append(
                {
                    "candidate_index": len(candidates),
                    "frame_pos": frame_pos,
                    "frame_index": int(item.get("frame_index") or frame_pos),
                    "rank": rank,
                    "score": float(hypothesis.get("score") or 0.0),
                    "hypothesis_id": str(hypothesis.get("hypothesis_id") or f"frame_{frame_pos}_rank_{rank}"),
                    "hypothesis": hypothesis,
                    "keypoints": keypoints,
                }
            )

    required_support = int(math.ceil(frame_count / 2.0)) if frame_count else 0
    if not candidates or required_support <= 0:
        return {
            "attempted": True,
            "selected": False,
            "frame_count": frame_count,
            "required_support_frame_count": required_support,
            "candidate_count": len(candidates),
            "cluster_count": 0,
            "support_frame_count": 0,
            "selected_cluster_id": None,
            "selected_frame_indices": [],
            "selected_hypothesis_ids": [],
            "selected_frames": [],
            "clusters": [],
            "identity_link_median_px": float(identity_link_median_px),
            "identity_min_shared_keypoints": int(min_shared_keypoints),
        }

    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            distance = _geor3_median_keypoint_distance(
                candidates[left]["keypoints"],
                candidates[right]["keypoints"],
                min_shared_keypoints=int(min_shared_keypoints),
            )
            if distance is not None and distance <= float(identity_link_median_px):
                union(left, right)

    grouped: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault(find(candidate["candidate_index"]), []).append(candidate)

    clusters: list[dict[str, Any]] = []
    for cluster_id, members in grouped.items():
        per_frame_best: dict[int, dict[str, Any]] = {}
        for candidate in members:
            current = per_frame_best.get(candidate["frame_pos"])
            if current is None or _geor3_candidate_order(candidate) < _geor3_candidate_order(current):
                per_frame_best[candidate["frame_pos"]] = candidate
        best_members = [per_frame_best[index] for index in sorted(per_frame_best)]
        support = len(best_members)
        mean_rank = sum(float(item["rank"]) for item in best_members) / float(support)
        mean_score = sum(float(item["score"]) for item in best_members) / float(support)
        clusters.append(
            {
                "cluster_id": int(cluster_id),
                "support_frame_count": int(support),
                "frame_indices": [int(item["frame_index"]) for item in best_members],
                "hypothesis_ids": [str(item["hypothesis_id"]) for item in best_members],
                "mean_rank": round(float(mean_rank), 6),
                "mean_score": round(float(mean_score), 6),
                "_best_members": best_members,
            }
        )

    clusters.sort(
        key=lambda item: (
            -int(item["support_frame_count"]),
            float(item["mean_rank"]),
            float(item["mean_score"]),
            int(item["cluster_id"]),
        )
    )
    best_cluster = clusters[0]
    selected = int(best_cluster["support_frame_count"]) >= required_support
    selected_frames: list[dict[str, Any]] = []
    selected_hypothesis_ids: list[str] = []
    selected_frame_indices: list[int] = []
    if selected:
        for member in best_cluster["_best_members"]:
            frame = dict(frames_with_hypothesis[int(member["frame_pos"])])
            frame["best"] = member["hypothesis"]
            frame["geor3_vote_rank"] = int(member["rank"])
            selected_frames.append(frame)
            selected_hypothesis_ids.append(str(member["hypothesis_id"]))
            selected_frame_indices.append(int(member["frame_index"]))

    json_clusters = []
    for item in clusters:
        json_clusters.append({key: value for key, value in item.items() if key != "_best_members"})
    return {
        "attempted": True,
        "selected": bool(selected),
        "frame_count": int(frame_count),
        "required_support_frame_count": int(required_support),
        "candidate_count": len(candidates),
        "cluster_count": len(clusters),
        "support_frame_count": int(best_cluster["support_frame_count"]),
        "selected_cluster_id": int(best_cluster["cluster_id"]) if selected else None,
        "selected_frame_indices": selected_frame_indices,
        "selected_hypothesis_ids": selected_hypothesis_ids,
        "selected_frames": selected_frames,
        "clusters": json_clusters,
        "identity_link_median_px": float(identity_link_median_px),
        "identity_min_shared_keypoints": int(min_shared_keypoints),
    }


def _geor3_json_safe_vote(vote: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in vote.items() if key != "selected_frames"}


def _geor3_candidate_order(candidate: Mapping[str, Any]) -> tuple[int, float, str]:
    return (int(candidate.get("rank") or 999), float(candidate.get("score") or 0.0), str(candidate.get("hypothesis_id") or ""))


def _coerce_keypoints(points: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    keypoints: dict[str, tuple[float, float]] = {}
    for name, xy in points.items():
        if isinstance(xy, (list, tuple)) and len(xy) == 2:
            x = _as_float(xy[0])
            y = _as_float(xy[1])
            if x is not None and y is not None:
                keypoints[str(name)] = (float(x), float(y))
    return keypoints


def _geor3_median_keypoint_distance(
    left: Mapping[str, tuple[float, float]],
    right: Mapping[str, tuple[float, float]],
    *,
    min_shared_keypoints: int,
) -> float | None:
    import math
    import statistics

    shared = sorted(set(left) & set(right))
    if len(shared) < int(min_shared_keypoints):
        return None
    distances = [math.hypot(float(left[name][0]) - float(right[name][0]), float(left[name][1]) - float(right[name][1])) for name in shared]
    return float(statistics.median(distances)) if distances else None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def propose_court_from_video(
    video_path: str | Path,
    *,
    max_frames: int = 24,
    top_k: int = 8,
    tracks_path: str | Path | None = None,
    geo_r3_enabled: bool = True,
    geo_r3_trigger_temporal_median_px: float = GEO_R3_TRIGGER_TEMPORAL_MEDIAN_PX,
    geo_r3_min_trigger_frames: int = GEO_R3_TRIGGER_MIN_FRAMES,
    geo_r3_vote_top_k: int = GEO_R3_VOTE_TOP_K,
    geo_r3_identity_link_median_px: float = GEO_R3_IDENTITY_LINK_MEDIAN_PX,
    geo_r3_identity_min_shared_keypoints: int = GEO_R3_IDENTITY_MIN_SHARED_KEYPOINTS,
) -> dict[str, Any]:
    """Real Stage 0-6 multi-frame court solver. STABLE CONTRACT.

    Samples frames, builds net/surface/line-bank evidence per frame, searches
    real homography hypotheses with a joint pickleball-vs-tennis template
    competition, takes a robust per-keypoint temporal consensus across the
    frames that produced a pickleball-tagged hypothesis, and computes real
    (non-label) verification metrics for the consensus. Always returns a
    fail-closed `CourtProposalReport` dict (`verified=False`,
    `not_cal3_verified=True`) -- this function never promotes calibration; it
    is a review-only proposal generator. Dependency-light: cv2 + numpy only
    plus this package's own modules.
    """

    from .court_detector_v2_hypotheses import generate_homography_hypotheses
    from .court_detector_v2_net import detect_court_net_evidence
    from .court_detector_v2_surface import build_surface_paint_evidence
    from .court_detector_v2_verify import (
        compute_tennis_overlay_rejection,
        compute_top_net_validation,
        verify_court_hypothesis,
    )

    clip = Path(str(video_path)).stem

    def _empty_report(reason: str) -> dict[str, Any]:
        return CourtProposalReport(
            clip=clip,
            video=str(video_path),
            image_size=(0, 0),
            frame_indices=[],
            proposals=[
                CourtProposal(
                    proposal_id="proposal_empty_0001",
                    source="detector_v2_multiframe",
                    court_keypoints={},
                    scores={"overall": 0.0},
                    gate={
                        "auto_usable": False,
                        "review_usable": False,
                        "failed": [reason, "not_verified"],
                        "warnings": [],
                    },
                )
            ],
        ).to_json_dict()

    try:
        selection = select_frames_for_proposals(video_path, max_frames=max_frames, top_k=top_k)
    except Exception as exc:
        return _empty_report(f"frame_selection_failed:{exc}")

    selected = selection["selected"]
    if not selected:
        return _empty_report("no_frames_selected")

    image_size = (int(selected[0]["frame"].shape[1]), int(selected[0]["frame"].shape[0]))
    tracks = _load_tracks_feet(tracks_path) if tracks_path is not None else None

    from .court_line_bank import build_merged_line_bank

    per_frame_results: list[dict[str, Any]] = []
    for item in selected:
        frame = item["frame"]
        try:
            net_evidence = detect_court_net_evidence(frame)
        except Exception:
            net_evidence = {}
        try:
            surface_evidence = build_surface_paint_evidence(frame)
        except Exception:
            surface_evidence = {}
        try:
            frame_line_bank = build_merged_line_bank(frame)
        except Exception:
            frame_line_bank = None
        try:
            hypotheses = generate_homography_hypotheses(
                frame,
                net_evidence=net_evidence,
                surface_evidence=surface_evidence,
                max_hypotheses=40,
                line_bank=frame_line_bank,
            )
        except Exception:
            hypotheses = []
        pickleball_hypotheses = [h for h in hypotheses if h.get("template") == "pickleball"]
        tennis_hypotheses = [h for h in hypotheses if h.get("template") == "tennis"]
        best = _best_by_net_and_feet(pickleball_hypotheses, net_evidence=net_evidence, tracks=tracks)
        top_pickleball_hypotheses = _geor3_ranked_top_hypotheses(
            pickleball_hypotheses,
            best,
            top_k=geo_r3_vote_top_k,
        )
        pickleball_runner_up = next((h for h in top_pickleball_hypotheses if h is not best), None)
        per_frame_results.append(
            {
                "frame_index": item["frame_index"],
                "sharpness": item["sharpness"],
                "frame": frame,
                "best": best,
                "top_pickleball_hypotheses": top_pickleball_hypotheses,
                "pickleball_runner_up": pickleball_runner_up,
                "tennis_best": tennis_hypotheses[0] if tennis_hypotheses else None,
                "net_evidence": net_evidence,
                "surface_evidence": surface_evidence,
                "line_bank_segments": list((frame_line_bank or {}).get("segments") or []),
                "all_hypotheses_count": len(hypotheses),
            }
        )

    frames_with_hypothesis = [item for item in per_frame_results if item["best"] is not None]
    if not frames_with_hypothesis:
        no_hypothesis_proposal = CourtProposal(
            proposal_id="proposal_no_hypothesis_0001",
            source="detector_v2_multiframe",
            court_keypoints={},
            scores={"overall": 0.0},
            gate={
                "auto_usable": False,
                "review_usable": False,
                "failed": ["no_pickleball_hypothesis_found", "not_verified"],
                "warnings": [],
            },
            evidence={
                "frame_selection": _json_safe_selection(selection),
                "fallback_used": False,
                "internal_support_score": 0.0,
            },
        )
        # Fix 6: zero pickleball hypotheses is the degenerate low-support
        # case -- fall back to the proven hough_keypoints family.
        sharpest = max(selected, key=lambda item: item["sharpness"])
        fallback = _hough_keypoints_fallback_proposal(
            sharpest["frame"], reason="no_pickleball_hypothesis_found", internal_support_score=0.0
        )
        proposals = [fallback, no_hypothesis_proposal] if fallback is not None else [no_hypothesis_proposal]
        report = CourtProposalReport(
            clip=clip,
            video=str(video_path),
            image_size=image_size,
            frame_indices=[item["frame_index"] for item in selected],
            proposals=proposals,
        )
        return report.to_json_dict()

    consensus_keypoints, temporal_stability, retained_indices = _temporal_consensus(
        [item["best"]["keypoints"] for item in frames_with_hypothesis]
    )
    r2_temporal_stability = dict(temporal_stability)
    active_frames_with_hypothesis = frames_with_hypothesis
    geor3_vote: dict[str, Any] = {
        "attempted": False,
        "enabled": bool(geo_r3_enabled),
        "triggered": False,
        "selected": False,
        "trigger": {
            "temporal_median_px": float(geo_r3_trigger_temporal_median_px),
            "min_frames": int(geo_r3_min_trigger_frames),
            "r2_temporal_stability_px": r2_temporal_stability,
        },
        "config": {
            "top_k": int(geo_r3_vote_top_k),
            "identity_link_median_px": float(geo_r3_identity_link_median_px),
            "identity_min_shared_keypoints": int(geo_r3_identity_min_shared_keypoints),
        },
    }
    if _geor3_temporal_trigger_fires(
        r2_temporal_stability,
        enabled=geo_r3_enabled,
        temporal_median_threshold_px=geo_r3_trigger_temporal_median_px,
        min_frames=geo_r3_min_trigger_frames,
    ):
        vote = _geor3_select_identity_vote(
            frames_with_hypothesis,
            top_k=geo_r3_vote_top_k,
            identity_link_median_px=geo_r3_identity_link_median_px,
            min_shared_keypoints=geo_r3_identity_min_shared_keypoints,
        )
        geor3_vote = {
            **_geor3_json_safe_vote(vote),
            "enabled": bool(geo_r3_enabled),
            "triggered": True,
            "trigger": {
                "temporal_median_px": float(geo_r3_trigger_temporal_median_px),
                "min_frames": int(geo_r3_min_trigger_frames),
                "r2_temporal_stability_px": r2_temporal_stability,
            },
            "config": {
                "top_k": int(geo_r3_vote_top_k),
                "identity_link_median_px": float(geo_r3_identity_link_median_px),
                "identity_min_shared_keypoints": int(geo_r3_identity_min_shared_keypoints),
            },
        }
        if vote.get("selected"):
            active_frames_with_hypothesis = list(vote.get("selected_frames") or [])
            consensus_keypoints, temporal_stability, retained_indices = _temporal_consensus(
                [item["best"]["keypoints"] for item in active_frames_with_hypothesis]
            )
    # The representative frame (used for the real evidence-based verify
    # metrics below) must come from the SAME persistent cluster the consensus
    # was built from -- picking an outlier frame here would score the
    # consensus keypoints against a frame whose own evidence disagrees with
    # them, which is not a fair evidence check.
    retained_frames = [active_frames_with_hypothesis[index] for index in retained_indices] or active_frames_with_hypothesis
    representative = min(retained_frames, key=lambda item: item["best"]["score"])

    # ROUND-2 FIX 4: verify metrics are scored against the FULL PERSISTENT
    # line bank (union of the retained frames' merged banks, deduped) plus
    # the surface-polygon boundary -- NOT against the segments that built the
    # winning hypothesis. This replaces the self-confirming distance-
    # transform-at-keypoints metric (which read 0.0 for compressed fits whose
    # mis-assigned keypoints all sat on real paint).
    from .court_detector_v2_verify import compute_visible_error_px_against_line_bank
    from .court_line_bank import court_line_pixel_mask, dedupe_line_segments

    persistent_segments = dedupe_line_segments(
        [segment for item in retained_frames for segment in item.get("line_bank_segments") or []]
    )
    representative_pixel_mask = court_line_pixel_mask(representative["frame"], dilation_px=5)
    representative_surface_polygon = (
        (representative.get("surface_evidence") or {}).get("surface_polygon") or {}
    ).get("interior_polygon")
    visible_error = compute_visible_error_px_against_line_bank(
        consensus_keypoints,
        segments=persistent_segments,
        pixel_mask=representative_pixel_mask,
        image_size=image_size,
        surface_polygon=representative_surface_polygon,
    )
    line_support = {
        "required_lines_present": bool(representative["best"]["required_lines_present"]),
        "semantic_line_count": int(representative["best"]["supported_line_count"]),
    }
    top_net_validation = compute_top_net_validation(representative["net_evidence"], consensus_keypoints)
    tennis_overlay_rejection = compute_tennis_overlay_rejection(representative["best"])

    verification = verify_court_hypothesis(
        hypothesis={"hypothesis_id": representative["best"].get("hypothesis_id", "detector_v2_multiframe_consensus")},
        visible_error_px=visible_error,
        line_support=line_support,
        temporal_stability_px=temporal_stability,
        top_net_validation=top_net_validation,
        tennis_overlay_rejection=tennis_overlay_rejection,
    )
    # ROUND-2 FIX 2: if BOTH near-side floor lines are unobservable
    # (off-frame), the near half of the court is pure extrapolation --
    # confidence is capped (review-only) and promotion is blocked regardless
    # of every other gate.
    per_line_support = visible_error.get("per_line_support") or {}
    near_lines_unobservable = all(
        (per_line_support.get(name) or {}).get("status") == "unobservable" for name in ("near_baseline", "near_nvz")
    )
    if near_lines_unobservable and "near_floor_lines_unobservable" not in verification["blockers"]:
        verification["blockers"].append("near_floor_lines_unobservable")
        verification["promotion_allowed"] = False
    # Same fail-closed principle for temporal evidence: with fewer than 3
    # consensus frames there is no meaningful temporal-stability evidence
    # (DESIGN.md Stage 4 requires multi-frame agreement), and a single frame
    # of a tennis-overlay court can pass every line-support gate on the
    # tennis paint alone (measured on IMG_1605: 283px-wrong fit, zero
    # blockers). Promotion is blocked; review-only confidence stands.
    if int(temporal_stability.get("frame_count") or 0) < 3 and "insufficient_temporal_frames" not in verification["blockers"]:
        verification["blockers"].append("insufficient_temporal_frames")
        verification["promotion_allowed"] = False

    runner_up: list[dict[str, Any]] = []
    for item in per_frame_results:
        if item["tennis_best"] is not None:
            runner_up.append(
                {
                    "frame_index": item["frame_index"],
                    "template": "tennis_overlay",
                    "score": round(float(item["tennis_best"]["score"]), 4),
                    "evidence_score": round(float(item["tennis_best"]["evidence_score"]), 4),
                    "keypoints": {name: list(xy) for name, xy in item["tennis_best"]["keypoints"].items()},
                }
            )
    runner_up.sort(key=lambda item: item["score"])
    # ROUND-2 FIX 5 artifact requirement: rejected/runner-up PICKLEBALL court
    # candidates are exposed too (adjacent-court review support).
    pickleball_runner_ups: list[dict[str, Any]] = []
    for item in retained_frames:
        candidate = item.get("pickleball_runner_up")
        if candidate is not None:
            pickleball_runner_ups.append(
                {
                    "frame_index": item["frame_index"],
                    "template": "pickleball_runner_up",
                    "score": round(float(candidate["score"]), 4),
                    "evidence_score": round(float(candidate["evidence_score"]), 4),
                    "keypoints": {name: list(xy) for name, xy in candidate["keypoints"].items()},
                }
            )
    pickleball_runner_ups.sort(key=lambda item: item["score"])

    pixel_support = (representative["best"].get("score_components") or {}).get("projected_pixel_support") or {}
    joint_competition = (representative["best"].get("score_components") or {}).get("joint_template_competition") or {}
    proposal = CourtProposal(
        proposal_id="proposal_detector_v2_multiframe_0001",
        source="detector_v2_multiframe",
        court_keypoints={name: (float(xy[0]), float(xy[1])) for name, xy in consensus_keypoints.items()},
        scores={
            "overall": round(float(representative["best"]["evidence_score"]), 4),
            "pickleball_template": round(float(representative["best"]["evidence_score"]), 4),
            "tennis_template": float(runner_up[0]["evidence_score"]) if runner_up else None,
            "template_margin": joint_competition.get("margin"),
            "line_support": round(min(1.0, line_support["semantic_line_count"] / 8.0), 4),
            "mask_support": pixel_support.get("mean_line_pixel_support_ratio"),
            "net_consistency": 1.0 if top_net_validation.get("passed") else 0.0,
            "temporal_jitter_px_p95": temporal_stability.get("p95"),
            "reprojection_px_median": (visible_error.get("floor_visible") or {}).get("median"),
            "reprojection_px_p95": (visible_error.get("floor_visible") or {}).get("p95"),
            "worst_corner_px": (visible_error.get("visible_corners") or {}).get("median"),
        },
        gate={
            "auto_usable": bool(verification["promotion_allowed"]),
            "review_usable": True,
            "failed": [] if verification["promotion_allowed"] else list(verification["blockers"]),
            "warnings": [],
        },
        evidence={
            "verification": verification,
            "representative_frame_index": representative["frame_index"],
            "frames_evaluated": len(per_frame_results),
            "frames_with_pickleball_hypothesis": len(frames_with_hypothesis),
            "geor3": geor3_vote,
            "r2_temporal_stability_px": r2_temporal_stability,
            "runner_up_hypotheses": runner_up[:5],
            "pickleball_runner_up_hypotheses": pickleball_runner_ups[:5],
            "per_expected_line_support": per_line_support,
            "internal_support_score": visible_error.get("supported_fraction"),
            "near_floor_lines_unobservable": near_lines_unobservable,
            "persistent_line_bank_segment_count": len(persistent_segments),
            "frame_selection": _json_safe_selection(selection),
            "tracks_used": tracks is not None,
            "fallback_used": False,
            "fallback_threshold": FALLBACK_INTERNAL_SUPPORT_THRESHOLD,
        },
    )

    # ROUND-2 FIX 6: selector fallback safety net (predeclared threshold,
    # see FALLBACK_INTERNAL_SUPPORT_THRESHOLD). The internal support score is
    # label-free: fraction of observable expected template lines supported by
    # the persistent line bank for the CONSENSUS keypoints.
    internal_support_score = visible_error.get("supported_fraction")
    proposals = [proposal]
    if internal_support_score is None or float(internal_support_score) < FALLBACK_INTERNAL_SUPPORT_THRESHOLD:
        fallback = _hough_keypoints_fallback_proposal(
            representative["frame"],
            reason="internal_support_score_below_threshold",
            internal_support_score=None if internal_support_score is None else float(internal_support_score),
        )
        if fallback is not None:
            proposal.evidence["fallback_used"] = True
            proposals = [fallback, proposal]

    report = CourtProposalReport(
        clip=clip,
        video=str(video_path),
        image_size=image_size,
        frame_indices=[item["frame_index"] for item in selected],
        proposals=proposals,
        motion_mode="static",
    )
    return report.to_json_dict()


def _json_safe_selection(selection: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_count": selection.get("candidate_count"),
        "candidate_frame_indices": list(selection.get("candidate_frame_indices") or []),
        "rejected_blur_frame_indices": list(selection.get("rejected_blur_frame_indices") or []),
        "max_sharpness": selection.get("max_sharpness"),
        "selected_frame_indices": [item["frame_index"] for item in selection.get("selected") or []],
    }


def _load_tracks_feet(tracks_path: str | Path) -> list[tuple[float, float]] | None:
    from .net_anchor_court import load_player_foot_points_from_tracks

    try:
        return [tuple(point) for point in load_player_foot_points_from_tracks(tracks_path)]
    except (OSError, ValueError, TypeError, KeyError) as exc:
        _ = exc
        return None


def _best_by_net_and_feet(
    hypotheses: list[dict[str, Any]],
    *,
    net_evidence: Mapping[str, Any] | None,
    tracks: list[tuple[float, float]] | None,
    top_n: int = 5,
    score_band: float = 40.0,
    min_net_confidence: float = 0.45,
) -> dict[str, Any] | None:
    """Adjacent-court disambiguation: prefer the court containing the net ROI and feet mass.

    `hypotheses` is already sorted best-first by internal score. Net/feet
    containment is a genuine disambiguator ONLY among comparably-scored
    candidates. Round-2 hardening (measured failure): the tie-break now
    (a) only considers candidates within `score_band` of the best score --
    a 250-point-worse fit is not a "tie" no matter what it contains -- and
    (b) ignores the net-ROI term entirely when the net anchor itself is
    low-confidence, because a garbage net ROI (measured on Outdoor:
    x in [0, 529], pure misdetection) otherwise systematically vetoes the
    correct court on EVERY frame.
    """

    if not hypotheses:
        return None
    net_confidence = float((net_evidence or {}).get("confidence") or 0.0)
    use_net_roi = net_evidence is not None and net_confidence >= min_net_confidence
    if len(hypotheses) == 1 or (not use_net_roi and not tracks):
        return hypotheses[0]

    def containment_bonus(hypothesis: dict[str, Any]) -> float:
        bonus = 0.0
        keypoints = hypothesis.get("keypoints") or {}
        net_center = keypoints.get("net_center")
        roi = (net_evidence or {}).get("roi") if use_net_roi else None
        if roi and net_center is not None:
            if float(roi.get("x_min", 0)) - 40.0 <= net_center[0] <= float(roi.get("x_max", 0)) + 40.0:
                bonus += 6.0
        if tracks:
            near_left = keypoints.get("near_left_corner")
            far_right = keypoints.get("far_right_corner")
            if near_left is not None and far_right is not None:
                x_min, x_max = sorted((near_left[0], far_right[0]))
                y_min, y_max = sorted((near_left[1], far_right[1]))
                inside = sum(1 for fx, fy in tracks if x_min - 20 <= fx <= x_max + 20 and y_min - 20 <= fy <= y_max + 20)
                bonus += min(10.0, inside * 1.5)
        return bonus

    best_score = float(hypotheses[0].get("score") or 0.0)
    shortlist = [
        hypothesis
        for hypothesis in hypotheses[: max(1, top_n)]
        if float(hypothesis.get("score") or 0.0) <= best_score + score_band
    ] or [hypotheses[0]]
    return max(shortlist, key=containment_bonus)
