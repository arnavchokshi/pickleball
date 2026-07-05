"""Training-free net-anchored pickleball court proposal solver.

This module is intentionally proposal-only. It detects the physical net first, searches
for regulation paint evidence in constrained line families, and emits draggable corner
and keypoint proposals with confidence estimates. It never writes
``court_calibration.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .court_calibration import homography_from_planar_points, project_planar_points
from .court_line_bank import normalize_hough_lines_p
from .court_templates import ft_to_m
from .schemas import PICKLEBALL_COURT_KEYPOINT_NAMES

DEFAULT_USER_INPUT_CONFIDENCE_THRESHOLD = 0.70
HIGH_CONFIDENCE_THRESHOLD = 0.70

_HALF_WIDTH_M = ft_to_m(10.0)
_HALF_LENGTH_M = ft_to_m(22.0)
_NVZ_M = ft_to_m(7.0)

WORLD_XY_BY_NAME: dict[str, tuple[float, float]] = {
    "near_left_corner": (-_HALF_WIDTH_M, -_HALF_LENGTH_M),
    "near_baseline_center": (0.0, -_HALF_LENGTH_M),
    "near_right_corner": (_HALF_WIDTH_M, -_HALF_LENGTH_M),
    "far_right_corner": (_HALF_WIDTH_M, _HALF_LENGTH_M),
    "far_baseline_center": (0.0, _HALF_LENGTH_M),
    "far_left_corner": (-_HALF_WIDTH_M, _HALF_LENGTH_M),
    "near_nvz_left": (-_HALF_WIDTH_M, -_NVZ_M),
    "near_nvz_center": (0.0, -_NVZ_M),
    "near_nvz_right": (_HALF_WIDTH_M, -_NVZ_M),
    "net_left_sideline": (-_HALF_WIDTH_M, 0.0),
    "net_center": (0.0, 0.0),
    "net_right_sideline": (_HALF_WIDTH_M, 0.0),
    "far_nvz_left": (-_HALF_WIDTH_M, _NVZ_M),
    "far_nvz_center": (0.0, _NVZ_M),
    "far_nvz_right": (_HALF_WIDTH_M, _NVZ_M),
}

CORNER_ALIASES: dict[str, str] = {
    "near_left": "near_left_corner",
    "near_right": "near_right_corner",
    "far_right": "far_right_corner",
    "far_left": "far_left_corner",
}

KEYPOINT_LINE_FAMILIES: dict[str, tuple[str, ...]] = {
    "near_left_corner": ("near_baseline", "left_sideline"),
    "near_baseline_center": ("near_baseline", "centerline"),
    "near_right_corner": ("near_baseline", "right_sideline"),
    "far_right_corner": ("far_baseline", "right_sideline"),
    "far_baseline_center": ("far_baseline", "centerline"),
    "far_left_corner": ("far_baseline", "left_sideline"),
    "near_nvz_left": ("near_nvz", "left_sideline"),
    "near_nvz_center": ("near_nvz", "centerline"),
    "near_nvz_right": ("near_nvz", "right_sideline"),
    "net_left_sideline": ("net", "left_sideline"),
    "net_center": ("net", "centerline"),
    "net_right_sideline": ("net", "right_sideline"),
    "far_nvz_left": ("far_nvz", "left_sideline"),
    "far_nvz_center": ("far_nvz", "centerline"),
    "far_nvz_right": ("far_nvz", "right_sideline"),
}

LINE_STAGE_BY_NAME = {
    "net": "net",
    "near_nvz": "kitchen",
    "far_nvz": "kitchen",
    "left_sideline": "sideline",
    "right_sideline": "sideline",
    "centerline": "sideline",
    "near_baseline": "baseline",
    "far_baseline": "baseline",
}
TOP_NET_KEYPOINT_NAMES = {"net_left_sideline", "net_center", "net_right_sideline"}


@dataclass(frozen=True)
class NetDetection:
    tape_line: tuple[tuple[float, float], tuple[float, float]]
    post_tops: tuple[tuple[float, float], tuple[float, float]]
    post_bases: tuple[tuple[float, float], tuple[float, float]]
    confidence: float
    evidence: dict[str, float | int]


@dataclass(frozen=True)
class LineEvidence:
    name: str
    endpoints: tuple[tuple[float, float], tuple[float, float]]
    confidence: float
    support_px: int
    source_stage: str
    color_bgr: list[float]


@dataclass(frozen=True)
class ProposalPoint:
    xy: list[float]
    confidence: float
    evidence: dict[str, int]
    residual_px: float


@dataclass(frozen=True)
class RefinementDecision:
    accepted: bool
    reason: str
    homography: list[list[float]]
    median_residual_px: float
    residuals_by_name: dict[str, float]


@dataclass(frozen=True)
class _Segment:
    p1: tuple[float, float]
    p2: tuple[float, float]
    angle: float
    length: float
    color_bgr: list[float]


@dataclass(frozen=True)
class _LineCluster:
    line: tuple[float, float, float]
    endpoints: tuple[tuple[float, float], tuple[float, float]]
    support_px: int
    confidence: float
    color_bgr: list[float]
    signed_distance: float


def project_standard_keypoints(homography: Sequence[Sequence[float]]) -> dict[str, list[float]]:
    world = [[WORLD_XY_BY_NAME[name][0], WORLD_XY_BY_NAME[name][1], 0.0] for name in PICKLEBALL_COURT_KEYPOINT_NAMES]
    projected = project_planar_points(homography, world)
    return {
        name: [float(xy[0]), float(xy[1])]
        for name, xy in zip(PICKLEBALL_COURT_KEYPOINT_NAMES, projected, strict=True)
    }


def accept_candidate_correspondences(
    base_correspondences: Mapping[str, Sequence[float]],
    candidate_correspondences: Mapping[str, Sequence[float]],
    *,
    residual_gate_px: float = 8.0,
) -> RefinementDecision:
    """Accept a new line/correspondence set only if it preserves geometric residuals."""

    if residual_gate_px <= 0.0:
        raise ValueError("residual_gate_px must be positive")
    if len(base_correspondences) < 4:
        raise ValueError("at least 4 base correspondences are required")
    combined = {**base_correspondences, **candidate_correspondences}
    homography = _homography_from_named_image_points(combined)
    projected = project_standard_keypoints(homography)
    residuals = {
        name: _distance(_xy(value), _xy(projected[name]))
        for name, value in combined.items()
        if name in projected
    }
    base_residuals = [residuals[name] for name in base_correspondences if name in residuals]
    candidate_residuals = [residuals[name] for name in candidate_correspondences if name in residuals]
    median_base = _median(base_residuals)
    median_all = _median(list(residuals.values()))
    if median_base > residual_gate_px or (candidate_residuals and _median(candidate_residuals) > residual_gate_px):
        return RefinementDecision(
            accepted=False,
            reason="candidate_worsens_residual",
            homography=homography,
            median_residual_px=median_all,
            residuals_by_name=residuals,
        )
    return RefinementDecision(
        accepted=True,
        reason="accepted",
        homography=homography,
        median_residual_px=median_all,
        residuals_by_name=residuals,
    )


def solve_net_anchor_court_from_frame(
    frame_bgr: Any,
    *,
    clip_id: str = "",
    player_foot_points: Sequence[Sequence[float]] = (),
    confidence_threshold: float = DEFAULT_USER_INPUT_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Solve one median/player-suppressed frame into a proposal artifact."""

    frame = _as_bgr_array(frame_bgr)
    height, width = int(frame.shape[0]), int(frame.shape[1])
    net = detect_net_anchor(frame)
    segments = _detect_line_segments(frame)
    line_evidence, detected_points = _detect_ladder_evidence(frame, segments, net)
    player_prior = _summarize_player_feet_prior(player_foot_points)
    line_directions = cluster_ground_line_directions([(segment.p1, segment.p2) for segment in segments])

    notes: list[str] = [
        "training_free_geometry_solver",
        "proposal_only_never_writes_court_calibration_json",
    ]

    hypotheses = _build_global_fit_hypotheses(
        detected_points=detected_points,
        net=net,
        line_evidence=line_evidence,
        image_size=(width, height),
        player_prior=player_prior,
    )
    best_hypothesis = hypotheses[0]
    homography = best_hypothesis["homography"]
    residuals_by_name = best_hypothesis["residuals_by_name"]
    accepted_names = best_hypothesis["accepted_correspondence_names"]
    if not accepted_names:
        notes.append("insufficient_paint_evidence_fallback_from_net_only")
    if player_prior["point_count"] > 0:
        notes.append("player_feet_ground_prior")

    artifact = build_proposals_from_homography(
        homography,
        image_size=(width, height),
        net=net,
        line_evidence=line_evidence,
        accepted_correspondence_names=accepted_names,
        residuals_by_name=residuals_by_name,
        confidence_threshold=confidence_threshold,
    )
    artifact["source"] = {
        **artifact["source"],
        "clip_id": clip_id,
        "frame_role": "player_suppressed_or_single_frame",
    }
    artifact["hypotheses"] = [_hypothesis_for_artifact(hypothesis) for hypothesis in hypotheses]
    artifact["vanishing_point_rotation_constraint"] = {
        "direction_families": line_directions,
        "source": "hough_ground_segments",
    }
    artifact["player_feet_prior"] = player_prior
    artifact["self_verification"] = _self_verification_for_hypothesis(best_hypothesis)
    confidence_cap, cap_reason = _confidence_cap_for_hypothesis(best_hypothesis)
    if confidence_cap < 1.0:
        _apply_hypothesis_confidence_cap(artifact, max_confidence=confidence_cap, reason=cap_reason)
    artifact["notes"].extend(notes)
    return artifact


def build_proposals_from_homography(
    homography: Sequence[Sequence[float]],
    *,
    image_size: tuple[int, int],
    net: NetDetection,
    line_evidence: Mapping[str, LineEvidence],
    accepted_correspondence_names: Sequence[str],
    residuals_by_name: Mapping[str, float],
    confidence_threshold: float = DEFAULT_USER_INPUT_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    width, height = int(image_size[0]), int(image_size[1])
    projected = project_standard_keypoints(homography)
    accepted = set(accepted_correspondence_names)
    line_conf_by_stage = _line_confidence_by_stage(line_evidence)
    keypoints: dict[str, dict[str, Any]] = {}

    for name in PICKLEBALL_COURT_KEYPOINT_NAMES:
        xy = projected[name]
        residual = float(residuals_by_name.get(name, _default_residual_for_point(name, line_evidence)))
        evidence_counts = _evidence_counts_for_keypoint(name, line_evidence, accepted)
        stage_count = sum(1 for value in evidence_counts.values() if value > 0)
        residual_score = _clamp01(1.0 - residual / 35.0)
        inside_score = 1.0 if -0.1 * width <= xy[0] <= 1.1 * width and -0.1 * height <= xy[1] <= 1.1 * height else 0.35
        line_stage_score = 0.0
        for stage in ("kitchen", "sideline", "baseline"):
            line_stage_score += line_conf_by_stage.get(stage, 0.0)
        line_stage_score = _clamp01(line_stage_score / 2.0)
        accepted_score = 1.0 if name in accepted else min(0.65, 0.20 * stage_count)
        confidence = _clamp01(
            0.24 * float(net.confidence)
            + 0.25 * residual_score
            + 0.20 * line_stage_score
            + 0.16 * accepted_score
            + 0.15 * inside_score
        )
        if not line_evidence:
            confidence = min(confidence, 0.42)
        keypoints[name] = ProposalPoint(
            xy=[float(xy[0]), float(xy[1])],
            confidence=round(confidence, 4),
            evidence=evidence_counts,
            residual_px=round(residual, 4),
        ).__dict__

    corners = {
        alias: keypoints[keypoint_name]
        for alias, keypoint_name in CORNER_ALIASES.items()
    }
    corner_confidences = [float(point["confidence"]) for point in corners.values()]
    solver_confidence = _median(corner_confidences) if corner_confidences else 0.0
    needs_user_input = [
        alias
        for alias, point in corners.items()
        if float(point["confidence"]) < confidence_threshold
    ]

    return {
        "schema_version": 1,
        "artifact_type": "racketsport_net_anchor_court_proposals",
        "source": {
            "image_size": [width, height],
        },
        "solver": {
            "name": "net_anchor_court",
            "version": 1,
            "strategy": "multi_hypothesis_global_fit_v2",
            "training_free": True,
            "writes_court_calibration": False,
            "confidence_threshold": confidence_threshold,
        },
        "homography": [[float(value) for value in row] for row in homography],
        "net": {
            "tape_line": _line_to_json(net.tape_line),
            "post_tops": _line_to_json(net.post_tops),
            "post_bases": _line_to_json(net.post_bases),
            "confidence": round(float(net.confidence), 4),
            "evidence": dict(net.evidence),
        },
        "line_evidence": {
            name: {
                "endpoints": _line_to_json(line.endpoints),
                "confidence": round(float(line.confidence), 4),
                "support_px": int(line.support_px),
                "source_stage": line.source_stage,
                "color_bgr": [round(float(v), 3) for v in line.color_bgr],
            }
            for name, line in sorted(line_evidence.items())
        },
        "corners": corners,
        "keypoints": keypoints,
        "solver_confidence": round(float(solver_confidence), 4),
        "needs_user_input": needs_user_input,
        "needs_user_confirmation": bool(needs_user_input),
        "self_verification": {
            "status": "unchecked",
            "reasons": [],
            "visible_median_error_px": None,
            "internal_median_residual_px": None,
            "promotion_allowed": False,
        },
        "notes": [],
    }


def _apply_hypothesis_confidence_cap(
    artifact: dict[str, Any],
    *,
    max_confidence: float,
    reason: str,
) -> None:
    cap = _clamp01(max_confidence)
    for section in ("corners", "keypoints"):
        for point in artifact.get(section, {}).values():
            point["confidence"] = round(min(float(point.get("confidence", 0.0)), cap), 4)
    corners = artifact.get("corners", {})
    corner_confidences = [float(point.get("confidence", 0.0)) for point in corners.values()]
    solver_confidence = _median(corner_confidences) if corner_confidences else 0.0
    artifact["solver_confidence"] = round(float(solver_confidence), 4)
    threshold = float(artifact.get("solver", {}).get("confidence_threshold", DEFAULT_USER_INPUT_CONFIDENCE_THRESHOLD))
    artifact["needs_user_input"] = [
        alias
        for alias, point in corners.items()
        if float(point.get("confidence", 0.0)) < threshold
    ]
    artifact["needs_user_confirmation"] = bool(artifact["needs_user_input"])
    artifact.setdefault("confidence_caps", []).append({"max_confidence": round(cap, 4), "reason": reason})
    _append_self_verification_failure(artifact, reason=reason)
    if reason not in artifact.setdefault("notes", []):
        artifact["notes"].append(reason)


def _confidence_cap_for_hypothesis(hypothesis: Mapping[str, Any]) -> tuple[float, str]:
    name = str(hypothesis.get("name", ""))
    residual = float(hypothesis.get("median_residual_px", 100.0))
    if name != "all_detected_correspondences":
        return 0.30, "seed_only_hypothesis_not_globally_verified"
    if residual > 18.0:
        return 0.50, "global_fit_residual_too_high"
    return 1.0, "globally_verified"


def _self_verification_for_hypothesis(hypothesis: Mapping[str, Any]) -> dict[str, Any]:
    name = str(hypothesis.get("name", ""))
    residual = float(hypothesis.get("median_residual_px", 100.0))
    reasons: list[str] = []
    if name != "all_detected_correspondences":
        reasons.append("seed_only_hypothesis_not_globally_verified")
    if name == "all_detected_correspondences" and residual > 18.0:
        reasons.append("global_fit_residual_too_high")
    status = "failed" if reasons else "passed_internal_geometry"
    return {
        "status": status,
        "reasons": reasons,
        "visible_median_error_px": None,
        "internal_median_residual_px": round(float(residual), 4),
        "promotion_allowed": status != "failed",
    }


def _append_self_verification_failure(artifact: dict[str, Any], *, reason: str) -> None:
    verification = artifact.setdefault(
        "self_verification",
        {
            "status": "failed",
            "reasons": [],
            "visible_median_error_px": None,
            "internal_median_residual_px": None,
            "promotion_allowed": False,
        },
    )
    verification["status"] = "failed"
    verification["promotion_allowed"] = False
    reasons = verification.setdefault("reasons", [])
    if reason not in reasons:
        reasons.append(reason)


def detect_net_anchor(frame_bgr: Any) -> NetDetection:
    frame = _as_bgr_array(frame_bgr)
    cv2, np = _cv2_np()
    height, width = int(frame.shape[0]), int(frame.shape[1])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 160)
    min_len = max(80, int(width * 0.20))
    raw = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=45, minLineLength=min_len, maxLineGap=18)
    best: tuple[float, tuple[tuple[float, float], tuple[float, float]], dict[str, float | int]] | None = None
    for item in normalize_hough_lines_p(raw):
        x1, y1, x2, y2 = item
        length = math.hypot(x2 - x1, y2 - y1)
        if length < min_len:
            continue
        angle = abs(_angle_distance(math.atan2(y2 - y1, x2 - x1), 0.0))
        if angle > math.radians(14.0):
            continue
        y_mid = (y1 + y2) / 2.0
        if not (height * 0.12 <= y_mid <= height * 0.88):
            continue
        tape_brightness = _sample_segment_gray(gray, (x1, y1), (x2, y2))
        below_offset = max(5.0, height * 0.018)
        below_brightness = _sample_segment_gray(gray, (x1, y1 + below_offset), (x2, y2 + below_offset))
        mesh_contrast = max(0.0, (tape_brightness - below_brightness) / 255.0)
        vertical_support = _post_support(edges, (x1, y1), (x2, y2), height)
        score = (length / width) * (0.55 + mesh_contrast) * (0.75 + 0.25 * vertical_support)
        if best is None or score > best[0]:
            best = (
                score,
                ((x1, y1), (x2, y2)),
                {
                    "support_px": int(round(length)),
                    "mesh_band_contrast": round(mesh_contrast, 4),
                    "post_count": int(round(vertical_support * 2.0)),
                    "tape_brightness": round(tape_brightness, 3),
                },
            )

    if best is None:
        y = height * 0.50
        endpoints = ((width * 0.30, y), (width * 0.70, y))
        return NetDetection(
            tape_line=endpoints,
            post_tops=endpoints,
            post_bases=((endpoints[0][0], y + height * 0.08), (endpoints[1][0], y + height * 0.08)),
            confidence=0.18,
            evidence={"support_px": 0, "mesh_band_contrast": 0.0, "post_count": 0},
        )

    endpoints = _sort_line_left_to_right(best[1])
    y_offset = max(20.0, height * 0.08)
    confidence = _clamp01(0.22 + 0.52 * min(1.0, best[2]["support_px"] / max(1.0, width * 0.55)) + 0.26 * float(best[2]["mesh_band_contrast"]))
    return NetDetection(
        tape_line=endpoints,
        post_tops=endpoints,
        post_bases=((endpoints[0][0], endpoints[0][1] + y_offset), (endpoints[1][0], endpoints[1][1] + y_offset)),
        confidence=confidence,
        evidence=best[2],
    )


def draw_net_anchor_overlay(frame_bgr: Any, artifact: Mapping[str, Any]) -> Any:
    frame = _as_bgr_array(frame_bgr).copy()
    cv2, _np = _cv2_np()
    line_colors = {
        "net": (255, 255, 255),
        "kitchen": (0, 220, 255),
        "sideline": (40, 220, 80),
        "baseline": (255, 120, 40),
    }
    for line in artifact.get("line_evidence", {}).values():
        p1, p2 = line["endpoints"]
        stage = line.get("source_stage", "")
        color = line_colors.get(stage, (180, 180, 180))
        cv2.line(frame, _int_point(p1), _int_point(p2), color, 3, cv2.LINE_AA)
    for alias, point in artifact.get("corners", {}).items():
        xy = point.get("xy", [0.0, 0.0])
        conf = float(point.get("confidence", 0.0))
        color = (40, 220, 80) if conf >= DEFAULT_USER_INPUT_CONFIDENCE_THRESHOLD else (0, 160, 255)
        cv2.circle(frame, _int_point(xy), 9, color, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"{alias}:{conf:.2f}",
            (int(round(xy[0] + 10)), int(round(xy[1] - 8))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    for point in artifact.get("keypoints", {}).values():
        xy = point.get("xy", [0.0, 0.0])
        cv2.circle(frame, _int_point(xy), 4, (255, 255, 0), -1, cv2.LINE_AA)
    return frame


def load_player_suppressed_frame(
    input_path: str | Path,
    *,
    max_frames: int = 72,
    stride: int = 6,
    start_frame: int = 0,
) -> tuple[Any, dict[str, Any]]:
    """Load an image, a directory of frames, or a video into a median frame."""

    cv2, np = _cv2_np()
    path = Path(input_path)
    if path.is_dir():
        frames = []
        for frame_path in sorted(path.glob("*.jpg"))[:max_frames]:
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is not None:
                frames.append(image)
        if not frames:
            raise ValueError(f"no readable .jpg frames in {path}")
        return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8), {
            "input_kind": "frame_directory",
            "sampled_frames": len(frames),
        }
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is not None:
        return image, {"input_kind": "image", "sampled_frames": 1}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open input as image/video: {path}")
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_idx = max(0, int(start_frame))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if (frame_idx - start_frame) % max(1, stride) == 0:
            frames.append(frame)
        frame_idx += 1
    cap.release()
    if not frames:
        raise ValueError(f"no frames decoded from video: {path}")
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8), {
        "input_kind": "video",
        "sampled_frames": len(frames),
        "video_frame_count": total,
        "start_frame": start_frame,
        "stride": stride,
    }


def load_player_foot_points_from_tracks(
    tracks_path: str | Path,
    *,
    min_confidence: float = 0.0,
) -> list[list[float]]:
    """Load image-space player footpoints from a tracks artifact.

    The current tracking artifact stores boxes as ``[x1, y1, x2, y2]``. The
    bottom midpoint is a weak image-space proxy for the ground contact region.
    """

    payload = json.loads(Path(tracks_path).read_text(encoding="utf-8"))
    points: list[list[float]] = []
    for player in payload.get("players", []):
        if not isinstance(player, Mapping):
            continue
        for frame in player.get("frames", []):
            if not isinstance(frame, Mapping):
                continue
            if float(frame.get("conf", 1.0)) < min_confidence:
                continue
            bbox = frame.get("bbox")
            if not isinstance(bbox, Sequence) or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [float(value) for value in bbox]
            if x2 < x1 or y2 < y1:
                continue
            points.append([(x1 + x2) / 2.0, y2])
    return points


def cluster_ground_line_directions(raw_segments: Sequence[Sequence[Sequence[float]]]) -> list[dict[str, Any]]:
    """Cluster raw ground-line segments into two dominant vanishing directions."""

    clusters: list[dict[str, Any]] = []
    for raw in raw_segments:
        if len(raw) != 2 or len(raw[0]) != 2 or len(raw[1]) != 2:
            continue
        x1, y1 = float(raw[0][0]), float(raw[0][1])
        x2, y2 = float(raw[1][0]), float(raw[1][1])
        length = math.hypot(x2 - x1, y2 - y1)
        if length <= 1e-6:
            continue
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
        placed = False
        for cluster in clusters:
            delta = min(abs(angle - cluster["angle_deg"]), 180.0 - abs(angle - cluster["angle_deg"]))
            if delta <= 15.0:
                old_support = float(cluster["support_px"])
                new_support = old_support + length
                cluster["angle_deg"] = (float(cluster["angle_deg"]) * old_support + angle * length) / new_support
                cluster["support_px"] = new_support
                cluster["segment_count"] = int(cluster["segment_count"]) + 1
                placed = True
                break
        if not placed:
            clusters.append({"angle_deg": angle, "support_px": length, "segment_count": 1})
    clusters = sorted(clusters, key=lambda item: float(item["support_px"]), reverse=True)
    dominant: list[dict[str, Any]] = []
    for cluster in clusters:
        if all(
            min(abs(float(cluster["angle_deg"]) - float(existing["angle_deg"])), 180.0 - abs(float(cluster["angle_deg"]) - float(existing["angle_deg"]))) > 25.0
            for existing in dominant
        ):
            dominant.append(cluster)
        if len(dominant) == 2:
            break
    return [
        {
            "angle_deg": round(float(cluster["angle_deg"]), 3),
            "support_px": round(float(cluster["support_px"]), 3),
            "segment_count": int(cluster["segment_count"]),
        }
        for cluster in dominant
    ]


def score_corner_proposals(
    artifact: Mapping[str, Any],
    gt_corners: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    rows = []
    high_errors = []
    high_wrong = 0
    for alias, keypoint_name in CORNER_ALIASES.items():
        if alias not in gt_corners:
            continue
        proposal = artifact["corners"][alias]
        pred = _xy(proposal["xy"])
        gt = _xy(gt_corners[alias])
        error = _distance(pred, gt)
        confidence = float(proposal["confidence"])
        high = confidence > HIGH_CONFIDENCE_THRESHOLD
        if high:
            high_errors.append(error)
            if error > 30.0:
                high_wrong += 1
        rows.append(
            {
                "corner": alias,
                "keypoint": keypoint_name,
                "pred_xy": [pred[0], pred[1]],
                "gt_xy": [gt[0], gt[1]],
                "error_px": round(error, 4),
                "confidence": confidence,
                "high_confidence": high,
            }
        )
    high_count = len(high_errors)
    high_median = _median(high_errors) if high_errors else None
    high_wrong_rate = (high_wrong / high_count) if high_count else 0.0
    kill = bool(high_count and (float(high_median) > 15.0 or high_wrong_rate > 0.10))
    return {
        "corner_errors": rows,
        "high_confidence_corner_count": high_count,
        "high_confidence_median_error_px": None if high_median is None else round(float(high_median), 4),
        "high_confidence_over_30px_count": high_wrong,
        "high_confidence_over_30px_rate": round(high_wrong_rate, 4),
        "kill_criterion_triggered": kill,
        "verdict": "mandatory_user_confirmation_only" if kill or high_count == 0 else "proposal_generator_not_tap_replacement",
    }


def _build_global_fit_hypotheses(
    *,
    detected_points: Mapping[str, Sequence[float]],
    net: NetDetection,
    line_evidence: Mapping[str, LineEvidence],
    image_size: tuple[int, int],
    player_prior: Mapping[str, Any],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    candidates: list[tuple[str, Mapping[str, Sequence[float]]]] = []
    if len(detected_points) >= 4:
        candidates.append(("all_detected_correspondences", detected_points))
        required_corners = {
            name: detected_points[name]
            for name in ("near_left_corner", "near_right_corner", "far_right_corner", "far_left_corner")
            if name in detected_points
        }
        if len(required_corners) == 4:
            candidates.append(("corner_subset_refinement_seed", required_corners))
    fallback = {
        name: xy
        for name, xy in project_standard_keypoints(_fallback_homography_from_net(net, image_size)).items()
        if name in ("near_left_corner", "near_right_corner", "far_right_corner", "far_left_corner")
    }
    candidates.append(("net_only_fallback", fallback))

    hypotheses = []
    seen: set[str] = set()
    for name, correspondences in candidates:
        if len(correspondences) < 4:
            continue
        signature = ",".join(sorted(correspondences))
        if signature in seen and name != "net_only_fallback":
            continue
        seen.add(signature)
        floor_correspondences = _floor_correspondences(correspondences)
        if len(floor_correspondences) < 4:
            continue
        try:
            homography = _homography_from_named_image_points(floor_correspondences)
        except ValueError:
            continue
        projected = project_standard_keypoints(homography)
        is_fallback = name == "net_only_fallback"
        is_seed_only = name == "corner_subset_refinement_seed"
        residuals = (
            {}
            if is_fallback or is_seed_only
            else {
                point_name: _distance(_xy(detected_xy), _xy(projected[point_name]))
                for point_name, detected_xy in floor_correspondences.items()
                if point_name in projected
            }
        )
        if is_fallback:
            median_residual = 100.0
        elif is_seed_only:
            median_residual = 42.0
        else:
            median_residual = _median(list(residuals.values())) if residuals else 100.0
        evidence_mass = _line_evidence_mass(line_evidence)
        player_bonus = min(0.35, 0.04 * int(player_prior.get("point_count", 0)))
        residual_score = math.exp(-((median_residual / 18.0) ** 2))
        score = evidence_mass + float(net.confidence) + residual_score + player_bonus
        hypotheses.append(
            {
                "name": name,
                "score": score,
                "evidence_mass": evidence_mass,
                "player_feet_bonus": player_bonus,
                "median_residual_px": median_residual,
                "homography": homography,
                "accepted_correspondence_names": [] if is_fallback or is_seed_only else sorted(floor_correspondences),
                "residuals_by_name": residuals,
                "pose_seed": {
                    "focal_px": round(float(image_size[0]) * 0.95, 3),
                    "tilt_deg": 0.0,
                    "roll_deg": round(math.degrees(math.atan2(net.tape_line[1][1] - net.tape_line[0][1], net.tape_line[1][0] - net.tape_line[0][0])), 3),
                    "source": "bounded_grid_seed_from_net_endpoints",
                },
            }
        )

    hypotheses.sort(key=lambda hypothesis: (float(hypothesis["score"]), -float(hypothesis["median_residual_px"])), reverse=True)
    if not hypotheses:
        homography = _fallback_homography_from_net(net, image_size)
        hypotheses.append(
            {
                "name": "net_only_fallback",
                "score": float(net.confidence),
                "evidence_mass": 0.0,
                "player_feet_bonus": 0.0,
                "median_residual_px": 100.0,
                "homography": homography,
                "accepted_correspondence_names": [],
                "residuals_by_name": {},
                "pose_seed": {"focal_px": round(float(image_size[0]) * 0.95, 3), "tilt_deg": 0.0, "roll_deg": 0.0, "source": "fallback"},
            }
        )
    return hypotheses[:top_k]


def _hypothesis_for_artifact(hypothesis: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": str(hypothesis["name"]),
        "score": round(float(hypothesis["score"]), 4),
        "evidence_mass": round(float(hypothesis["evidence_mass"]), 4),
        "player_feet_bonus": round(float(hypothesis.get("player_feet_bonus", 0.0)), 4),
        "median_residual_px": round(float(hypothesis["median_residual_px"]), 4),
        "accepted_correspondence_names": list(hypothesis["accepted_correspondence_names"]),
        "pose_seed": dict(hypothesis["pose_seed"]),
        "refinement": {
            "method": "least_squares_homography_over_accumulated_correspondences",
            "residual_px": round(float(hypothesis["median_residual_px"]), 4),
        },
    }


def _line_evidence_mass(line_evidence: Mapping[str, LineEvidence]) -> float:
    mass = 0.0
    for line in line_evidence.values():
        stage_weight = 0.55 if line.name == "net" else 1.0
        mass += stage_weight * float(line.confidence) * min(1.0, max(0.0, float(line.support_px) / 700.0))
    return round(mass, 4)


def _summarize_player_feet_prior(player_foot_points: Sequence[Sequence[float]]) -> dict[str, Any]:
    points = [[float(point[0]), float(point[1])] for point in player_foot_points if len(point) == 2]
    if not points:
        return {
            "point_count": 0,
            "ground_region_bbox": None,
            "centroid": None,
            "confidence": 0.0,
            "source": "tracks_json_optional_absent",
        }
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return {
        "point_count": len(points),
        "ground_region_bbox": [min(xs), min(ys), max(xs), max(ys)],
        "centroid": [sum(xs) / len(xs), sum(ys) / len(ys)],
        "confidence": round(min(0.35, 0.04 * len(points)), 4),
        "source": "tracks_json_bbox_bottom_midpoints",
    }


def _detect_ladder_evidence(
    frame: Any,
    segments: Sequence[_Segment],
    net: NetDetection,
) -> tuple[dict[str, LineEvidence], dict[str, list[float]]]:
    height, width = int(frame.shape[0]), int(frame.shape[1])
    net_line = _line_from_points(net.tape_line[0], net.tape_line[1])
    net_angle = math.atan2(net.tape_line[1][1] - net.tape_line[0][1], net.tape_line[1][0] - net.tape_line[0][0])
    cross_clusters = _cluster_crosscourt_segments(segments, net_angle, net_line, image_height=height)

    line_evidence: dict[str, LineEvidence] = {
        "net": LineEvidence(
            name="net",
            endpoints=net.tape_line,
            confidence=net.confidence,
            support_px=int(net.evidence.get("support_px", _line_length(net.tape_line))),
            source_stage="net",
            color_bgr=_sample_segment_bgr(frame, net.tape_line[0], net.tape_line[1]),
        )
    }
    selected = _select_crosscourt_clusters(cross_clusters, net_line)
    for name, cluster in selected.items():
        stage = LINE_STAGE_BY_NAME[name]
        line_evidence[name] = LineEvidence(
            name=name,
            endpoints=cluster.endpoints,
            confidence=cluster.confidence,
            support_px=cluster.support_px,
            source_stage=stage,
            color_bgr=cluster.color_bgr,
        )

    sideline_inputs = [
        selected[name]
        for name in ("near_baseline", "near_nvz", "far_nvz", "far_baseline")
        if name in selected
    ]
    if len(sideline_inputs) >= 2:
        left_line = _fit_line_through_ordered_endpoints(sideline_inputs, left=True, net_angle=net_angle)
        right_line = _fit_line_through_ordered_endpoints(sideline_inputs, left=False, net_angle=net_angle)
        if left_line is not None and right_line is not None:
            for name, cluster in (("left_sideline", left_line), ("right_sideline", right_line)):
                line_evidence[name] = LineEvidence(
                    name=name,
                    endpoints=cluster.endpoints,
                    confidence=cluster.confidence,
                    support_px=cluster.support_px,
                    source_stage="sideline",
                    color_bgr=cluster.color_bgr,
                )
            center_line = _interpolate_sideline(left_line, right_line, name="centerline")
            line_evidence["centerline"] = LineEvidence(
                name="centerline",
                endpoints=center_line.endpoints,
                confidence=center_line.confidence,
                support_px=center_line.support_px,
                source_stage="sideline",
                color_bgr=center_line.color_bgr,
            )

    detected_points = _intersections_from_line_evidence(line_evidence)
    if len(detected_points) < 4:
        # Conservative rescue for clean synthetic fixtures and simple clips: if we have four
        # crosscourt segments, use their ordered paint endpoints directly as keypoints.
        for line_name, left_name, right_name in (
            ("near_baseline", "near_left_corner", "near_right_corner"),
            ("near_nvz", "near_nvz_left", "near_nvz_right"),
            ("far_nvz", "far_nvz_left", "far_nvz_right"),
            ("far_baseline", "far_left_corner", "far_right_corner"),
        ):
            evidence = line_evidence.get(line_name)
            if evidence is None:
                continue
            left, right = _ordered_line_endpoints(evidence.endpoints, net_angle)
            detected_points.setdefault(left_name, [left[0], left[1]])
            detected_points.setdefault(right_name, [right[0], right[1]])
    return line_evidence, detected_points


def _detect_line_segments(frame: Any) -> list[_Segment]:
    cv2, np = _cv2_np()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 45, 145)
    height, width = gray.shape
    raw = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=38,
        minLineLength=max(28, int(width * 0.045)),
        maxLineGap=max(12, int(width * 0.018)),
    )
    segments: list[_Segment] = []
    for item in normalize_hough_lines_p(raw):
        x1, y1, x2, y2 = item
        length = math.hypot(x2 - x1, y2 - y1)
        if length < max(28.0, width * 0.045):
            continue
        segments.append(
            _Segment(
                p1=(x1, y1),
                p2=(x2, y2),
                angle=math.atan2(y2 - y1, x2 - x1),
                length=length,
                color_bgr=_sample_segment_bgr(frame, (x1, y1), (x2, y2)),
            )
        )
    return segments


def _cluster_crosscourt_segments(
    segments: Sequence[_Segment],
    net_angle: float,
    net_line: tuple[float, float, float],
    *,
    image_height: int,
) -> list[_LineCluster]:
    candidates = [
        seg
        for seg in segments
        if abs(_angle_distance(seg.angle, net_angle)) <= math.radians(16.0)
    ]
    if not candidates:
        return []
    normal = _normal_with_positive_y(net_angle)
    sorted_segments = sorted(candidates, key=lambda seg: _dot(normal, _midpoint(seg.p1, seg.p2)))
    tolerance = max(8.0, image_height * 0.018)
    groups: list[list[_Segment]] = []
    group_ds: list[float] = []
    for segment in sorted_segments:
        d = _dot(normal, _midpoint(segment.p1, segment.p2))
        placed = False
        for idx, group_d in enumerate(group_ds):
            if abs(d - group_d) <= tolerance:
                groups[idx].append(segment)
                group_ds[idx] = (group_d * (len(groups[idx]) - 1) + d) / len(groups[idx])
                placed = True
                break
        if not placed:
            groups.append([segment])
            group_ds.append(d)

    clusters: list[_LineCluster] = []
    for group, signed_distance in zip(groups, group_ds, strict=True):
        support = int(round(sum(seg.length for seg in group)))
        if support < 70:
            continue
        points = [point for seg in group for point in (seg.p1, seg.p2)]
        line = _fit_line(points)
        endpoints = _endpoints_along_line(points, line, net_angle)
        color = _mean_color([seg.color_bgr for seg in group], weights=[seg.length for seg in group])
        clusters.append(
            _LineCluster(
                line=line,
                endpoints=endpoints,
                support_px=support,
                confidence=_clamp01(0.20 + min(0.60, support / 850.0)),
                color_bgr=color,
                signed_distance=_dot(normal, _midpoint(endpoints[0], endpoints[1])) - _line_signed_distance(net_line, normal),
            )
        )
    return clusters


def _select_crosscourt_clusters(
    clusters: Sequence[_LineCluster],
    net_line: tuple[float, float, float],
) -> dict[str, _LineCluster]:
    selected: dict[str, _LineCluster] = {}
    near = sorted([c for c in clusters if c.signed_distance > 10.0], key=lambda c: c.signed_distance)
    far = sorted([c for c in clusters if c.signed_distance < -10.0], key=lambda c: abs(c.signed_distance))
    if near:
        selected["near_nvz"] = near[0]
    if len(near) >= 2:
        selected["near_baseline"] = near[-1]
    if far:
        selected["far_nvz"] = far[0]
    if len(far) >= 2:
        selected["far_baseline"] = far[-1]

    # Suppress obvious duplicates where the nearest/farthest selector picked the same
    # painted stripe due to fragmented Hough support.
    for side in (("near_nvz", "near_baseline"), ("far_nvz", "far_baseline")):
        a, b = side
        if a in selected and b in selected and selected[a] is selected[b]:
            del selected[b]
    _ = net_line
    return selected


def _fit_line_through_ordered_endpoints(
    clusters: Sequence[_LineCluster],
    *,
    left: bool,
    net_angle: float,
) -> _LineCluster | None:
    points = []
    colors = []
    support = 0
    for cluster in clusters:
        left_pt, right_pt = _ordered_line_endpoints(cluster.endpoints, net_angle)
        points.append(left_pt if left else right_pt)
        colors.append(cluster.color_bgr)
        support += max(1, cluster.support_px // 4)
    if len(points) < 2:
        return None
    line = _fit_line(points)
    endpoints = _endpoints_along_points(points)
    return _LineCluster(
        line=line,
        endpoints=endpoints,
        support_px=support,
        confidence=_clamp01(0.24 + 0.14 * len(points)),
        color_bgr=_mean_color(colors),
        signed_distance=0.0,
    )


def _interpolate_sideline(left: _LineCluster, right: _LineCluster, *, name: str) -> _LineCluster:
    p1 = ((left.endpoints[0][0] + right.endpoints[0][0]) / 2.0, (left.endpoints[0][1] + right.endpoints[0][1]) / 2.0)
    p2 = ((left.endpoints[1][0] + right.endpoints[1][0]) / 2.0, (left.endpoints[1][1] + right.endpoints[1][1]) / 2.0)
    line = _line_from_points(p1, p2)
    _ = name
    return _LineCluster(
        line=line,
        endpoints=(p1, p2),
        support_px=(left.support_px + right.support_px) // 2,
        confidence=min(left.confidence, right.confidence, 0.55),
        color_bgr=_mean_color([left.color_bgr, right.color_bgr]),
        signed_distance=0.0,
    )


def _intersections_from_line_evidence(line_evidence: Mapping[str, LineEvidence]) -> dict[str, list[float]]:
    lines = {name: _line_from_points(value.endpoints[0], value.endpoints[1]) for name, value in line_evidence.items()}
    out: dict[str, list[float]] = {}
    pairs = {
        "near_left_corner": ("near_baseline", "left_sideline"),
        "near_baseline_center": ("near_baseline", "centerline"),
        "near_right_corner": ("near_baseline", "right_sideline"),
        "far_right_corner": ("far_baseline", "right_sideline"),
        "far_baseline_center": ("far_baseline", "centerline"),
        "far_left_corner": ("far_baseline", "left_sideline"),
        "near_nvz_left": ("near_nvz", "left_sideline"),
        "near_nvz_center": ("near_nvz", "centerline"),
        "near_nvz_right": ("near_nvz", "right_sideline"),
        "net_left_sideline": ("net", "left_sideline"),
        "net_center": ("net", "centerline"),
        "net_right_sideline": ("net", "right_sideline"),
        "far_nvz_left": ("far_nvz", "left_sideline"),
        "far_nvz_center": ("far_nvz", "centerline"),
        "far_nvz_right": ("far_nvz", "right_sideline"),
    }
    for name, (a, b) in pairs.items():
        if a in lines and b in lines:
            try:
                out[name] = list(_intersect_lines(lines[a], lines[b]))
            except ValueError:
                continue
    return out


def _fallback_homography_from_net(net: NetDetection, image_size: tuple[int, int]) -> list[list[float]]:
    width, height = image_size
    (x1, y1), (x2, y2) = net.tape_line
    net_mid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    tape_len = max(80.0, math.hypot(x2 - x1, y2 - y1))
    far_width = tape_len * 0.86
    near_width = min(width * 0.92, tape_len * 1.55)
    far_y = max(8.0, net_mid[1] - height * 0.22)
    near_y = min(height - 8.0, net_mid[1] + height * 0.34)
    image = {
        "near_left_corner": [net_mid[0] - near_width / 2.0, near_y],
        "near_right_corner": [net_mid[0] + near_width / 2.0, near_y],
        "far_right_corner": [net_mid[0] + far_width / 2.0, far_y],
        "far_left_corner": [net_mid[0] - far_width / 2.0, far_y],
    }
    return _homography_from_named_image_points(image)


def _homography_from_named_image_points(named_points: Mapping[str, Sequence[float]]) -> list[list[float]]:
    world_pts = []
    image_pts = []
    for name, image_xy in named_points.items():
        if name not in WORLD_XY_BY_NAME or name in TOP_NET_KEYPOINT_NAMES:
            continue
        x, y = WORLD_XY_BY_NAME[name]
        world_pts.append([x, y, 0.0])
        image_pts.append([float(image_xy[0]), float(image_xy[1])])
    if len(world_pts) < 4:
        raise ValueError("homography requires at least 4 named court correspondences")
    return homography_from_planar_points(world_pts, image_pts)


def _floor_correspondences(correspondences: Mapping[str, Sequence[float]]) -> dict[str, Sequence[float]]:
    return {
        name: xy
        for name, xy in correspondences.items()
        if name in WORLD_XY_BY_NAME and name not in TOP_NET_KEYPOINT_NAMES
    }


def _evidence_counts_for_keypoint(
    name: str,
    line_evidence: Mapping[str, LineEvidence],
    accepted: set[str],
) -> dict[str, int]:
    counts = {"net": 0, "kitchen": 0, "sideline": 0, "baseline": 0}
    if "net" in line_evidence:
        # The physical net is the global anchor for the whole proposal, not only for
        # the three projected net keypoints. Keep that provenance visible on every
        # proposed corner the user may confirm or drag.
        counts["net"] = 1
    for family in KEYPOINT_LINE_FAMILIES.get(name, ()):
        if family not in line_evidence:
            continue
        stage = LINE_STAGE_BY_NAME.get(family)
        if stage in counts:
            counts[stage] += 1
    if name in accepted:
        for family in KEYPOINT_LINE_FAMILIES.get(name, ()):
            stage = LINE_STAGE_BY_NAME.get(family)
            if stage in counts:
                counts[stage] = max(counts[stage], 1)
    return counts


def _line_confidence_by_stage(line_evidence: Mapping[str, LineEvidence]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for line in line_evidence.values():
        grouped.setdefault(line.source_stage, []).append(float(line.confidence))
    return {stage: _median(values) for stage, values in grouped.items()}


def _default_residual_for_point(name: str, line_evidence: Mapping[str, LineEvidence]) -> float:
    families = KEYPOINT_LINE_FAMILIES.get(name, ())
    available = sum(1 for family in families if family in line_evidence)
    if name in ("net_left_sideline", "net_center", "net_right_sideline") and "net" in line_evidence:
        return 10.0
    if available >= 2:
        return 12.0
    if available == 1:
        return 24.0
    return 42.0


def _cv2_np() -> tuple[Any, Any]:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("net_anchor_court requires opencv-python and numpy") from exc
    return cv2, np


def _as_bgr_array(frame_bgr: Any) -> Any:
    _cv2, np = _cv2_np()
    frame = np.asarray(frame_bgr)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be an HxWx3 BGR image")
    return frame


def _sample_segment_gray(gray: Any, p1: Sequence[float], p2: Sequence[float], *, samples: int = 48) -> float:
    _cv2, np = _cv2_np()
    height, width = gray.shape
    xs = np.linspace(float(p1[0]), float(p2[0]), samples)
    ys = np.linspace(float(p1[1]), float(p2[1]), samples)
    values = []
    for x, y in zip(xs, ys, strict=True):
        xi = int(round(min(max(x, 0.0), width - 1)))
        yi = int(round(min(max(y, 0.0), height - 1)))
        values.append(float(gray[yi, xi]))
    return float(np.median(values))


def _sample_segment_bgr(frame: Any, p1: Sequence[float], p2: Sequence[float], *, samples: int = 32) -> list[float]:
    _cv2, np = _cv2_np()
    height, width = frame.shape[:2]
    xs = np.linspace(float(p1[0]), float(p2[0]), samples)
    ys = np.linspace(float(p1[1]), float(p2[1]), samples)
    colors = []
    for x, y in zip(xs, ys, strict=True):
        xi = int(round(min(max(x, 0.0), width - 1)))
        yi = int(round(min(max(y, 0.0), height - 1)))
        colors.append(frame[yi, xi].astype(float))
    if not colors:
        return [0.0, 0.0, 0.0]
    median = np.median(np.stack(colors, axis=0), axis=0)
    return [float(value) for value in median.tolist()]


def _post_support(edges: Any, left: Sequence[float], right: Sequence[float], height: int) -> float:
    import numpy as np

    supports = []
    for x, y in (left, right):
        xi = int(round(min(max(float(x), 0.0), edges.shape[1] - 1)))
        y0 = int(round(min(max(float(y) - height * 0.04, 0.0), edges.shape[0] - 1)))
        y1 = int(round(min(max(float(y) + height * 0.09, 0.0), edges.shape[0] - 1)))
        strip = edges[y0:y1, max(0, xi - 3): min(edges.shape[1], xi + 4)]
        supports.append(float(np.count_nonzero(strip)) / max(1.0, strip.size * 0.08))
    return _clamp01(sum(min(1.0, value) for value in supports) / 2.0)


def _sort_line_left_to_right(
    line: tuple[tuple[float, float], tuple[float, float]]
) -> tuple[tuple[float, float], tuple[float, float]]:
    return line if line[0][0] <= line[1][0] else (line[1], line[0])


def _line_to_json(line: tuple[tuple[float, float], tuple[float, float]]) -> list[list[float]]:
    return [[float(line[0][0]), float(line[0][1])], [float(line[1][0]), float(line[1][1])]]


def _int_point(point: Sequence[float]) -> tuple[int, int]:
    return int(round(float(point[0]))), int(round(float(point[1])))


def _xy(value: Sequence[float]) -> tuple[float, float]:
    return float(value[0]), float(value[1])


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _line_length(line: tuple[tuple[float, float], tuple[float, float]]) -> float:
    return _distance(line[0], line[1])


def _midpoint(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    return (float(a[0] + b[0]) / 2.0, float(a[1] + b[1]) / 2.0)


def _angle_distance(a: float, b: float) -> float:
    diff = (a - b + math.pi / 2.0) % math.pi - math.pi / 2.0
    return diff


def _normal_with_positive_y(angle: float) -> tuple[float, float]:
    nx, ny = -math.sin(angle), math.cos(angle)
    if ny < 0:
        nx, ny = -nx, -ny
    return nx, ny


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1])


def _line_signed_distance(line: tuple[float, float, float], normal: Sequence[float]) -> float:
    a, b, c = line
    point_on_line_closest_to_origin = (-a * c, -b * c)
    return _dot(normal, point_on_line_closest_to_origin)


def _line_from_points(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    x1, y1 = float(a[0]), float(a[1])
    x2, y2 = float(b[0]), float(b[1])
    dx, dy = x2 - x1, y2 - y1
    norm = math.hypot(dx, dy)
    if norm <= 1e-9:
        raise ValueError("degenerate line")
    acoef = dy / norm
    bcoef = -dx / norm
    ccoef = -(acoef * x1 + bcoef * y1)
    return acoef, bcoef, ccoef


def _fit_line(points: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    _cv2, np = _cv2_np()
    arr = np.asarray([[float(p[0]), float(p[1])] for p in points], dtype=np.float64)
    if arr.shape[0] < 2:
        raise ValueError("fit line requires at least two points")
    mean = arr.mean(axis=0)
    centered = arr - mean
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    tangent = vh[0]
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    c = -float(normal @ mean)
    return float(normal[0]), float(normal[1]), c


def _endpoints_along_line(
    points: Sequence[Sequence[float]],
    line: tuple[float, float, float],
    reference_angle: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    tangent = (math.cos(reference_angle), math.sin(reference_angle))
    projections = [_dot(tangent, p) for p in points]
    p_min = _project_point_to_line(points[int(min(range(len(points)), key=lambda idx: projections[idx]))], line)
    p_max = _project_point_to_line(points[int(max(range(len(points)), key=lambda idx: projections[idx]))], line)
    return (p_min, p_max) if _dot(tangent, p_min) <= _dot(tangent, p_max) else (p_max, p_min)


def _endpoints_along_points(points: Sequence[Sequence[float]]) -> tuple[tuple[float, float], tuple[float, float]]:
    if len(points) < 2:
        raise ValueError("endpoints require at least two points")
    p0, p1 = points[0], points[-1]
    return (float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))


def _project_point_to_line(point: Sequence[float], line: tuple[float, float, float]) -> tuple[float, float]:
    a, b, c = line
    x, y = float(point[0]), float(point[1])
    d = a * x + b * y + c
    return x - a * d, y - b * d


def _ordered_line_endpoints(
    endpoints: tuple[tuple[float, float], tuple[float, float]],
    reference_angle: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    tangent = (math.cos(reference_angle), math.sin(reference_angle))
    return endpoints if _dot(tangent, endpoints[0]) <= _dot(tangent, endpoints[1]) else (endpoints[1], endpoints[0])


def _intersect_lines(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float]:
    det = a[0] * b[1] - b[0] * a[1]
    if math.isclose(det, 0.0, abs_tol=1e-9):
        raise ValueError("parallel lines")
    x = (a[1] * b[2] - b[1] * a[2]) / det
    y = (b[0] * a[2] - a[0] * b[2]) / det
    return float(x), float(y)


def _mean_color(colors: Sequence[Sequence[float]], weights: Sequence[float] | None = None) -> list[float]:
    if not colors:
        return [0.0, 0.0, 0.0]
    if weights is None:
        weights = [1.0] * len(colors)
    total = max(1e-9, sum(float(weight) for weight in weights))
    return [
        sum(float(color[channel]) * float(weight) for color, weight in zip(colors, weights, strict=True)) / total
        for channel in range(3)
    ]


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


__all__ = [
    "DEFAULT_USER_INPUT_CONFIDENCE_THRESHOLD",
    "HIGH_CONFIDENCE_THRESHOLD",
    "CORNER_ALIASES",
    "LineEvidence",
    "NetDetection",
    "ProposalPoint",
    "RefinementDecision",
    "accept_candidate_correspondences",
    "build_proposals_from_homography",
    "cluster_ground_line_directions",
    "detect_net_anchor",
    "draw_net_anchor_overlay",
    "load_player_foot_points_from_tracks",
    "load_player_suppressed_frame",
    "project_standard_keypoints",
    "score_corner_proposals",
    "solve_net_anchor_court_from_frame",
]
