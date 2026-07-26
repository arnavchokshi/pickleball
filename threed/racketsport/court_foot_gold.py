"""Build and score a compact human-reference court/foot review packet.

Automatic observations remain immutable. Reviewer edits are stored as an
overlay so calibration, foot-localization, and end-to-end error can be
decomposed without nearest-point matching.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from threed.racketsport.court_structured_solver import FLOOR_WORLD_XY_M
from .external_gt_body_prediction_schema import canonical_mhr70_keypoint_name


FOOT_SEMANTIC_POLICY_VERSION = "mhr70_index_authoritative_v1"


@dataclass(frozen=True)
class GoldClipSpec:
    clip_id: str
    video_path: Path
    court_lock_path: Path
    artifacts_dir: Path | None = None


def build_gold_packet(
    specs: Sequence[GoldClipSpec],
    output_dir: str | Path,
    *,
    frames_per_clip: int = 12,
    template_path: str | Path | None = None,
) -> dict[str, Any]:
    """Extract review frames and immutable court/foot prelabels."""

    destination = Path(output_dir)
    frame_dir = destination / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    clips: list[dict[str, Any]] = []
    total_frames = 0
    for spec in specs:
        lock = _read_json(spec.court_lock_path)
        homography = np.asarray(lock["homography_image_from_court"], dtype=np.float64)
        court_points = {name: _project(homography, xy) for name, xy in FLOOR_WORLD_XY_M.items()}
        tracks = _optional_json(spec.artifacts_dir, "tracks.json")
        sam3d = _optional_json(spec.artifacts_dir, "sam3d_keypoints_2d.json")
        placement = _optional_json(spec.artifacts_dir, "placement.json")
        track_index, player_meta = _track_index(tracks)
        foot_index = _foot_index(sam3d)
        placement_index = _placement_index(placement)
        available = sorted(set(foot_index) or set(track_index))
        video = cv2.VideoCapture(str(spec.video_path))
        if not video.isOpened():
            raise ValueError(f"could not open video: {spec.video_path}")
        total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(video.get(cv2.CAP_PROP_FPS) or 0.0)
        selected = _spread_indices(available or list(range(max(total, 1))), frames_per_clip)
        rows: list[dict[str, Any]] = []
        for frame_index in selected:
            video.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = video.read()
            if not ok:
                continue
            image_name = f"{spec.clip_id}_f{int(frame_index):06d}.jpg"
            cv2.imwrite(str(frame_dir / image_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            players: list[dict[str, Any]] = []
            player_ids = sorted(set(track_index.get(frame_index, {})) | set(foot_index.get(frame_index, {})))
            for player_id in player_ids:
                points: dict[str, list[float] | None] = dict(
                    foot_index.get(frame_index, {}).get(player_id, {})
                )
                bbox = track_index.get(frame_index, {}).get(player_id)
                prelabel_source = "sam3d_body_foot_keypoints" if points else "missing"
                if bbox is not None:
                    points["bbox_bottom_center"] = [
                        float((bbox[0] + bbox[2]) * 0.5),
                        float(bbox[3]),
                    ]
                    if prelabel_source == "missing":
                        width = float(bbox[2] - bbox[0])
                        center_x = float((bbox[0] + bbox[2]) * 0.5)
                        points["left_contact"] = [center_x - 0.10 * width, float(bbox[3])]
                        points["right_contact"] = [center_x + 0.10 * width, float(bbox[3])]
                        prelabel_source = "bbox_bottom_low_confidence"
                _add_contact_points(points)
                for name in (
                    "left_ankle",
                    "left_heel",
                    "left_toe",
                    "left_contact",
                    "right_ankle",
                    "right_heel",
                    "right_toe",
                    "right_contact",
                ):
                    points.setdefault(name, None)
                placement_row = placement_index.get(frame_index, {}).get(player_id, {})
                players.append(
                    {
                        "player_id": player_id,
                        "role": player_meta.get(player_id, {}).get("role"),
                        "side": player_meta.get(player_id, {}).get("side"),
                        "bbox_xyxy": bbox,
                        "points": points,
                        "prelabel_source": prelabel_source,
                        "support_foot": _suggest_support_foot(points),
                        "contact_state": _contact_state(placement_row),
                        "placement": placement_row,
                    }
                )
            rows.append(
                {
                    "frame_id": f"{spec.clip_id}:{int(frame_index)}",
                    "frame_index": int(frame_index),
                    "t": (float(frame_index) / fps) if fps > 0 else None,
                    "image": f"frames/{image_name}",
                    "image_size": [int(frame.shape[1]), int(frame.shape[0])],
                    "automatic_court_points": court_points,
                    "players": players,
                }
            )
        video.release()
        total_frames += len(rows)
        clips.append(
            {
                "clip_id": spec.clip_id,
                "video_path": str(spec.video_path.resolve()),
                "court_lock_path": str(spec.court_lock_path.resolve()),
                "artifacts_dir": None if spec.artifacts_dir is None else str(spec.artifacts_dir.resolve()),
                "fps": fps,
                "automatic_homography_image_from_court": homography.tolist(),
                "frames": rows,
            }
        )
    packet = {
        "artifact_type": "racketsport_court_foot_human_reference_packet",
        "schema_version": 1,
        "verified": False,
        "measurement_authority": "human_reference_estimate_only",
        "foot_semantic_policy_version": FOOT_SEMANTIC_POLICY_VERSION,
        "instructions": {
            "court": "Correct only wrong named court points; do not nearest-match semantics.",
            "feet": "Correct visible ankle, heel, toe, and sole-contact pixels for the same player and same foot.",
            "missing": "Mark covered or invisible points as occluded; never guess them.",
            "contact": "Choose planted, airborne, or uncertain and the actual support foot.",
        },
        "frames_per_clip": int(frames_per_clip),
        "frame_count": int(total_frames),
        "clips": clips,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "review_packet.json").write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    if template_path is None:
        template_path = Path(__file__).resolve().parents[2] / "web/replay/public/court_foot_review_template.html"
    html = Path(template_path).read_text().replace("__PACKET_JSON__", _safe_script_json(packet))
    (destination / "START_HERE.html").write_text(html)
    return packet


def build_stabilization_review_packet(
    packet_path: str | Path,
    output_dir: str | Path,
    *,
    moments_per_category: int = 8,
    template_path: str | Path | None = None,
) -> dict[str, Any]:
    """Lock a deterministic one-player-per-moment placement/NVZ review.

    Selection uses only existing immutable prelabels.  The result records the
    source packet hash and category assignment so candidate output cannot
    influence which moments become the final selection set.
    """

    source_path = Path(packet_path)
    source = _read_json(source_path)
    destination = Path(output_dir)
    frame_dir = destination / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    candidates: dict[str, list[dict[str, Any]]] = {
        "clear_outside": [],
        "line_or_inside": [],
        "ambiguous_or_dynamic": [],
    }
    for clip in source.get("clips") or []:
        if not isinstance(clip, Mapping):
            continue
        homography = np.asarray(clip["automatic_homography_image_from_court"], dtype=np.float64)
        for frame in clip.get("frames") or []:
            if not isinstance(frame, Mapping):
                continue
            for player in frame.get("players") or []:
                if not isinstance(player, Mapping):
                    continue
                category, priority = _stabilization_category(player, homography=homography)
                candidates[category].append(
                    {
                        "clip": clip,
                        "frame": frame,
                        "player": player,
                        "priority": priority,
                    }
                )
    selected: list[dict[str, Any]] = []
    for category in ("clear_outside", "line_or_inside", "ambiguous_or_dynamic"):
        rows = sorted(
            candidates[category],
            key=lambda row: (
                float(row["priority"]),
                str(row["clip"].get("clip_id")),
                int(row["frame"].get("frame_index", 0)),
                str(row["player"].get("player_id")),
            ),
        )
        chosen = _round_robin_clips(rows, count=moments_per_category)
        if len(chosen) != moments_per_category:
            raise ValueError(
                f"stabilization packet needs {moments_per_category} {category} moments; found {len(chosen)}"
            )
        for row in chosen:
            row["category"] = category
        selected.extend(chosen)

    clip_rows: dict[str, dict[str, Any]] = {}
    source_root = source_path.parent
    for ordinal, row in enumerate(selected):
        clip = row["clip"]
        frame = row["frame"]
        player = row["player"]
        clip_id = str(clip["clip_id"])
        target_clip = clip_rows.setdefault(
            clip_id,
            {
                **{key: value for key, value in clip.items() if key != "frames"},
                "frames": [],
            },
        )
        source_image = source_root / str(frame["image"])
        image_name = f"moment_{ordinal + 1:02d}_{source_image.name}"
        shutil.copy2(source_image, frame_dir / image_name)
        target_clip["frames"].append(
            {
                **dict(frame),
                "frame_id": f"stabilization:{ordinal + 1:02d}:{frame['frame_id']}",
                "image": f"frames/{image_name}",
                "players": [dict(player)],
                "stabilization_category": row["category"],
                "review_scope": "single_active_player_feet_support_contact_and_visible_nvz_line",
            }
        )
    packet = {
        **{key: value for key, value in source.items() if key not in {"clips", "frame_count", "frames_per_clip"}},
        "artifact_type": "racketsport_foot_anchor_stabilization_review_packet",
        "foot_semantic_policy_version": FOOT_SEMANTIC_POLICY_VERSION,
        "frame_count": len(selected),
        "frames_per_clip": None,
        "clips": [clip_rows[key] for key in sorted(clip_rows)],
        "instructions": {
            "scope": "Review only the one shown active player in each moment.",
            "feet": "Correct visible heel, toe, and sole-contact pixels; mark covered points occluded.",
            "support": "Choose left, right, bilateral, airborne, or uncertain contact.",
            "nvz": "Correct the two visible endpoints of the applicable NVZ painted line.",
            "do_not_review": "Do not review other players, every joint, ball, paddle, or mesh.",
        },
        "stabilization_review": {
            "locked_for_final_selection": True,
            "candidate_outputs_used_for_selection": False,
            "selection_version": "foot_anchor_stabilization_stratified_v1",
            "source_packet": str(source_path.resolve()),
            "source_packet_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "category_counts": {
                category: sum(row["category"] == category for row in selected)
                for category in candidates
            },
        },
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "review_packet.json").write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if template_path is None:
        template_path = Path(__file__).resolve().parents[2] / "web/replay/public/court_foot_review_template.html"
    html = Path(template_path).read_text().replace("__PACKET_JSON__", _safe_script_json(packet))
    (destination / "START_HERE.html").write_text(html, encoding="utf-8")
    return packet


def _stabilization_category(
    player: Mapping[str, Any],
    *,
    homography: np.ndarray,
) -> tuple[str, float]:
    contact_state = str(player.get("contact_state") or "uncertain").lower()
    support = str(player.get("support_foot") or "")
    points = player.get("points") if isinstance(player.get("points"), Mapping) else {}
    contact = points.get(f"{support}_contact") if support in {"left", "right"} else None
    if contact is None:
        contact = points.get("left_contact") or points.get("right_contact")
    world = _unproject(homography, contact) if contact is not None else None
    source = str(player.get("prelabel_source") or "missing")
    if (
        world is None
        or contact_state in {"airborne", "missing"}
        or source in {"missing", "bbox_bottom_low_confidence"}
    ):
        return "ambiguous_or_dynamic", 0.0 if contact_state == "airborne" else 1.0
    line_distance = abs(abs(float(world[1])) - 2.1336)
    if abs(float(world[1])) < 2.1336:
        return "line_or_inside", abs(float(world[1]))
    if line_distance <= 0.12:
        return "line_or_inside", line_distance
    if line_distance <= 0.35:
        return "ambiguous_or_dynamic", line_distance
    return "clear_outside", -line_distance


def _round_robin_clips(rows: Sequence[dict[str, Any]], *, count: int) -> list[dict[str, Any]]:
    by_clip: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_clip.setdefault(str(row["clip"].get("clip_id")), []).append(row)
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        added = False
        for clip_id in sorted(by_clip):
            if by_clip[clip_id] and len(selected) < count:
                selected.append(by_clip[clip_id].pop(0))
                added = True
        if not added:
            break
    return selected


def score_gold_review(packet: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, Any]:
    """Score exact-semantic support pixels with a three-way error budget."""

    if (
        packet.get("artifact_type") == "racketsport_foot_anchor_stabilization_review_packet"
        and packet.get("foot_semantic_policy_version") != FOOT_SEMANTIC_POLICY_VERSION
    ):
        raise ValueError(
            "stabilization review packet predates authoritative MHR70 foot semantics; "
            "regenerate it before scoring"
        )

    review_frames = review.get("frames") if isinstance(review, Mapping) else None
    review_frames = review_frames if isinstance(review_frames, Mapping) else {}
    samples: list[dict[str, Any]] = []
    clip_summaries: list[dict[str, Any]] = []
    coverage = _review_coverage(packet, review_frames)
    for clip in packet.get("clips") or []:
        if not isinstance(clip, Mapping):
            continue
        frame_rows = [row for row in clip.get("frames") or [] if isinstance(row, Mapping)]
        human_h = _fit_human_court_homography(frame_rows, review_frames)
        auto_h = np.asarray(clip["automatic_homography_image_from_court"], dtype=np.float64)
        clip_samples: list[dict[str, Any]] = []
        if human_h is not None:
            for row in frame_rows:
                frame_review = review_frames.get(str(row.get("frame_id")))
                if not isinstance(frame_review, Mapping) or frame_review.get("status") != "accepted":
                    continue
                for player in row.get("players") or []:
                    if not isinstance(player, Mapping):
                        continue
                    player_id = str(player.get("player_id"))
                    player_review = (frame_review.get("players") or {}).get(player_id)
                    if not isinstance(player_review, Mapping):
                        continue
                    support = player_review.get("support_foot") or player.get("support_foot")
                    if support not in {"left", "right"}:
                        continue
                    semantic = f"{support}_contact"
                    automatic_px = (player.get("points") or {}).get(semantic)
                    player_review_points = player_review.get("points") or {}
                    reviewed_px = _reviewed_point(automatic_px, player_review_points, semantic)
                    if automatic_px is None or reviewed_px is None:
                        continue
                    reference_source = (
                        "manual_correction"
                        if _has_reviewed_xy(player_review_points, semantic)
                        else "accepted_prelabel"
                    )
                    auto_world_from_review = _unproject(auto_h, reviewed_px)
                    human_world_from_review = _unproject(human_h, reviewed_px)
                    human_world_from_auto = _unproject(human_h, automatic_px)
                    auto_world_from_auto = _unproject(auto_h, automatic_px)
                    if any(
                        value is None
                        for value in (
                            auto_world_from_review,
                            human_world_from_review,
                            human_world_from_auto,
                            auto_world_from_auto,
                        )
                    ):
                        continue
                    assert auto_world_from_review is not None
                    assert human_world_from_review is not None
                    assert human_world_from_auto is not None
                    assert auto_world_from_auto is not None
                    human_line = _nearest_regulation_line(human_world_from_review)
                    auto_line = _signed_distance_to_named_line(auto_world_from_auto, human_line["name"])
                    sample = {
                        "clip_id": clip.get("clip_id"),
                        "frame_id": row.get("frame_id"),
                        "player_id": player.get("player_id"),
                        "support_foot": support,
                        "contact_state": player_review.get("contact_state"),
                        "reference_source": reference_source,
                        "pixel_error": _distance(automatic_px, reviewed_px),
                        "calibration_error_m": _distance(auto_world_from_review, human_world_from_review),
                        "foot_localization_error_m": _distance(human_world_from_auto, human_world_from_review),
                        "end_to_end_error_m": _distance(auto_world_from_auto, human_world_from_review),
                        "nearest_regulation_line": human_line["name"],
                        "reference_signed_line_distance_m": human_line["signed_distance_m"],
                        "automatic_signed_line_distance_m": auto_line,
                        "signed_line_distance_error_m": abs(auto_line - human_line["signed_distance_m"]),
                        "correct_side_of_line": bool(
                            auto_line == 0.0
                            or human_line["signed_distance_m"] == 0.0
                            or math.copysign(1.0, auto_line)
                            == math.copysign(1.0, human_line["signed_distance_m"])
                        ),
                        "correct_kitchen_classification": _in_kitchen(auto_world_from_auto)
                        == _in_kitchen(human_world_from_review),
                    }
                    clip_samples.append(sample)
                    samples.append(sample)
        clip_summary = _metric_summary(str(clip.get("clip_id")), clip_samples)
        clip_summary["manual_correction_sample_count"] = sum(
            sample["reference_source"] == "manual_correction" for sample in clip_samples
        )
        clip_summary["accepted_prelabel_sample_count"] = sum(
            sample["reference_source"] == "accepted_prelabel" for sample in clip_samples
        )
        clip_summaries.append(clip_summary)
    manual_samples = [sample for sample in samples if sample["reference_source"] == "manual_correction"]
    accepted_prelabel_samples = [
        sample for sample in samples if sample["reference_source"] == "accepted_prelabel"
    ]
    return {
        "artifact_type": "racketsport_court_foot_human_reference_report",
        "schema_version": 1,
        "verified": False,
        "measurement_authority": "human_reference_estimate_only",
        "matching_policy": "exact_player_id_exact_support_foot_exact_semantic_name",
        "sample_count": len(samples),
        "manual_correction_sample_count": len(manual_samples),
        "accepted_prelabel_sample_count": len(accepted_prelabel_samples),
        "review_coverage": coverage,
        "summary": _metric_summary("all", samples),
        "summary_manual_corrections": _metric_summary("manual_corrections", manual_samples),
        "summary_accepted_prelabels": _metric_summary("accepted_prelabels", accepted_prelabel_samples),
        "clips": clip_summaries,
        "samples": samples,
    }


def _review_coverage(
    packet: Mapping[str, Any],
    review_frames: Mapping[str, Any],
) -> dict[str, Any]:
    packet_frame_ids = {
        str(row.get("frame_id"))
        for clip in packet.get("clips") or []
        if isinstance(clip, Mapping)
        for row in clip.get("frames") or []
        if isinstance(row, Mapping)
    }
    statuses = {"accepted": 0, "skipped": 0, "needs_edit": 0, "missing": 0}
    manual_court_points = 0
    occluded_court_points = 0
    manual_foot_points = 0
    occluded_foot_points = 0
    for frame_id in packet_frame_ids:
        reviewed = review_frames.get(frame_id)
        if not isinstance(reviewed, Mapping):
            statuses["missing"] += 1
            continue
        status = str(reviewed.get("status") or "missing")
        statuses[status if status in statuses else "missing"] += 1
        for value in (reviewed.get("court_points") or {}).values():
            if not isinstance(value, Mapping):
                continue
            occluded_court_points += value.get("occluded") is True
            manual_court_points += _mapping_has_xy(value)
        for player in (reviewed.get("players") or {}).values():
            if not isinstance(player, Mapping):
                continue
            for value in (player.get("points") or {}).values():
                if not isinstance(value, Mapping):
                    continue
                occluded_foot_points += value.get("occluded") is True
                manual_foot_points += _mapping_has_xy(value)
    return {
        "packet_frame_count": len(packet_frame_ids),
        "accepted_frame_count": statuses["accepted"],
        "skipped_frame_count": statuses["skipped"],
        "needs_edit_frame_count": statuses["needs_edit"],
        "missing_frame_count": statuses["missing"],
        "manual_court_point_count": int(manual_court_points),
        "occluded_court_point_count": int(occluded_court_points),
        "manual_foot_point_count": int(manual_foot_points),
        "occluded_foot_point_count": int(occluded_foot_points),
    }


def _mapping_has_xy(value: Mapping[str, Any]) -> bool:
    xy = value.get("xy")
    return (
        isinstance(xy, Sequence)
        and not isinstance(xy, (str, bytes))
        and len(xy) == 2
    )


def _has_reviewed_xy(review_points: Any, name: str) -> bool:
    if not isinstance(review_points, Mapping):
        return False
    value = review_points.get(name)
    return isinstance(value, Mapping) and value.get("occluded") is not True and _mapping_has_xy(value)


def _fit_human_court_homography(
    frame_rows: Sequence[Mapping[str, Any]],
    review_frames: Mapping[str, Any],
) -> np.ndarray | None:
    world: list[list[float]] = []
    image: list[list[float]] = []
    for row in frame_rows:
        reviewed = review_frames.get(str(row.get("frame_id")))
        if not isinstance(reviewed, Mapping) or reviewed.get("status") != "accepted":
            continue
        court_review = reviewed.get("court_points")
        court_review = court_review if isinstance(court_review, Mapping) else {}
        automatic = row.get("automatic_court_points") or {}
        for name, xy_world in FLOOR_WORLD_XY_M.items():
            xy = _reviewed_point(automatic.get(name), court_review, name)
            if xy is None:
                continue
            world.append([float(xy_world[0]), float(xy_world[1])])
            image.append([float(xy[0]), float(xy[1])])
    if len(world) < 4:
        return None
    homography, _ = cv2.findHomography(
        np.asarray(world, dtype=np.float64),
        np.asarray(image, dtype=np.float64),
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )
    return None if homography is None else homography / homography[2, 2]


def _reviewed_point(automatic: Any, review_points: Mapping[str, Any], name: str) -> list[float] | None:
    value = review_points.get(name)
    if isinstance(value, Mapping):
        if value.get("occluded") is True:
            return None
        value = value.get("xy")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        return [float(value[0]), float(value[1])]
    if isinstance(automatic, Sequence) and not isinstance(automatic, (str, bytes)) and len(automatic) == 2:
        return [float(automatic[0]), float(automatic[1])]
    return None


def _metric_summary(name: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"name": name, "sample_count": len(rows)}
    for key in (
        "pixel_error",
        "calibration_error_m",
        "foot_localization_error_m",
        "end_to_end_error_m",
        "signed_line_distance_error_m",
    ):
        values = np.asarray([float(row[key]) for row in rows if row.get(key) is not None], dtype=np.float64)
        metrics[key] = (
            None
            if values.size == 0
            else {
                "median": float(np.median(values)),
                "p90": float(np.percentile(values, 90)),
                "p95": float(np.percentile(values, 95)),
                "max": float(np.max(values)),
            }
        )
    for key in ("correct_side_of_line", "correct_kitchen_classification"):
        values = [bool(row[key]) for row in rows if key in row]
        metrics[key] = None if not values else sum(values) / len(values)
    return metrics


def _track_index(
    payload: Mapping[str, Any] | None,
) -> tuple[dict[int, dict[int, list[float]]], dict[int, dict[str, Any]]]:
    frames: dict[int, dict[int, list[float]]] = {}
    meta: dict[int, dict[str, Any]] = {}
    for player in (payload or {}).get("players") or []:
        player_id = int(player.get("id"))
        meta[player_id] = {"role": player.get("role"), "side": player.get("side")}
        for row in player.get("frames") or []:
            bbox = row.get("bbox")
            if isinstance(bbox, Sequence) and len(bbox) == 4:
                frames.setdefault(int(row["frame_idx"]), {})[player_id] = [float(value) for value in bbox]
    return frames, meta


def _foot_index(payload: Mapping[str, Any] | None) -> dict[int, dict[int, dict[str, list[float]]]]:
    frames: dict[int, dict[int, dict[str, list[float]]]] = {}
    for player in (payload or {}).get("players") or []:
        player_id = int(player.get("id"))
        for row in player.get("frames") or []:
            points: dict[str, list[float]] = {}
            for point in row.get("keypoints") or []:
                if not isinstance(point, Mapping) or not isinstance(point.get("xy_px"), Sequence):
                    continue
                name = canonical_mhr70_keypoint_name(point.get("name"), point.get("index"))
                if not name:
                    continue
                points[name] = [float(point["xy_px"][0]), float(point["xy_px"][1])]
            for side in ("left", "right"):
                toe_points = [
                    points[name]
                    for name in (
                        f"{side}_toe",
                        f"{side}_big_toe_tip",
                        f"{side}_small_toe_tip",
                    )
                    if name in points
                ]
                if toe_points:
                    points[f"{side}_toe"] = [
                        float(sum(point[axis] for point in toe_points) / len(toe_points))
                        for axis in (0, 1)
                    ]
            frames.setdefault(int(row["frame_idx"]), {})[player_id] = points
    return frames


def _placement_index(payload: Mapping[str, Any] | None) -> dict[int, dict[int, dict[str, Any]]]:
    frames: dict[int, dict[int, dict[str, Any]]] = {}
    for player in (payload or {}).get("players") or []:
        player_id = int(player.get("id"))
        for row in player.get("frames") or []:
            frames.setdefault(int(row["frame_idx"]), {})[player_id] = {
                key: row.get(key)
                for key in (
                    "contact_state",
                    "selected_support_signal",
                    "nearest_regulation_line",
                    "uncertainty_decomposition",
                    "measurement_provenance",
                    "stance",
                )
                if key in row
            }
    return frames


def _contact_state(row: Mapping[str, Any]) -> str:
    state = row.get("contact_state")
    if isinstance(state, Mapping):
        state = state.get("state")
    if state in {"planted", "airborne", "uncertain"}:
        return str(state)
    return "planted" if row.get("stance") is True else "uncertain"


def _add_contact_points(points: dict[str, list[float] | None]) -> None:
    for side in ("left", "right"):
        choices = [points.get(f"{side}_heel"), points.get(f"{side}_toe")]
        visible = [xy for xy in choices if xy is not None]
        if visible:
            points[f"{side}_contact"] = list(max(visible, key=lambda xy: xy[1]))


def _suggest_support_foot(points: Mapping[str, Sequence[float] | None]) -> str | None:
    left = points.get("left_contact")
    right = points.get("right_contact")
    if left is None and right is None:
        return None
    if right is None:
        return "left"
    if left is None:
        return "right"
    return "left" if float(left[1]) >= float(right[1]) else "right"


def _spread_indices(values: Sequence[int], count: int) -> list[int]:
    ordered = sorted(dict.fromkeys(int(value) for value in values))
    if len(ordered) <= count:
        return ordered
    positions = np.linspace(0, len(ordered) - 1, count)
    return [ordered[int(round(position))] for position in positions]


def _project(homography: np.ndarray, xy: Sequence[float]) -> list[float]:
    value = homography @ np.asarray([float(xy[0]), float(xy[1]), 1.0], dtype=np.float64)
    return [float(value[0] / value[2]), float(value[1] / value[2])]


def _unproject(homography: np.ndarray, xy: Sequence[float]) -> list[float] | None:
    try:
        inverse = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return None
    return _project(inverse, xy)


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return float(math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1])))


def _nearest_line_distances(xy: Sequence[float]) -> dict[str, float]:
    return {
        "left_sideline": float(xy[0]) + 3.048,
        "right_sideline": float(xy[0]) - 3.048,
        "near_baseline": float(xy[1]) + 6.7056,
        "far_baseline": float(xy[1]) - 6.7056,
        "near_nvz": float(xy[1]) + 2.1336,
        "net": float(xy[1]),
        "far_nvz": float(xy[1]) - 2.1336,
    }


def _nearest_regulation_line(xy: Sequence[float]) -> dict[str, Any]:
    candidates = _nearest_line_distances(xy)
    name = min(candidates, key=lambda key: abs(candidates[key]))
    return {"name": name, "signed_distance_m": candidates[name]}


def _signed_distance_to_named_line(xy: Sequence[float], name: str) -> float:
    return float(_nearest_line_distances(xy)[name])


def _in_kitchen(xy: Sequence[float]) -> bool:
    return abs(float(xy[0])) <= 3.048 and abs(float(xy[1])) <= 2.1336


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _optional_json(root: Path | None, name: str) -> dict[str, Any] | None:
    if root is None:
        return None
    path = root / name
    return _read_json(path) if path.is_file() else None


def _safe_script_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
