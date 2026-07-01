"""Selected-sample visual overlays for BODY world-joint review."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from .court_auto_evidence import calibration_for_image_size
from .court_calibration import calibration_image_size, project_planar_points, project_world_points
from .player_track_overlay import load_tracks
from .schemas import CourtCalibration, Tracks, validate_artifact_file


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_body_world_label_review_overlay"
INDEX_FILENAME = "body_world_label_review_overlay_index.json"

TRACK_BBOX_COLOR = (60, 220, 255)
JOINT_COLOR = (80, 255, 160)
OUT_OF_FRAME_JOINT_COLOR = (120, 120, 120)
TEXT_COLOR = (255, 255, 255)
TRACK_ANCHOR_COLOR = (255, 200, 80)
MIN_PASSED_JOINT_BBOX_CONTAINMENT_RATIO = 0.60
MIN_FAILED_JOINT_BBOX_CONTAINMENT_RATIO = 0.35
MAX_PASSED_JOINT_BBOX_CENTER_DELTA_DIAG = 0.50
MAX_FAILED_JOINT_BBOX_CENTER_DELTA_DIAG = 0.75
MAX_PASSED_FLOOR_ANCHOR_DELTA_PX = 24.0
MAX_FAILED_FLOOR_ANCHOR_DELTA_PX = 48.0
MAX_PASSED_FLOOR_ANCHOR_DELTA_DIAG = 0.20
MAX_FAILED_FLOOR_ANCHOR_DELTA_DIAG = 0.35
MIN_COMPETING_PLAYER_CONTAINMENT_MARGIN = 0.25
MIN_COMPETING_PLAYER_ALIGNMENT_SCORE_MARGIN = 0.35


def build_body_world_label_review_overlays(
    *,
    queue_path: str | Path,
    tracks_path: str | Path,
    calibration_path: str | Path,
    out_dir: str | Path,
    cv2_module: Any | None = None,
) -> dict[str, Any]:
    """Render per-sample BODY joint overlays for human review.

    These overlays are qualitative review artifacts. They intentionally keep
    ``not_ground_truth=true`` and do not promote BODY predictions into labels.
    """

    cv2 = cv2_module or _cv2()
    queue_file = Path(queue_path)
    tracks_file = Path(tracks_path)
    calibration_file = Path(calibration_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    queue = _read_json(queue_file)
    tracks = load_tracks(tracks_file)
    calibration = _load_calibration(calibration_file)
    track_frames = _track_frames_by_sample(tracks)
    track_frames_by_frame = _track_frames_by_frame(tracks)
    samples = _samples(queue)

    overlays = [
        _render_sample_overlay(
            cv2=cv2,
            sample=sample,
            tracks=tracks,
            track_frames=track_frames,
            track_frames_by_frame=track_frames_by_frame,
            calibration=calibration,
            out_dir=out,
        )
        for sample in samples
    ]
    rendered_count = sum(1 for overlay in overlays if overlay.get("rendered") is True)
    missing_frame_count = sum(1 for overlay in overlays if overlay.get("image_status") != "loaded")
    projection_failed_count = sum(1 for overlay in overlays if overlay.get("projection_status") != "projected")
    missing_track_bbox_count = sum(1 for overlay in overlays if overlay.get("track_bbox_status") == "missing")
    alignment_failed_count = sum(
        1 for overlay in overlays if _alignment_status(overlay) == "failed"
    )
    alignment_warning_count = sum(
        1 for overlay in overlays if _alignment_status(overlay) == "warning"
    )
    competing_player_warning_count = sum(
        1 for overlay in overlays if _competing_player_alignment_status(overlay) == "warning"
    )
    floor_anchor_projection_failed_count = sum(
        1 for overlay in overlays if _floor_anchor_projection_status(overlay) == "failed"
    )
    floor_anchor_projection_warning_count = sum(
        1 for overlay in overlays if _floor_anchor_projection_status(overlay) == "warning"
    )
    blockers = _blockers(
        sample_count=len(samples),
        missing_frame_count=missing_frame_count,
        projection_failed_count=projection_failed_count,
        floor_anchor_projection_failed_count=floor_anchor_projection_failed_count,
        alignment_failed_count=alignment_failed_count,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": _status(
            sample_count=len(samples),
            missing_frame_count=missing_frame_count,
            projection_failed_count=projection_failed_count,
            floor_anchor_projection_failed_count=floor_anchor_projection_failed_count,
            floor_anchor_projection_warning_count=floor_anchor_projection_warning_count,
            alignment_failed_count=alignment_failed_count,
            alignment_warning_count=alignment_warning_count,
            competing_player_warning_count=competing_player_warning_count,
        ),
        "queue_path": str(queue_file),
        "tracks_path": str(tracks_file),
        "calibration_path": str(calibration_file),
        "out_dir": str(out),
        "index_path": str(out / INDEX_FILENAME),
        "clip": str(queue.get("clip", "")),
        "source_video": str(queue.get("source_video", "")),
        "sample_count": len(samples),
        "rendered_count": rendered_count,
        "missing_frame_count": missing_frame_count,
        "projection_failed_count": projection_failed_count,
        "missing_track_bbox_count": missing_track_bbox_count,
        "floor_anchor_projection_failed_count": floor_anchor_projection_failed_count,
        "floor_anchor_projection_warning_count": floor_anchor_projection_warning_count,
        "alignment_failed_count": alignment_failed_count,
        "alignment_warning_count": alignment_warning_count,
        "competing_player_warning_count": competing_player_warning_count,
        "blockers": blockers,
        "overlays": overlays,
        "not_ground_truth": True,
        "trusted_for_world_mpjpe": False,
        "qualitative_status": "review_overlay_not_gate_verified",
    }
    _write_json(out / INDEX_FILENAME, manifest)
    return manifest


def build_body_world_label_review_overlays_from_run(
    *,
    run_dir: str | Path,
    out_dir: str | Path | None = None,
    cv2_module: Any | None = None,
) -> dict[str, Any]:
    run = Path(run_dir)
    bundle = run / "body_world_label_review_bundle"
    return build_body_world_label_review_overlays(
        queue_path=bundle / "body_world_label_review_queue.json",
        tracks_path=run / "tracks.json",
        calibration_path=run / "court_calibration.json",
        out_dir=out_dir or bundle / "overlays",
        cv2_module=cv2_module,
    )


def _render_sample_overlay(
    *,
    cv2: Any,
    sample: Mapping[str, Any],
    tracks: Tracks,
    track_frames: Mapping[tuple[int, int], Mapping[str, Any]],
    track_frames_by_frame: Mapping[int, list[Mapping[str, Any]]],
    calibration: CourtCalibration,
    out_dir: Path,
) -> dict[str, Any]:
    sample_id = str(sample.get("sample_id", ""))
    frame_index = _maybe_int(sample.get("frame_index"))
    player_id = _maybe_int(sample.get("player_id"))
    image_path = Path(str(sample.get("image_path", "")))
    overlay_path = out_dir / f"{_safe_name(sample_id or f'frame_{frame_index}_player_{player_id}')}_overlay.jpg"
    base = {
        "sample_id": sample_id,
        "frame_index": frame_index,
        "player_id": player_id,
        "image_path": str(image_path),
        "overlay_path": str(overlay_path),
        "rendered": False,
        "image_status": "missing",
        "track_bbox_status": "missing",
        "projection_status": "not_rendered_missing_frame",
        "projected_joint_count": 0,
        "in_frame_projected_joint_count": 0,
        "track_bbox": None,
        "warnings": [],
    }
    if not image_path.is_file():
        return base

    frame = cv2.imread(str(image_path))
    if frame is None:
        return {**base, "image_status": "unreadable"}

    height, width = frame.shape[:2]
    bbox_scale_x, bbox_scale_y = _bbox_scale_for_review_frame(calibration, width=int(width), height=int(height))
    scaled_calibration = calibration_for_image_size(calibration, width=int(width), height=int(height))
    overlay = dict(base)
    overlay["image_status"] = "loaded"
    overlay["frame_size"] = [int(width), int(height)]
    overlay["track_bbox_scale_x"] = round(bbox_scale_x, 6)
    overlay["track_bbox_scale_y"] = round(bbox_scale_y, 6)

    track_frame = track_frames.get((frame_index, player_id)) if frame_index is not None and player_id is not None else None
    if track_frame is not None:
        bbox = _scale_bbox([float(value) for value in track_frame["bbox"]], scale_x=bbox_scale_x, scale_y=bbox_scale_y)
        overlay["track_bbox"] = _round_values(bbox)
        overlay["track_bbox_status"] = "matched"
        _draw_track_bbox(cv2, frame, bbox, sample_id=sample_id, conf=track_frame.get("conf"))
        floor_alignment = _track_floor_projection_alignment(
            scaled_calibration,
            track_frame.get("world_xy"),
            bbox,
        )
        pnp_floor_alignment = _track_floor_projection_alignment(
            scaled_calibration,
            track_frame.get("world_xy"),
            bbox,
            projection_mode="pnp",
        )
        overlay["track_floor_projection_alignment"] = floor_alignment
        overlay["pnp_track_floor_projection_alignment"] = pnp_floor_alignment
        if floor_alignment["status"] == "failed":
            overlay["warnings"].append("body_floor_anchor_projection_failed")
        elif floor_alignment["status"] == "warning":
            overlay["warnings"].append("body_floor_anchor_projection_warning")

    joints_world = _vectors(sample.get("predicted_joints_world"), length=3)
    projected: list[list[float]] = []
    if not joints_world:
        overlay["projection_status"] = "missing_predicted_joints_world"
    else:
        try:
            projected = _project_world_points_for_review(scaled_calibration, joints_world)
            in_frame_count = _draw_projected_joints(cv2, frame, projected, width=int(width), height=int(height))
            overlay["projection_status"] = "projected"
            overlay["projection_mode"] = "homography_grounded_pnp_vertical"
            overlay["projected_joint_count"] = len(projected)
            overlay["in_frame_projected_joint_count"] = in_frame_count
            overlay["projected_joint_bounds"] = _projected_bounds(projected)
        except (ValueError, OverflowError) as exc:
            overlay["projection_status"] = "projection_failed"
            overlay["projection_error"] = str(exc)

    if track_frame is not None and overlay["projection_status"] == "projected":
        alignment = _joint_bbox_alignment(projected, bbox)
        overlay["joint_bbox_alignment"] = alignment
        if alignment["status"] == "failed":
            overlay["warnings"].append("body_joint_overlay_alignment_failed")
        elif alignment["status"] == "warning":
            overlay["warnings"].append("body_joint_overlay_alignment_warning")
        competing_alignment = _competing_player_alignment(
            projected,
            bbox,
            track_frames_by_frame.get(frame_index, []) if frame_index is not None else [],
            target_player_id=player_id,
            scale_x=bbox_scale_x,
            scale_y=bbox_scale_y,
        )
        overlay["competing_player_alignment"] = competing_alignment
        if competing_alignment["status"] == "warning":
            overlay["warnings"].append("body_joint_overlay_competing_player_warning")

    if sample.get("track_world_xy") is not None:
        _draw_track_anchor_label(cv2, frame, sample)
    _draw_sample_header(cv2, frame, sample_id=sample_id, frame_index=frame_index, player_id=player_id)

    if not cv2.imwrite(str(overlay_path), frame):
        raise RuntimeError(f"cannot write BODY review overlay image: {overlay_path}")
    overlay["rendered"] = True
    return overlay


def _track_frames_by_sample(tracks: Tracks) -> dict[tuple[int, int], dict[str, Any]]:
    by_sample: dict[tuple[int, int], dict[str, Any]] = {}
    for player in tracks.players:
        for frame in player.frames:
            frame_index = int(round(float(frame.t) * float(tracks.fps)))
            by_sample[(frame_index, int(player.id))] = {
                "bbox": [float(value) for value in frame.bbox],
                "conf": float(frame.conf),
                "world_xy": [float(value) for value in frame.world_xy],
            }
    return by_sample


def _track_frames_by_frame(tracks: Tracks) -> dict[int, list[dict[str, Any]]]:
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for player in tracks.players:
        for frame in player.frames:
            frame_index = int(round(float(frame.t) * float(tracks.fps)))
            by_frame.setdefault(frame_index, []).append(
                {
                    "player_id": int(player.id),
                    "bbox": [float(value) for value in frame.bbox],
                    "conf": float(frame.conf),
                    "world_xy": [float(value) for value in frame.world_xy],
                }
            )
    return by_frame


def _bbox_scale_for_review_frame(calibration: CourtCalibration, *, width: int, height: int) -> tuple[float, float]:
    try:
        base_width, base_height = calibration_image_size(
            calibration,
            fallback_target=(float(width), float(height)),
        )
    except ValueError:
        return 1.0, 1.0
    if base_width <= 0.0 or base_height <= 0.0:
        return 1.0, 1.0
    return float(width) / float(base_width), float(height) / float(base_height)


def _scale_bbox(bbox: list[float], *, scale_x: float, scale_y: float) -> list[float]:
    x1, y1, x2, y2 = bbox[:4]
    return [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]


def _draw_track_bbox(cv2: Any, frame: Any, bbox: list[float], *, sample_id: str, conf: Any) -> None:
    x1, y1, x2, y2 = _int_bbox(bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), TRACK_BBOX_COLOR, 2)
    cv2.circle(frame, ((x1 + x2) // 2, y2), 5, TRACK_ANCHOR_COLOR, -1)
    label = sample_id
    if conf is not None:
        try:
            label = f"{sample_id} bbox {float(conf):.2f}"
        except (TypeError, ValueError):
            pass
    cv2.putText(frame, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TRACK_BBOX_COLOR, 2)


def _draw_projected_joints(cv2: Any, frame: Any, projected: list[list[float]], *, width: int, height: int) -> int:
    in_frame_count = 0
    for point in projected:
        if len(point) < 2 or not _finite(point[0]) or not _finite(point[1]):
            continue
        center = (int(round(point[0])), int(round(point[1])))
        in_frame = 0 <= center[0] < width and 0 <= center[1] < height
        if in_frame:
            in_frame_count += 1
        cv2.circle(frame, center, 3, JOINT_COLOR if in_frame else OUT_OF_FRAME_JOINT_COLOR, -1, cv2.LINE_AA)
    return in_frame_count


def _draw_track_anchor_label(cv2: Any, frame: Any, sample: Mapping[str, Any]) -> None:
    track_world_xy = sample.get("track_world_xy")
    if not isinstance(track_world_xy, list | tuple) or len(track_world_xy) < 2:
        return
    label = f"world_xy {float(track_world_xy[0]):.2f}, {float(track_world_xy[1]):.2f}"
    cv2.putText(frame, label, (16, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 2)


def _draw_sample_header(cv2: Any, frame: Any, *, sample_id: str, frame_index: int | None, player_id: int | None) -> None:
    text = sample_id or f"frame={frame_index} player={player_id}"
    cv2.putText(frame, text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, TEXT_COLOR, 2)


def _projected_bounds(points: list[list[float]]) -> dict[str, float]:
    finite_points = [point for point in points if len(point) >= 2 and _finite(point[0]) and _finite(point[1])]
    if not finite_points:
        return {}
    xs = [float(point[0]) for point in finite_points]
    ys = [float(point[1]) for point in finite_points]
    return {
        "min_x": round(min(xs), 3),
        "min_y": round(min(ys), 3),
        "max_x": round(max(xs), 3),
        "max_y": round(max(ys), 3),
    }


def _project_world_points_for_review(
    calibration: CourtCalibration,
    joints_world: list[list[float]],
) -> list[list[float]]:
    projected: list[list[float]] = []
    for joint in joints_world:
        x, y, z = float(joint[0]), float(joint[1]), float(joint[2])
        ground = project_planar_points(calibration.homography, [[x, y]])[0]
        if math.isclose(z, 0.0, abs_tol=1e-9):
            projected.append(ground)
            continue
        pnp_ground, pnp_joint = project_world_points(
            calibration.extrinsics,
            calibration.intrinsics,
            [[x, y, 0.0], [x, y, z]],
        )
        projected.append(
            [
                ground[0] + (pnp_joint[0] - pnp_ground[0]),
                ground[1] + (pnp_joint[1] - pnp_ground[1]),
            ]
        )
    return projected


def _joint_bbox_alignment(projected: list[list[float]], bbox: list[float]) -> dict[str, Any]:
    points = [point for point in projected if len(point) >= 2 and _finite(point[0]) and _finite(point[1])]
    if not points:
        return {
            "status": "not_measured",
            "reason": "missing_projected_joints",
            "projected_joint_count": 0,
        }
    x1, y1, x2, y2 = bbox[:4]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    bbox_diag = math.hypot(width, height)
    inside_count = sum(1 for x, y, *_ in points if x1 <= float(x) <= x2 and y1 <= float(y) <= y2)
    containment_ratio = inside_count / len(points)
    bbox_center = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
    joint_center = [
        sum(float(point[0]) for point in points) / len(points),
        sum(float(point[1]) for point in points) / len(points),
    ]
    center_delta_px = math.hypot(joint_center[0] - bbox_center[0], joint_center[1] - bbox_center[1])
    center_delta_bbox_diag = center_delta_px / bbox_diag if bbox_diag > 0.0 else float("inf")
    status = "passed"
    if (
        center_delta_bbox_diag > MAX_FAILED_JOINT_BBOX_CENTER_DELTA_DIAG
        or (
            containment_ratio < MIN_FAILED_JOINT_BBOX_CONTAINMENT_RATIO
            and center_delta_bbox_diag > MAX_PASSED_JOINT_BBOX_CENTER_DELTA_DIAG
        )
    ):
        status = "failed"
    elif (
        containment_ratio < MIN_PASSED_JOINT_BBOX_CONTAINMENT_RATIO
        or center_delta_bbox_diag > MAX_PASSED_JOINT_BBOX_CENTER_DELTA_DIAG
    ):
        status = "warning"
    return {
        "status": status,
        "inside_bbox_joint_count": inside_count,
        "projected_joint_count": len(points),
        "containment_ratio": round(containment_ratio, 4),
        "center_delta_px": round(center_delta_px, 3),
        "center_delta_bbox_diag": round(center_delta_bbox_diag, 4),
        "bbox_center": _round_values(bbox_center),
        "projected_joint_center": _round_values(joint_center),
    }


def _competing_player_alignment(
    projected: list[list[float]],
    target_bbox: list[float],
    frame_track_frames: list[Mapping[str, Any]],
    *,
    target_player_id: int | None,
    scale_x: float,
    scale_y: float,
) -> dict[str, Any]:
    target_alignment = _joint_bbox_alignment(projected, target_bbox)
    if target_alignment.get("status") == "not_measured":
        return {
            "status": "not_measured",
            "reason": "missing_target_alignment",
            "target_player_id": target_player_id,
        }

    competitors: list[tuple[float, int, list[float], dict[str, Any]]] = []
    for track_frame in frame_track_frames:
        player_id = _maybe_int(track_frame.get("player_id"))
        if player_id is None or player_id == target_player_id:
            continue
        raw_bbox = track_frame.get("bbox")
        if not isinstance(raw_bbox, list | tuple) or len(raw_bbox) < 4:
            continue
        bbox = _scale_bbox([float(value) for value in raw_bbox[:4]], scale_x=scale_x, scale_y=scale_y)
        alignment = _joint_bbox_alignment(projected, bbox)
        if alignment.get("status") == "not_measured":
            continue
        competitors.append((_joint_bbox_alignment_score(alignment), player_id, bbox, alignment))

    if not competitors:
        return {
            "status": "not_measured",
            "reason": "no_competing_players",
            "target_player_id": target_player_id,
        }

    target_score = _joint_bbox_alignment_score(target_alignment)
    best_score, best_player_id, best_bbox, best_alignment = max(
        competitors,
        key=lambda item: (
            item[0],
            _alignment_float(item[3], "containment_ratio"),
            -_alignment_float(item[3], "center_delta_bbox_diag", default=float("inf")),
        ),
    )
    target_containment = _alignment_float(target_alignment, "containment_ratio")
    best_containment = _alignment_float(best_alignment, "containment_ratio")
    score_margin = best_score - target_score
    containment_margin = best_containment - target_containment
    status = "passed"
    reason = "no_materially_better_competing_player"
    if (
        best_containment >= MIN_PASSED_JOINT_BBOX_CONTAINMENT_RATIO
        and containment_margin >= MIN_COMPETING_PLAYER_CONTAINMENT_MARGIN
        and score_margin >= MIN_COMPETING_PLAYER_ALIGNMENT_SCORE_MARGIN
    ):
        status = "warning"
        reason = "competing_player_bbox_fits_projected_joints_better"
    return {
        "status": status,
        "reason": reason,
        "target_player_id": target_player_id,
        "best_player_id": best_player_id,
        "best_player_bbox": _round_values(best_bbox),
        "target_containment_ratio": round(target_containment, 4),
        "best_player_containment_ratio": round(best_containment, 4),
        "target_center_delta_bbox_diag": round(_alignment_float(target_alignment, "center_delta_bbox_diag"), 4),
        "best_player_center_delta_bbox_diag": round(_alignment_float(best_alignment, "center_delta_bbox_diag"), 4),
        "target_alignment_score": round(target_score, 4),
        "best_player_alignment_score": round(best_score, 4),
        "containment_margin": round(containment_margin, 4),
        "score_margin": round(score_margin, 4),
    }


def _joint_bbox_alignment_score(alignment: Mapping[str, Any]) -> float:
    containment = _alignment_float(alignment, "containment_ratio")
    center_delta = _alignment_float(alignment, "center_delta_bbox_diag", default=float("inf"))
    if not _finite(center_delta):
        return -float("inf")
    return containment - center_delta


def _track_floor_projection_alignment(
    calibration: CourtCalibration,
    track_world_xy: Any,
    bbox: list[float],
    *,
    projection_mode: str = "homography",
) -> dict[str, Any]:
    world_xy = _vector(track_world_xy, length=2)
    x1, y1, x2, y2 = bbox[:4]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    bbox_diag = math.hypot(width, height)
    bbox_footpoint = [(x1 + x2) / 2.0, y2]
    base = {
        "status": "not_measured",
        "projection_mode": projection_mode,
        "bbox_footpoint": _round_values(bbox_footpoint),
        "track_world_xy": world_xy,
    }
    if not world_xy:
        return {**base, "reason": "missing_track_world_xy"}
    try:
        if projection_mode == "pnp":
            projected = project_world_points(
                calibration.extrinsics,
                calibration.intrinsics,
                [[world_xy[0], world_xy[1], 0.0]],
            )[0]
        else:
            projected = project_planar_points(calibration.homography, [[world_xy[0], world_xy[1]]])[0]
    except (ValueError, OverflowError) as exc:
        return {**base, "status": "failed", "reason": "projection_failed", "projection_error": str(exc)}

    center_delta_px = math.hypot(float(projected[0]) - bbox_footpoint[0], float(projected[1]) - bbox_footpoint[1])
    center_delta_bbox_diag = center_delta_px / bbox_diag if bbox_diag > 0.0 else float("inf")
    pass_limit_px = max(MAX_PASSED_FLOOR_ANCHOR_DELTA_PX, bbox_diag * MAX_PASSED_FLOOR_ANCHOR_DELTA_DIAG)
    fail_limit_px = max(MAX_FAILED_FLOOR_ANCHOR_DELTA_PX, bbox_diag * MAX_FAILED_FLOOR_ANCHOR_DELTA_DIAG)
    status = "passed"
    if center_delta_px > fail_limit_px:
        status = "failed"
    elif center_delta_px > pass_limit_px:
        status = "warning"
    return {
        **base,
        "status": status,
        "projected_track_world_xy": _round_values([float(projected[0]), float(projected[1])]),
        "center_delta_px": round(center_delta_px, 3),
        "center_delta_bbox_diag": round(center_delta_bbox_diag, 4),
        "pass_limit_px": round(pass_limit_px, 3),
        "fail_limit_px": round(fail_limit_px, 3),
    }


def _alignment_status(overlay: Mapping[str, Any]) -> str:
    alignment = overlay.get("joint_bbox_alignment")
    if not isinstance(alignment, Mapping):
        return "not_measured"
    return str(alignment.get("status", "not_measured"))


def _competing_player_alignment_status(overlay: Mapping[str, Any]) -> str:
    alignment = overlay.get("competing_player_alignment")
    if not isinstance(alignment, Mapping):
        return "not_measured"
    return str(alignment.get("status", "not_measured"))


def _floor_anchor_projection_status(overlay: Mapping[str, Any]) -> str:
    alignment = overlay.get("track_floor_projection_alignment")
    if not isinstance(alignment, Mapping):
        return "not_measured"
    return str(alignment.get("status", "not_measured"))


def _status(
    *,
    sample_count: int,
    missing_frame_count: int,
    projection_failed_count: int,
    floor_anchor_projection_failed_count: int,
    floor_anchor_projection_warning_count: int,
    alignment_failed_count: int,
    alignment_warning_count: int,
    competing_player_warning_count: int,
) -> str:
    if sample_count == 0:
        return "blocked_no_review_samples"
    if missing_frame_count:
        return "blocked_missing_review_frames"
    if projection_failed_count:
        return "blocked_projection_failed"
    if floor_anchor_projection_failed_count:
        return "rendered_floor_anchor_projection_failed"
    if alignment_failed_count:
        return "rendered_alignment_failed"
    if floor_anchor_projection_warning_count or alignment_warning_count or competing_player_warning_count:
        return "ready_for_review_with_overlay_warnings"
    return "ready_for_review"


def _blockers(
    *,
    sample_count: int,
    missing_frame_count: int,
    projection_failed_count: int,
    floor_anchor_projection_failed_count: int,
    alignment_failed_count: int,
) -> list[str]:
    blockers: list[str] = []
    if sample_count == 0:
        blockers.append("missing_review_samples")
    if missing_frame_count:
        blockers.append("missing_review_frame")
    if projection_failed_count:
        blockers.append("unprojectable_body_world_joints")
    if floor_anchor_projection_failed_count:
        blockers.append("body_floor_anchor_projection_failed")
    if alignment_failed_count:
        blockers.append("body_joint_overlay_alignment_failed")
    return blockers


def _samples(queue: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    samples = queue.get("samples")
    if not isinstance(samples, list):
        return []
    return [sample for sample in samples if isinstance(sample, Mapping)]


def _vectors(value: Any, *, length: int) -> list[list[float]]:
    if not isinstance(value, list):
        return []
    vectors: list[list[float]] = []
    for item in value:
        if not isinstance(item, list | tuple) or len(item) < length:
            continue
        parsed = [float(item[idx]) for idx in range(length)]
        if all(_finite(component) for component in parsed):
            vectors.append(parsed)
    return vectors


def _vector(value: Any, *, length: int) -> list[float]:
    if not isinstance(value, list | tuple) or len(value) < length:
        return []
    parsed = [float(value[idx]) for idx in range(length)]
    return parsed if all(_finite(component) for component in parsed) else []


def _int_bbox(bbox: list[float]) -> tuple[int, int, int, int]:
    return tuple(int(round(value)) for value in bbox[:4])  # type: ignore[return-value]


def _round_values(values: list[float]) -> list[float]:
    return [round(float(value), 3) for value in values]


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or "sample"


def _maybe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _alignment_float(alignment: Mapping[str, Any], key: str, *, default: float = 0.0) -> float:
    try:
        value = float(alignment.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if _finite(value) else default


def _load_calibration(path: Path) -> CourtCalibration:
    parsed = validate_artifact_file("court_calibration", path)
    if not isinstance(parsed, CourtCalibration):
        raise ValueError("court calibration artifact did not parse as CourtCalibration")
    return parsed


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for BODY world-label review overlay rendering") from exc
    return cv2


__all__ = [
    "ARTIFACT_TYPE",
    "INDEX_FILENAME",
    "build_body_world_label_review_overlays",
    "build_body_world_label_review_overlays_from_run",
]
