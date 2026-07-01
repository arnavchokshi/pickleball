"""SAT-HMR fallback conversion into BODY world artifacts.

This module does not run SAT-HMR. It consumes raw SAT-HMR per-person
prediction pickles written by a real inference run, matches them to scheduled
tracked players, and reuses the existing worldhmr grounding path.
"""

from __future__ import annotations

import json
import math
import pickle
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .body_joint_quality import build_body_joint_quality, write_body_joint_quality
from .body_mesh_readiness import build_body_mesh_readiness, write_body_mesh_readiness
from .schemas import CourtCalibration, Tracks, validate_artifact_file
from .worldhmr import FOOT_LOCK_SKATE_FREE_MAX_SLIDE_M, build_body_artifacts_from_fast_sam


ARTIFACT_TYPE = "racketsport_sat_hmr_body_fallback"
SCHEMA_VERSION = 1
_FRAME_RE = re.compile(r"(?:^|_)frame[_-]?(\d+)", re.IGNORECASE)
DEFAULT_SAT_HMR_MAX_ROOT_SPEED_MPS = 8.0
DEFAULT_SAT_HMR_MAX_TRACK_ANCHOR_SMOOTHING_RESIDUAL_M = 0.75


def build_sat_hmr_body_fallback(
    *,
    clip: str,
    predictions_dir: str | Path,
    tracks_path: str | Path,
    calibration_path: str | Path,
    body_compute_execution_path: str | Path,
    out_dir: str | Path,
    frame_compute_plan_path: str | Path | None = None,
    smoothing_alpha: float = 1.0,
    max_root_speed_mps: float | None = DEFAULT_SAT_HMR_MAX_ROOT_SPEED_MPS,
    max_track_anchor_smoothing_residual_m: float | None = DEFAULT_SAT_HMR_MAX_TRACK_ANCHOR_SMOOTHING_RESIDUAL_M,
    min_assignment_iou: float = 0.05,
) -> dict[str, Any]:
    """Write BODY artifacts from raw SAT-HMR predictions.

    The resulting artifacts are real model-output conversions, but they are not
    BODY promotion gates: world-MPJPE remains blocked unless reviewed BODY world
    labels are supplied by a separate gate.
    """

    predictions_base = Path(predictions_dir)
    if not predictions_base.is_dir():
        raise FileNotFoundError(f"missing SAT-HMR predictions directory: {predictions_base}")

    tracks = validate_artifact_file("tracks", tracks_path)
    calibration = validate_artifact_file("court_calibration", calibration_path)
    if not isinstance(tracks, Tracks):
        raise ValueError("tracks artifact did not parse as Tracks")
    if not isinstance(calibration, CourtCalibration):
        raise ValueError("court_calibration artifact did not parse as CourtCalibration")

    body_execution = _read_json_object(body_compute_execution_path)
    frame_plan = _read_optional_json(frame_compute_plan_path)
    track_frames = _track_frames_by_player_and_index(tracks)
    predictions = _load_predictions(predictions_base)
    samples, assignments = _match_predictions_to_schedule(
        predictions=predictions,
        body_execution=body_execution,
        track_frames=track_frames,
        fps=float(tracks.fps),
        min_assignment_iou=min_assignment_iou,
    )
    if not samples:
        raise ValueError("no SAT-HMR predictions matched scheduled BODY player frames")

    smpl_motion, skeleton3d, grounding_metrics = build_body_artifacts_from_fast_sam(
        samples,
        calibration=calibration,
        fps=float(tracks.fps),
        smoothing_alpha=smoothing_alpha,
        max_root_speed_mps=max_root_speed_mps,
        max_track_anchor_smoothing_residual_m=max_track_anchor_smoothing_residual_m,
        model="sat_hmr_world_joints",
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "smpl_motion.json", smpl_motion)
    _write_json(out / "skeleton3d.json", skeleton3d)

    body_joint_quality = build_body_joint_quality(
        clip=clip,
        smpl_motion=smpl_motion,
        skeleton3d=skeleton3d,
        body_compute_execution=body_execution,
        smpl_motion_path=str(out / "smpl_motion.json"),
        skeleton3d_path=str(out / "skeleton3d.json"),
        body_compute_execution_path=str(body_compute_execution_path),
    )
    write_body_joint_quality(out / "body_joint_quality.json", body_joint_quality)
    body_mesh_readiness = build_body_mesh_readiness(
        clip=clip,
        smpl_motion=smpl_motion,
        skeleton3d=skeleton3d,
        frame_compute_plan=frame_plan,
        body_compute_execution=body_execution,
        smpl_motion_path=str(out / "smpl_motion.json"),
        skeleton3d_path=str(out / "skeleton3d.json"),
        frame_compute_plan_path=str(frame_compute_plan_path or ""),
        body_compute_execution_path=str(body_compute_execution_path),
    )
    write_body_mesh_readiness(out / "body_mesh_readiness.json", body_mesh_readiness)

    max_slide_m = float(grounding_metrics.get("max_foot_lock_slide_m", 0.0))
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "clip": clip,
        "status": "ran_not_gate_verified",
        "source_mode": "sat_hmr_raw_prediction_fallback",
        "predictions_dir": str(predictions_base),
        "tracks_path": str(tracks_path),
        "calibration_path": str(calibration_path),
        "body_compute_execution_path": str(body_compute_execution_path),
        "out_dir": str(out),
        "paths": {
            "smpl_motion": str(out / "smpl_motion.json"),
            "skeleton3d": str(out / "skeleton3d.json"),
            "body_joint_quality": str(out / "body_joint_quality.json"),
            "body_mesh_readiness": str(out / "body_mesh_readiness.json"),
        },
        "assignment_summary": {
            "assigned_prediction_count": len(assignments),
            "scheduled_player_frame_count": _scheduled_player_frame_count(body_execution),
            "min_assignment_iou": float(min_assignment_iou),
        },
        "assignments": assignments,
        "grounding_metrics": grounding_metrics,
        "foot_slide_gate": {
            "name": "foot_slide_max_m",
            "threshold_m": FOOT_LOCK_SKATE_FREE_MAX_SLIDE_M,
            "value_m": max_slide_m,
            "passed": max_slide_m <= FOOT_LOCK_SKATE_FREE_MAX_SLIDE_M,
        },
        "world_mpjpe_gate": {
            "name": "world_mpjpe_mm",
            "target_range_mm": [50.0, 70.0],
            "value_mm": None,
            "passed": False,
            "status": "blocked_missing_body_world_gt",
        },
        "not_gate_verified": True,
    }
    _write_json(out / "sat_hmr_body_fallback_report.json", report)
    return report


def _load_predictions(predictions_dir: Path) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for path in sorted(predictions_dir.glob("*.pkl")):
        with path.open("rb") as handle:
            raw = pickle.load(handle)
        prediction = _normalize_prediction(raw, source_path=path)
        predictions.append(prediction)
    if not predictions:
        raise ValueError(f"no SAT-HMR prediction pickles found in {predictions_dir}")
    return predictions


def _normalize_prediction(raw: Any, *, source_path: Path) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"SAT-HMR prediction must be a mapping: {source_path}")
    frame_idx = _frame_idx(raw, source_path=source_path)
    joints_camera = _vector3_list(_first_present(raw, ("joints_camera", "pred_j3ds", "allSmplJoints3d")))
    vertices_camera = _vector3_list(_first_present(raw, ("vertices_camera", "pred_verts", "verts")), required=False)
    if not joints_camera and not vertices_camera:
        raise ValueError(f"missing SAT-HMR 3D joints/vertices in {source_path}")

    params = raw.get("params") if isinstance(raw.get("params"), Mapping) else {}
    pred_poses = _float_list(_first_present(raw, ("pred_poses", "pose", "poses"), default=[]), required=False)
    global_orient = _vector3(
        _first_present(raw, ("global_orient", "pred_global_orient"), default=_nested_param(params, "global_orient")),
        default=pred_poses[:3] if len(pred_poses) >= 3 else [0.0, 0.0, 0.0],
    )
    body_pose = _float_list(
        _first_present(raw, ("body_pose", "pred_body_pose"), default=_nested_param(params, "body_pose")),
        required=False,
    )
    if not body_pose and len(pred_poses) > 3:
        body_pose = pred_poses[3:]
    betas = _float_list(
        _first_present(raw, ("betas", "pred_betas"), default=_nested_param(params, "betas")),
        required=False,
    )
    camera_translation = _vector3(
        _first_present(
            raw,
            ("camera_translation", "pred_transl", "transl"),
            default=_nested_param(params, "transl"),
        ),
        default=[0.0, 0.0, 0.0],
    )
    return {
        "source_path": str(source_path),
        "frame_idx": frame_idx,
        "person_index": _optional_int(raw.get("person_index")),
        "confidence": _confidence(_first_present(raw, ("confidence", "score", "pred_conf"), default=1.0)),
        "bbox_xyxy": _bbox(raw.get("bbox_xyxy", raw.get("bbox"))),
        "joints_camera": joints_camera,
        "vertices_camera": vertices_camera,
        "global_orient": global_orient,
        "body_pose": body_pose,
        "betas": betas,
        "camera_translation": camera_translation,
    }


def _match_predictions_to_schedule(
    *,
    predictions: Sequence[Mapping[str, Any]],
    body_execution: Mapping[str, Any],
    track_frames: Mapping[tuple[int, int], Any],
    fps: float,
    min_assignment_iou: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions_by_frame: dict[int, list[Mapping[str, Any]]] = {}
    for prediction in predictions:
        predictions_by_frame.setdefault(int(prediction["frame_idx"]), []).append(prediction)

    used_prediction_ids: set[int] = set()
    samples: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    scheduled_frames = body_execution.get("scheduled_frames", [])
    if not isinstance(scheduled_frames, list):
        raise ValueError("body_compute_execution.scheduled_frames must be a list")
    for scheduled in scheduled_frames:
        if not isinstance(scheduled, Mapping):
            continue
        frame_idx = int(scheduled.get("frame_idx"))
        target_player_ids = scheduled.get("target_player_ids", [])
        if not isinstance(target_player_ids, list):
            continue
        frame_predictions = predictions_by_frame.get(frame_idx, [])
        for player_id_value in target_player_ids:
            player_id = int(player_id_value)
            track_frame = track_frames.get((player_id, frame_idx))
            if track_frame is None:
                continue
            prediction = _best_prediction_for_track(
                frame_predictions,
                track_bbox=list(track_frame.bbox),
                used_prediction_ids=used_prediction_ids,
                min_assignment_iou=min_assignment_iou,
            )
            if prediction is None:
                continue
            used_prediction_ids.add(id(prediction))
            sample = _prediction_to_world_sample(
                prediction,
                frame_idx=frame_idx,
                player_id=player_id,
                t=float(track_frame.t),
                track_world_xy=list(track_frame.world_xy),
                track_conf=float(track_frame.conf),
            )
            samples.append(sample)
            assignments.append(
                {
                    "frame_idx": frame_idx,
                    "player_id": player_id,
                    "prediction_path": str(prediction["source_path"]),
                    "iou": _iou(prediction.get("bbox_xyxy"), list(track_frame.bbox)),
                }
            )
    return samples, assignments


def _prediction_to_world_sample(
    prediction: Mapping[str, Any],
    *,
    frame_idx: int,
    player_id: int,
    t: float,
    track_world_xy: Sequence[float],
    track_conf: float,
) -> dict[str, Any]:
    confidence = min(float(prediction["confidence"]), track_conf)
    return {
        "frame_idx": frame_idx,
        "player_id": player_id,
        "t": t,
        "confidence": confidence,
        "track_world_xy": list(track_world_xy),
        "camera_translation": list(prediction["camera_translation"]),
        "joints_camera": [list(joint) for joint in prediction["joints_camera"]],
        "vertices_camera": [list(vertex) for vertex in prediction["vertices_camera"]],
        "global_orient": list(prediction["global_orient"]),
        "body_pose": list(prediction["body_pose"]),
        "left_hand_pose": [],
        "right_hand_pose": [],
        "betas": list(prediction["betas"]),
    }


def _track_frames_by_player_and_index(tracks: Tracks) -> dict[tuple[int, int], Any]:
    result: dict[tuple[int, int], Any] = {}
    fps = float(tracks.fps)
    for player in tracks.players:
        for frame in player.frames:
            frame_idx = int(round(float(frame.t) * fps))
            result[(int(player.id), frame_idx)] = frame
    return result


def _best_prediction_for_track(
    predictions: Sequence[Mapping[str, Any]],
    *,
    track_bbox: Sequence[float],
    used_prediction_ids: set[int],
    min_assignment_iou: float,
) -> Mapping[str, Any] | None:
    available = [prediction for prediction in predictions if id(prediction) not in used_prediction_ids]
    if not available:
        return None
    best = max(
        available,
        key=lambda prediction: (
            _iou(prediction.get("bbox_xyxy"), track_bbox),
            float(prediction.get("confidence", 0.0)),
        ),
    )
    if _iou(best.get("bbox_xyxy"), track_bbox) < min_assignment_iou:
        return None
    return best


def _frame_idx(raw: Mapping[str, Any], *, source_path: Path) -> int:
    value = raw.get("frame_idx")
    if value is not None:
        return int(value)
    image_name = raw.get("image_name")
    candidates = [str(image_name)] if image_name is not None else []
    candidates.append(source_path.stem)
    for candidate in candidates:
        match = _FRAME_RE.search(candidate)
        if match:
            return int(match.group(1))
    raise ValueError(f"could not infer frame_idx for SAT-HMR prediction: {source_path}")


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str], *, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _nested_param(params: Mapping[str, Any], key: str) -> Any:
    value = params.get(key)
    return _squeeze(value)


def _squeeze(value: Any) -> Any:
    item = _to_python(value)
    while isinstance(item, list) and len(item) == 1 and isinstance(item[0], list):
        item = item[0]
    return item


def _vector3(value: Any, *, default: Sequence[float]) -> list[float]:
    value = _squeeze(value)
    if value is None or value == []:
        value = default
    values = _float_list(value)
    if len(values) != 3:
        raise ValueError("expected a 3-vector")
    return values


def _vector3_list(value: Any, *, required: bool = True) -> list[list[float]]:
    value = _to_python(value)
    if value is None:
        if required:
            return []
        return []
    if isinstance(value, list) and len(value) == 1 and value and isinstance(value[0], list):
        inner = value[0]
        if inner and isinstance(inner[0], list):
            value = inner
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("expected a sequence of 3-vectors")
    result: list[list[float]] = []
    for item in value:
        vector = _float_list(item)
        if len(vector) != 3:
            raise ValueError("expected a sequence of 3-vectors")
        result.append(vector)
    return result


def _float_list(value: Any, *, required: bool = True) -> list[float]:
    value = _squeeze(value)
    if value is None:
        if required:
            raise ValueError("expected numeric sequence")
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("expected numeric sequence")
    result: list[float] = []
    for item in value:
        number = float(item)
        if not math.isfinite(number):
            raise ValueError("expected finite numeric sequence")
        result.append(number)
    return result


def _bbox(value: Any) -> list[float] | None:
    if value is None:
        return None
    bbox = _float_list(value)
    if len(bbox) != 4:
        raise ValueError("bbox must contain four values")
    return bbox


def _confidence(value: Any) -> float:
    number = float(_squeeze(value))
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError("confidence must be in [0, 1]")
    return number


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _to_python(value: Any) -> Any:
    item = value
    for method_name in ("detach", "cpu"):
        method = getattr(item, method_name, None)
        if callable(method):
            item = method()
    tolist = getattr(item, "tolist", None)
    if callable(tolist):
        return tolist()
    return item


def _iou(left: Any, right: Sequence[float]) -> float:
    if left is None:
        return 0.0
    lx1, ly1, lx2, ly2 = [float(value) for value in left]
    rx1, ry1, rx2, ry2 = [float(value) for value in right]
    ix1 = max(lx1, rx1)
    iy1 = max(ly1, ry1)
    ix2 = min(lx2, rx2)
    iy2 = min(ly2, ry2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    denom = left_area + right_area - inter
    return inter / denom if denom > 0.0 else 0.0


def _scheduled_player_frame_count(body_execution: Mapping[str, Any]) -> int:
    summary = body_execution.get("summary")
    if isinstance(summary, Mapping) and summary.get("scheduled_player_frame_count") is not None:
        return int(summary["scheduled_player_frame_count"])
    total = 0
    scheduled = body_execution.get("scheduled_frames", [])
    if isinstance(scheduled, list):
        for frame in scheduled:
            if isinstance(frame, Mapping) and isinstance(frame.get("target_player_ids"), list):
                total += len(frame["target_player_ids"])
    return total


def _read_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_optional_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    return _read_json_object(candidate)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "DEFAULT_SAT_HMR_MAX_ROOT_SPEED_MPS",
    "DEFAULT_SAT_HMR_MAX_TRACK_ANCHOR_SMOOTHING_RESIDUAL_M",
    "build_sat_hmr_body_fallback",
]
