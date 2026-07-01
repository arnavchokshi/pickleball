"""Strict builders for court-positioning JSON artifacts."""

from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence

from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINT_BY_NAME
from threed.racketsport.court_positioning import CourtBoundaryDecision, MetricCourtPlacement, transform_court_to_world
from threed.racketsport.player_grounding import GroundedFoot
from threed.racketsport.schemas import (
    CallsArtifact,
    CourtCalibration,
    CourtCallEvent,
    CourtKeypoints,
    PICKLEBALL_COURT_KEYPOINT_NAMES,
    PlayerGroundArtifact,
    PlayerGroundFoot,
    PlayerGroundFrame,
)


def build_court_keypoints_artifact(
    *,
    frame_indexes: Sequence[int],
    keypoints: Mapping[str, Mapping[str, Any]],
    target_court_score: float,
    source: str,
    coordinate_space: Literal["undistorted_source_video_pixels"] = "undistorted_source_video_pixels",
) -> dict[str, Any]:
    """Build the aggregated Stage B court_keypoints.json artifact."""

    frame_index_payload = [int(frame_index) for frame_index in frame_indexes]
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_court_keypoints",
        "frame_indexes": frame_index_payload,
        "coordinate_space": coordinate_space,
        "keypoints": [
            _court_keypoint_payload(name, keypoints.get(name), default_inlier_frames=frame_index_payload)
            for name in PICKLEBALL_COURT_KEYPOINT_NAMES
        ],
        "target_court_score": float(target_court_score),
        "source": source,
        "not_gate_verified": True,
    }
    return CourtKeypoints.model_validate(payload).model_dump(mode="json")


def build_metric_court_calibration_artifact(
    *,
    placement: MetricCourtPlacement,
    intrinsics: Mapping[str, Any],
    homography: Sequence[Sequence[float]],
    image_keypoints: Mapping[str, Mapping[str, Any]],
    extrinsics: Mapping[str, Any],
    reprojection_error_px: Mapping[str, Any],
    per_keypoint_residual_px: Sequence[float],
    gsd_model: Mapping[str, Any],
    capture_quality: Mapping[str, Any],
    source: str,
    solved_over_frames: Sequence[int],
    sport: Literal["pickleball", "tennis"] = "pickleball",
) -> dict[str, Any]:
    """Build the Stage C metric court_calibration.json payload."""

    image_pts: list[list[float]] = []
    world_pts: list[list[float]] = []
    for name in placement.solved_keypoints:
        raw = image_keypoints.get(name)
        if raw is None:
            raise ValueError(f"missing image keypoint for solved keypoint: {name}")
        uv = raw.get("uv", raw.get("xy"))
        if not isinstance(uv, Sequence) or isinstance(uv, (str, bytes)):
            raise ValueError(f"image keypoint {name} requires uv or xy")
        image_pts.append([float(uv[0]), float(uv[1])])
        world_pts.append(transform_court_to_world(PICKLEBALL_KEYPOINT_BY_NAME[name].world_xyz_m, placement.T_world_court))

    payload = {
        "schema_version": 1,
        "sport": sport,
        "coordinate_frame": "court_netcenter_z_up_m",
        "T_world_court": [[float(value) for value in row] for row in placement.T_world_court],
        "homography": [[float(value) for value in row] for row in homography],
        "intrinsics": dict(intrinsics),
        "extrinsics": dict(extrinsics),
        "reprojection_error_px": dict(reprojection_error_px),
        "per_keypoint_residual_px": [float(value) for value in per_keypoint_residual_px],
        "metric_confidence": placement.metric_confidence,
        "gsd_model": dict(gsd_model),
        "capture_quality": dict(capture_quality),
        "image_pts": image_pts,
        "world_pts": world_pts,
        "source": source,
        "solved_over_frames": [int(frame) for frame in solved_over_frames],
    }
    return CourtCalibration.model_validate(payload).model_dump(mode="json")


def build_player_ground_artifact(
    *,
    fps: float,
    players: Sequence[Mapping[str, Any]],
    source: str = "metric_floor_pose_grounding_v1",
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_player_ground",
        "fps": float(fps),
        "players": [_player_payload(player) for player in players],
        "source": source,
        "not_gate_verified": True,
    }
    return PlayerGroundArtifact.model_validate(payload).model_dump(mode="json")


def build_call_event(
    *,
    t: float,
    player_id: int,
    foot: Literal["L", "R"],
    decision: CourtBoundaryDecision,
    frames: Sequence[int],
) -> dict[str, Any]:
    payload = {
        "t": float(t),
        "player_id": int(player_id),
        "foot": foot,
        "boundary": _artifact_boundary(decision.boundary),
        "decision": decision.decision,
        "signed_dist_m": float(decision.signed_dist_m),
        "sigma_p_m": float(decision.sigma_p_m),
        "frames": [int(frame) for frame in frames],
        "metric_confidence": decision.metric_confidence,
        "capture_quality_grade": decision.capture_quality_grade,
    }
    return CourtCallEvent.model_validate(payload).model_dump(mode="json")


def build_calls_artifact(
    events: Sequence[Mapping[str, Any]],
    *,
    source: str,
) -> dict[str, Any]:
    event_payloads = [CourtCallEvent.model_validate(event).model_dump(mode="json") for event in events]
    too_close = sum(1 for event in event_payloads if event["decision"] == "too_close_to_call")
    summary = {
        "total_events": len(event_payloads),
        "hard_call_count": len(event_payloads) - too_close,
        "too_close_to_call_count": too_close,
        "status": "not_gate_verified",
    }
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_court_calls",
        "source": source,
        "events": event_payloads,
        "summary": summary,
        "not_gate_verified": True,
    }
    return CallsArtifact.model_validate(payload).model_dump(mode="json")


def _court_keypoint_payload(
    name: str,
    raw: Mapping[str, Any] | None,
    *,
    default_inlier_frames: Sequence[int],
) -> dict[str, Any]:
    if raw is None:
        raise ValueError("court_keypoints must contain exactly the 15 canonical pickleball keypoints in schema order")
    xy = raw.get("uv", raw.get("xy"))
    if not isinstance(xy, Sequence) or isinstance(xy, (str, bytes)):
        raise ValueError(f"court keypoint {name} requires uv or xy")
    inlier_frames = raw.get("inlier_frames", default_inlier_frames)
    if not isinstance(inlier_frames, Sequence) or isinstance(inlier_frames, (str, bytes)):
        raise ValueError(f"court keypoint {name} inlier_frames must be a sequence")
    return {
        "name": name,
        "uv": [float(xy[0]), float(xy[1])],
        "confidence": float(raw["confidence"]),
        "inlier_frames": [int(frame) for frame in inlier_frames],
        "recovered": bool(raw.get("recovered", False)),
    }


def _player_payload(player: Mapping[str, Any]) -> dict[str, Any]:
    frames = player.get("frames")
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise ValueError("player frames must be a sequence")
    return {"id": int(player["id"]), "frames": [_frame_payload(frame) for frame in frames]}


def _frame_payload(frame: Mapping[str, Any]) -> dict[str, Any]:
    feet = frame.get("feet")
    if not isinstance(feet, Sequence) or isinstance(feet, (str, bytes)):
        raise ValueError("frame feet must be a sequence")
    foot_payloads = [_foot_payload(foot) for foot in feet]
    if {foot["side"] for foot in foot_payloads} != {"L", "R"} or len(foot_payloads) != 2:
        raise ValueError("player_ground frame must include both L and R feet")
    payload = {
        "t": float(frame["t"]),
        "feet": foot_payloads,
        "root_world": [float(value) for value in frame["root_world"]],
        "joints_world": [[float(value) for value in joint] for joint in frame.get("joints_world", [])],
    }
    if frame.get("mesh_ref") is not None:
        payload["mesh_ref"] = str(frame["mesh_ref"])
    return PlayerGroundFrame.model_validate(payload).model_dump(mode="json")


def _foot_payload(foot: GroundedFoot | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(foot, GroundedFoot):
        raw = {
            "side": foot.side,
            "court_xy": foot.court_xy,
            "height_m": foot.height_m,
            "contact": foot.contact,
            "sigma_p_m": foot.sigma_p_m,
            "confidence": foot.confidence,
            "world_xyz": foot.world_xyz,
            "source_points": list(foot.source_points),
        }
    else:
        raw = dict(foot)
    return PlayerGroundFoot.model_validate(raw).model_dump(mode="json")


def _artifact_boundary(boundary: str) -> str:
    if boundary in {"near_kitchen", "far_kitchen"}:
        return "kitchen"
    if boundary in {"sideline", "baseline", "centerline"}:
        return boundary
    raise ValueError(f"unsupported boundary: {boundary}")
