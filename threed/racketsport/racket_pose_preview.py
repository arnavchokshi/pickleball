"""Preview-only paddle pose artifact builder from explicit four-corner candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .racket6dof import estimate_planar_paddle_pose_with_diagnostics, validate_paddle_dimensions
from .schemas import CourtCalibration, RacketCandidateFrame, RacketCandidates, RacketPose, validate_artifact_file


SOURCE_MODE = "explicit_four_corner_candidates_pnp_ippe_preview"


def build_racket_pose_preview(
    court_calibration: CourtCalibration,
    racket_candidates: RacketCandidates,
    *,
    max_reprojection_error_px: float = 6.0,
    ambiguity_margin_threshold_px: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a non-gating paddle pose preview from explicit paddle corners.

    Unlike ``RacketStageRunner``, this preview keeps ambiguous IPPE solutions so
    the review world can show a paddle mesh for human inspection. It still drops
    schema-invalid candidates and candidates above the reprojection threshold.
    """

    if max_reprojection_error_px < 0.0:
        raise ValueError("max_reprojection_error_px must be non-negative")
    if ambiguity_margin_threshold_px < 0.0:
        raise ValueError("ambiguity_margin_threshold_px must be non-negative")

    camera_matrix = _camera_matrix(court_calibration)
    dist_coeffs = court_calibration.intrinsics.dist
    players: list[dict[str, Any]] = []
    summary = {
        "schema_version": 1,
        "source_mode": SOURCE_MODE,
        "candidate_player_count": len(racket_candidates.players),
        "preview_player_count": 0,
        "candidate_frame_count": 0,
        "preview_frame_count": 0,
        "invalid_candidate_count": 0,
        "rejected_high_reprojection_count": 0,
        "ambiguous_frame_count": 0,
        "max_reprojection_error_px": float(max_reprojection_error_px),
        "ambiguity_margin_threshold_px": float(ambiguity_margin_threshold_px),
        "not_gate_verified": True,
    }

    for player_payload in racket_candidates.players:
        paddle_dims = validate_paddle_dimensions(player_payload.paddle_dims_in)
        paddle_dims_dict = {"length": paddle_dims.length_in, "width": paddle_dims.width_in}
        frames = []
        for frame_payload in player_payload.frames:
            summary["candidate_frame_count"] += 1
            try:
                frame = _pose_frame(
                    frame_payload,
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                    paddle_dims_in=paddle_dims_dict,
                    ambiguity_margin_threshold_px=ambiguity_margin_threshold_px,
                )
            except (TypeError, ValueError):
                summary["invalid_candidate_count"] += 1
                continue
            if frame["reprojection_error_px"] > max_reprojection_error_px:
                summary["rejected_high_reprojection_count"] += 1
                continue
            if frame["ambiguous"]:
                summary["ambiguous_frame_count"] += 1
            summary["preview_frame_count"] += 1
            frames.append(frame)
        if frames:
            players.append(
                {
                    "id": player_payload.id,
                    "paddle_dims_in": paddle_dims_dict,
                    "frames": frames,
                    "contacts": [],
                }
            )

    summary["preview_player_count"] = len(players)
    pose = RacketPose.model_validate(
        {
            "schema_version": 1,
            "fps": racket_candidates.fps,
            "world_frame": "camera",
            "translation_unit": "cm",
            "players": players,
        }
    )
    return pose.model_dump(mode="json"), summary


def build_racket_pose_preview_from_files(
    *,
    court_calibration_path: str | Path,
    racket_candidates_path: str | Path,
    max_reprojection_error_px: float = 6.0,
    ambiguity_margin_threshold_px: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    calibration = validate_artifact_file("court_calibration", Path(court_calibration_path))
    if not isinstance(calibration, CourtCalibration):
        raise ValueError("court calibration artifact did not parse as CourtCalibration")
    candidates = validate_artifact_file("racket_candidates", Path(racket_candidates_path))
    if not isinstance(candidates, RacketCandidates):
        raise ValueError("racket candidates artifact did not parse as RacketCandidates")
    return build_racket_pose_preview(
        calibration,
        candidates,
        max_reprojection_error_px=max_reprojection_error_px,
        ambiguity_margin_threshold_px=ambiguity_margin_threshold_px,
    )


def write_racket_pose_preview(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pose_frame(
    frame_payload: RacketCandidateFrame,
    *,
    camera_matrix: Sequence[Sequence[float]],
    dist_coeffs: Sequence[float],
    paddle_dims_in: Mapping[str, float],
    ambiguity_margin_threshold_px: float,
) -> dict[str, Any]:
    estimate = estimate_planar_paddle_pose_with_diagnostics(
        frame_payload.corners_px,
        camera_matrix,
        paddle_dims_in,
        dist_coeffs=dist_coeffs,
        ambiguity_margin_threshold_px=ambiguity_margin_threshold_px,
    )
    pose = estimate.pose
    return {
        "t": frame_payload.t,
        "pose_se3": {
            "R": [list(row) for row in pose.R],
            "t": list(pose.t),
        },
        "conf": max(0.0, min(1.0, frame_payload.conf * pose.confidence)),
        "world_frame": "camera",
        "translation_unit": "cm",
        "source": f"{frame_payload.source}:pnp_ippe_preview",
        "reprojection_error_px": estimate.reprojection_error_px,
        "ambiguous": estimate.ambiguous,
    }


def _camera_matrix(calibration: CourtCalibration) -> list[list[float]]:
    intrinsics = calibration.intrinsics
    return [
        [intrinsics.fx, 0.0, intrinsics.cx],
        [0.0, intrinsics.fy, intrinsics.cy],
        [0.0, 0.0, 1.0],
    ]


__all__ = [
    "SOURCE_MODE",
    "build_racket_pose_preview",
    "build_racket_pose_preview_from_files",
    "write_racket_pose_preview",
]
