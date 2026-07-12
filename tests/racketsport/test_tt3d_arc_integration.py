from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from threed.racketsport import ball_arc_chain
from threed.racketsport.ball_arc_solver import (
    AnchorEvent,
    BallArcSolverConfig,
    BallObservation,
    PhysicsParameters,
    fit_flight_segment,
)


BALL_RADIUS_M = 0.0371


def _camera_calibration() -> dict[str, Any]:
    position = np.asarray((0.0, -10.0, 5.0), dtype=float)
    target = np.asarray((0.0, 2.0, 1.0), dtype=float)
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0), dtype=float))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack((right, down, forward))
    translation = -rotation @ position
    return {
        "intrinsics": {"fx": 900.0, "fy": 900.0, "cx": 640.0, "cy": 360.0},
        "extrinsics": {
            "R": [[float(value) for value in row] for row in rotation],
            "t": [float(value) for value in translation],
        },
    }


def _project(calibration: dict[str, Any], point: tuple[float, float, float]) -> tuple[float, float]:
    rotation = np.asarray(calibration["extrinsics"]["R"], dtype=float)
    translation = np.asarray(calibration["extrinsics"]["t"], dtype=float)
    camera = rotation @ np.asarray(point, dtype=float) + translation
    intrinsics = calibration["intrinsics"]
    return (
        float(intrinsics["fx"] * camera[0] / camera[2] + intrinsics["cx"]),
        float(intrinsics["fy"] * camera[1] / camera[2] + intrinsics["cy"]),
    )


def _piecewise_track() -> dict[str, Any]:
    calibration = _camera_calibration()
    bounce_time = 0.5
    bounce = np.asarray((0.0, 2.0, BALL_RADIUS_M), dtype=float)
    incoming = np.asarray((2.0, 4.0, -5.0), dtype=float)
    outgoing = np.asarray((1.7, 3.4, 3.25), dtype=float)
    gravity = np.asarray((0.0, 0.0, -9.80665), dtype=float)
    frames = []
    for frame, pts_s in enumerate(np.linspace(0.1, 0.9, 25)):
        dt = float(pts_s) - bounce_time
        velocity = incoming if dt <= 0.0 else outgoing
        point = bounce + velocity * dt + 0.5 * gravity * dt * dt
        frames.append(
            {
                "frame": frame,
                "t": float(pts_s),
                "xy": list(_project(calibration, tuple(float(value) for value in point))),
                "conf": 0.9,
                "visible": True,
            }
        )
    return {"schema_version": 1, "source": "unit", "fps": 30.0, "frames": frames}


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_joint_anchor_sidecar_persists_all_hypotheses_and_preserves_raw_bytes(tmp_path: Path) -> None:
    ball_track = _piecewise_track()
    ball_track_path = _write_json(tmp_path / "ball_track.json", ball_track)
    before = _sha256(ball_track_path)
    baseline = {
        "segments": [
            {
                "segment_id": 4,
                "status": "fit_bvp_fallback",
                "frame_start": 0,
                "frame_end": 24,
            }
        ]
    }

    sidecar, anchors = ball_arc_chain._build_joint_anchor_candidate_sidecar(
        clip="unit_tt3d",
        ball_track=ball_track,
        calibration=_camera_calibration(),
        frame_times=None,
        net_plane=None,
        baseline_artifact=baseline,
        source_ball_track_path=ball_track_path,
        solver_config=ball_arc_chain.default_ball_arc_solver_config(),
        generated_at="2026-07-12T00:00:00Z",
    )

    assert _sha256(ball_track_path) == before
    assert sidecar["inputs"]["raw_observation_sha256_before"] == before
    assert sidecar["inputs"]["raw_observation_sha256_after"] == before
    assert sidecar["inputs"]["raw_observations_byte_identical"] is True
    assert sidecar["candidate_flag_default"] is False
    assert sidecar["summary"]["hypothesis_count"] >= 1
    assert len(anchors) == 1
    assert anchors[0].kind == "bounce"
    assert anchors[0].status == "candidate_hypothesis"
    assert anchors[0].details is not None and anchors[0].details["measured"] is False

    hypotheses = sidecar["windows"][0]["hypotheses"]
    assert hypotheses[0]["integration"]["disposition"] == "proposed_rank_1"
    assert all(
        hypothesis["integration"]["disposition"] == "rejected_lower_rank"
        for hypothesis in hypotheses[1:]
    )
    serialized = json.dumps(sidecar, sort_keys=True)
    assert '"measured": true' not in serialized.lower()
    assert '"marks_measured": false' in serialized.lower()

    chosen_id = anchors[0].anchor_id
    final_artifact = {
        "event_selection": {
            "selected": [{"anchor_id": chosen_id, "selection_reason": "reduced_residual_and_plausible"}],
            "rejected": [],
        },
        "segments": [
            {
                "segment_id": 0,
                "status": "fit",
                "frame_start": 0,
                "frame_end": 24,
                "inlier_count": 25,
                "outlier_count": 0,
                "max_reprojection_error_px": 1.0,
            }
        ],
    }
    ball_arc_chain._finalize_joint_anchor_candidate_sidecar(sidecar, final_artifact)
    ball_arc_chain._attach_joint_anchor_fail_closed_audit(
        sidecar,
        final_artifact,
        {"summary": {"fail_closed_suppressed_segment_ids": []}},
    )
    assert sidecar["summary"]["chosen_anchor_count"] == 1
    assert sidecar["summary"]["solver_rejected_anchor_count"] == 0
    assert sidecar["fail_closed_audit"]["fail_open_segment_ids"] == []
    assert sidecar["fail_closed_audit"]["fail_open_sample_count"] == 0


def _no_drag_point(
    p0: tuple[float, float, float],
    v0: tuple[float, float, float],
    t: float,
) -> tuple[float, float, float]:
    return (
        p0[0] + v0[0] * t,
        p0[1] + v0[1] * t,
        p0[2] + v0[2] * t - 0.5 * 9.80665 * t * t,
    )


def _pinning_fixture() -> tuple[dict[str, Any], AnchorEvent, AnchorEvent, list[BallObservation]]:
    calibration = _camera_calibration()
    t1 = 0.6
    p0 = (0.0, -1.0, 1.0)
    p1 = (0.6, 1.0, BALL_RADIUS_M)
    v0 = (
        (p1[0] - p0[0]) / t1,
        (p1[1] - p0[1]) / t1,
        (p1[2] - p0[2] + 0.5 * 9.80665 * t1 * t1) / t1,
    )
    start = AnchorEvent("start", "contact", 0.0, 0, p0, 0.04, "candidate_hypothesis", immovable=False)
    end = AnchorEvent("end", "bounce", t1, 18, p1, 0.04, "candidate_hypothesis", immovable=False)
    observations = [
        BallObservation(
            frame=frame,
            t=frame / 30.0,
            xy=_project(calibration, _no_drag_point(p0, v0, frame / 30.0)),
            confidence=0.95,
            visible=True,
        )
        for frame in range(19)
    ]
    observations[7] = replace(observations[7], xy=(1550.0, 150.0))
    return calibration, start, end, observations


def test_pinning_flag_is_default_off_and_dedicated_pass_keeps_tail_visible() -> None:
    calibration, start, end, observations = _pinning_fixture()
    base = BallArcSolverConfig(
        max_reprojection_inlier_px=8.0,
        robust_pixel_sigma=2.0,
        enable_event_discovery=False,
        enable_event_subset_selection=False,
    )
    implicit_off = fit_flight_segment(
        segment_id=0,
        start_anchor=start,
        end_anchor=end,
        observations=observations,
        calibration=calibration,
        physics=PhysicsParameters.no_drag(),
        config=base,
    )
    explicit_off = fit_flight_segment(
        segment_id=0,
        start_anchor=start,
        end_anchor=end,
        observations=observations,
        calibration=calibration,
        physics=PhysicsParameters.no_drag(),
        config=replace(base, enable_both_ends_pinning_inlier_pass=False),
    )
    assert implicit_off.to_json() == explicit_off.to_json()

    pinned = fit_flight_segment(
        segment_id=0,
        start_anchor=start,
        end_anchor=end,
        observations=observations,
        calibration=calibration,
        physics=PhysicsParameters.no_drag(),
        config=replace(base, enable_both_ends_pinning_inlier_pass=True),
    )
    diagnostics = pinned.diagnostics or {}
    pass_report = diagnostics["both_ends_pinning_inlier_pass"]
    assert pass_report["enabled"] is True
    assert pass_report["status"] == "refit"
    assert pass_report["both_ends_pinned"] is True
    assert pass_report["refit_observation_count"] == len(observations) - 1
    assert diagnostics["endpoint_refinement"]["frozen_for_candidate_association"] is True
    assert diagnostics["endpoint_refinement"]["delta_p0_m"] == [0.0, 0.0, 0.0]
    assert diagnostics["endpoint_refinement"]["delta_p1_m"] == [0.0, 0.0, 0.0]
    assert pinned.start_anchor.world_xyz == pytest.approx(start.world_xyz)
    assert pinned.end_anchor.world_xyz == pytest.approx(end.world_xyz)
    assert 7 in pinned.outlier_frames
    assert pinned.max_reprojection_error_px is not None and pinned.max_reprojection_error_px > 50.0


def test_joint_anchor_flags_refuse_invalid_or_rejected_combinations(tmp_path: Path) -> None:
    ball_track_path = _write_json(tmp_path / "ball_track.json", _piecewise_track())
    calibration_path = _write_json(tmp_path / "court_calibration.json", _camera_calibration())

    with pytest.raises(ValueError, match="pinning requires"):
        ball_arc_chain.run_default_ball_arc_chain(
            clip="unit",
            ball_track_path=ball_track_path,
            court_calibration_path=calibration_path,
            out_dir=tmp_path / "pinning_without_tt3d",
            enable_joint_anchor_pinning=True,
        )
    with pytest.raises(ValueError, match="rejected RANSAC"):
        ball_arc_chain.run_default_ball_arc_chain(
            clip="unit",
            ball_track_path=ball_track_path,
            court_calibration_path=calibration_path,
            out_dir=tmp_path / "tt3d_ransac",
            enable_joint_anchor_search=True,
            enable_ransac_arc_gate=True,
        )


def test_pinning_config_rejects_invalid_inlier_floor() -> None:
    with pytest.raises(ValueError, match="pinning_min_inliers"):
        BallArcSolverConfig(pinning_min_inliers=1)
