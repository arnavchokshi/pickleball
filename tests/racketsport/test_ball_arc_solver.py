from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.racketsport import solve_ball_arcs as solve_ball_arcs_cli
from threed.racketsport.ball_arc_solver import (
    AnchorEvent,
    BallArcSolverConfig,
    BallObservation,
    PhysicsParameters,
    build_bounce_anchor,
    fit_flight_segment,
    fit_weak_flight_segment,
    order_event_anchors,
    solve_ball_arc_track,
)


BALL_RADIUS_M = 0.0371
GRAVITY = 9.80665


def test_bounce_anchor_intersects_ball_radius_plane_and_uses_gsd_sigma() -> None:
    calibration = _projection_calibration(with_gsd=True)
    world = (0.8, -1.1, BALL_RADIUS_M)
    xy = _project(calibration, world)

    anchor = build_bounce_anchor(
        {"frame": 18, "t": 0.6, "review_id": "bounce_0001"},
        calibration,
        ball_radius_m=BALL_RADIUS_M,
        ball_xy=xy,
    )

    assert anchor.kind == "bounce"
    assert anchor.status == "human_reviewed"
    assert anchor.immovable is True
    assert anchor.frame == 18
    assert anchor.world_xyz == pytest.approx(world, abs=1e-6)
    assert 0.05 <= anchor.sigma_m <= 0.20
    assert anchor.sigma_m > 0.05


def test_order_event_anchors_sorts_and_prefers_human_reviewed_duplicates() -> None:
    proposed_duplicate = _anchor("b1-proposed", "bounce", 1.0, (1.0, 0.0, BALL_RADIUS_M), status="solver_proposed")
    reviewed_duplicate = _anchor("b1-reviewed", "bounce", 1.0, (1.1, 0.0, BALL_RADIUS_M), status="human_reviewed")
    contact = _anchor("c0", "contact", 0.2, (0.0, -1.0, 1.0), status="contact_prior", player_id=3)
    later = _anchor("c2", "contact", 1.7, (1.5, 1.0, 1.1), status="contact_prior", player_id=1)

    ordered = order_event_anchors([later, proposed_duplicate, contact, reviewed_duplicate])

    assert [anchor.anchor_id for anchor in ordered] == ["c0", "b1-reviewed", "c2"]
    assert ordered[1].world_xyz == pytest.approx((1.1, 0.0, BALL_RADIUS_M))
    assert ordered[1].immovable is True


def test_fit_flight_segment_recovers_velocity_and_prunes_planted_fp() -> None:
    calibration = _projection_calibration()
    physics = PhysicsParameters.no_drag()
    t0 = 0.0
    t1 = 0.5
    p0 = (0.2, -2.0, 1.15)
    v0 = (2.4, 3.0, _vz_for_endpoint(p0[2], BALL_RADIUS_M, t1 - t0))
    p1 = _no_drag_position(p0, v0, t1)
    start = _anchor("contact-0", "contact", t0, p0, sigma_m=0.04)
    end = _anchor("bounce-0", "bounce", t1, p1, sigma_m=0.04, status="human_reviewed")
    observations = _observations_from_arc(calibration, p0, v0, t0=t0, t1=t1, fps=60.0)
    observations.append(BallObservation(frame=114, t=14.0 / 60.0, xy=(1550.0, 200.0), confidence=0.95, visible=True))

    result = fit_flight_segment(
        segment_id=0,
        start_anchor=start,
        end_anchor=end,
        observations=observations,
        calibration=calibration,
        physics=physics,
        config=BallArcSolverConfig(max_reprojection_inlier_px=8.0, robust_pixel_sigma=2.0),
    )

    assert result.status == "fit"
    assert result.initial_velocity_mps == pytest.approx(v0, abs=0.12)
    assert result.outlier_count == 1
    assert result.inlier_count >= len(observations) - 1
    assert result.reprojection_rmse_px < 2.0


def test_fit_flight_segment_blocks_invalid_bounds_from_below_floor_anchor() -> None:
    calibration = _projection_calibration()
    start = _anchor("contact-under-floor", "contact", 0.0, (2.6, 6.5, -1.77), sigma_m=0.35)
    end = _anchor("contact-next", "contact", 0.25, (2.2, 6.4, -0.4), sigma_m=0.35)
    observations = [
        BallObservation(frame=frame, t=frame / 60.0, xy=_project(calibration, (2.4, 6.45, 0.2)), confidence=0.9, visible=True)
        for frame in range(1, 6)
    ]

    result = fit_flight_segment(
        segment_id=7,
        start_anchor=start,
        end_anchor=end,
        observations=observations,
        calibration=calibration,
        physics=PhysicsParameters.no_drag(),
        config=BallArcSolverConfig(),
    )

    assert result.status == "blocked:invalid_segment_bounds"
    assert result.physical_sanity["violations"] == ["invalid_segment_bounds"]


def test_weak_single_anchor_fit_recovers_depth_from_apparent_size() -> None:
    calibration = _projection_calibration()
    physics = PhysicsParameters.no_drag()
    start = _anchor("contact-weak-start", "contact", 0.0, (0.1, -1.4, 1.05), sigma_m=0.08)
    velocity = (1.2, 3.4, _vz_for_endpoint(start.world_xyz[2], 1.25, 0.8))
    observations = _observations_from_arc(
        calibration,
        start.world_xyz,
        velocity,
        t0=0.0,
        t1=0.8,
        fps=60.0,
        include_size=True,
    )

    weak = fit_weak_flight_segment(
        segment_id=12,
        anchor=start,
        observations=observations,
        calibration=calibration,
        physics=physics,
        config=BallArcSolverConfig(
            min_segment_observations=4,
            weak_segment_min_observations=4,
            robust_pixel_sigma=3.0,
            weak_size_depth_sigma_m=0.35,
        ),
    )

    assert weak.status == "fit_weak"
    assert weak.start_anchor.anchor_id == start.anchor_id
    assert weak.end_anchor.status == "weak_ray_endpoint"
    assert weak.initial_velocity_mps == pytest.approx(velocity, abs=0.35)
    assert weak.physical_sanity["violation"] is False
    assert weak.size_residuals_m["count"] >= 20
    assert weak.predict(0.8, physics, BallArcSolverConfig()) == pytest.approx(
        _no_drag_position(start.world_xyz, velocity, 0.8),
        abs=0.22,
    )


def test_solve_track_adds_arc_weak_tail_after_rejected_rally_endpoint() -> None:
    calibration = _projection_calibration()
    physics = PhysicsParameters.no_drag()
    fps = 60.0
    bounce0 = (0.0, -2.0, BALL_RADIUS_M)
    bounce1_t = 0.42
    bounce1 = (0.25, -0.2, BALL_RADIUS_M)
    first_velocity = (
        (bounce1[0] - bounce0[0]) / bounce1_t,
        (bounce1[1] - bounce0[1]) / bounce1_t,
        _vz_for_endpoint(BALL_RADIUS_M, BALL_RADIUS_M, bounce1_t),
    )
    weak_velocity = (0.45, 2.1, 4.6)
    frames: list[dict] = []
    for frame in range(int(1.2 * fps) + 1):
        t = frame / fps
        if t <= bounce1_t:
            world = _no_drag_position(bounce0, first_velocity, t)
        else:
            world = _no_drag_position(bounce1, weak_velocity, t - bounce1_t)
        frames.append(
            {
                "t": t,
                "xy": list(_project(calibration, world)),
                "diameter_px": _apparent_diameter_px(calibration, world),
                "size_conf": 0.9,
                "conf": 0.92,
                "visible": True,
                "world_xyz": None,
                "approx": False,
            }
        )

    artifact = solve_ball_arc_track(
        ball_track={"schema_version": 1, "fps": fps, "source": "synthetic", "frames": frames, "bounces": []},
        calibration=calibration,
        reviewed_bounces={
            "schema_version": 1,
            "status": "human_reviewed",
            "bounces": [
                {"frame": 0, "t": 0.0, "review_id": "bounce_start"},
                {"frame": int(round(bounce1_t * fps)), "t": bounce1_t, "review_id": "bounce_mid"},
            ],
        },
        rally_spans={"spans": [{"t0": 0.0, "t1": bounce1_t}]},
        physics=physics,
        config=BallArcSolverConfig(
            enable_event_discovery=False,
            enable_weak_segments=True,
            weak_segment_min_observations=4,
            selection_max_speed_mps=35.0,
            max_plausible_speed_mps=35.0,
            min_segment_observations=3,
            robust_pixel_sigma=2.0,
        ),
    )

    weak_frames = [frame for frame in artifact["frames"] if frame["band"] == "arc_weak"]
    assert artifact["summary"]["weak_segment_count"] == 1
    assert artifact["summary"]["arc_weak_count"] == len(weak_frames)
    assert len(weak_frames) >= 40
    assert artifact["summary"]["coverage_world_xyz_count"] == len(frames)
    assert artifact["validation"]["weak_segments"]["fit_count"] == 1
    assert artifact["validation"]["leave_one_out"]["ray_distance_m"]["count"] > 0


def test_solve_track_hides_weak_tail_when_speed_gate_fails() -> None:
    calibration = _projection_calibration()
    physics = PhysicsParameters.no_drag()
    fps = 60.0
    start = (0.0, -1.0, BALL_RADIUS_M)
    fast_velocity = (46.0, 0.0, _vz_for_endpoint(BALL_RADIUS_M, 0.8, 0.4))
    frames = []
    for frame in range(int(0.4 * fps) + 1):
        t = frame / fps
        world = _no_drag_position(start, fast_velocity, t)
        frames.append(
            {
                "t": t,
                "xy": list(_project(calibration, world)),
                "diameter_px": _apparent_diameter_px(calibration, world),
                "size_conf": 0.95,
                "conf": 0.9,
                "visible": True,
                "world_xyz": None,
                "approx": False,
            }
        )

    artifact = solve_ball_arc_track(
        ball_track={"schema_version": 1, "fps": fps, "source": "synthetic", "frames": frames, "bounces": []},
        calibration=calibration,
        reviewed_bounces={
            "schema_version": 1,
            "status": "human_reviewed",
            "bounces": [{"frame": 0, "t": 0.0, "review_id": "bounce_start"}],
        },
        rally_spans={"spans": [{"t0": 0.0, "t1": 0.0}]},
        physics=physics,
        config=BallArcSolverConfig(
            enable_event_discovery=False,
            enable_weak_segments=True,
            weak_segment_min_observations=4,
            selection_max_speed_mps=35.0,
            max_plausible_speed_mps=35.0,
            min_segment_observations=3,
        ),
    )

    assert artifact["summary"]["weak_segment_count"] == 0
    assert artifact["validation"]["weak_segments"]["rejected_count"] >= 1
    assert all(frame["band"] != "arc_weak" for frame in artifact["frames"])


def test_event_subset_selection_rejects_oversegmented_false_contacts() -> None:
    calibration = _projection_calibration()
    physics = PhysicsParameters.no_drag()
    fps = 60.0
    t0 = 0.0
    t1 = 2.0
    p0 = (-0.4, -3.0, BALL_RADIUS_M)
    v0 = (0.55, 3.0, _vz_for_endpoint(BALL_RADIUS_M, BALL_RADIUS_M, t1 - t0))
    observations = _observations_from_arc(calibration, p0, v0, t0=t0, t1=t1, fps=fps)
    frames = [
        {"t": obs.t, "xy": list(obs.xy), "conf": obs.confidence, "visible": True, "world_xyz": None, "approx": False}
        for obs in observations
    ]
    false_contacts = [
        _anchor(f"false-contact-{index}", "contact", t, (3.0 + index * 0.1, 5.0, 1.0), sigma_m=0.08)
        for index, t in enumerate((0.25, 0.5, 0.75, 1.0, 1.25, 1.5))
    ]

    artifact = solve_ball_arc_track(
        ball_track={"schema_version": 1, "fps": fps, "source": "synthetic", "frames": frames, "bounces": []},
        calibration=calibration,
        reviewed_bounces={
            "schema_version": 1,
            "status": "human_reviewed",
            "bounces": [
                {"frame": int(t0 * fps), "t": t0, "review_id": "bounce_start"},
                {"frame": int(t1 * fps), "t": t1, "review_id": "bounce_end"},
            ],
        },
        extra_anchors=false_contacts,
        physics=physics,
        config=BallArcSolverConfig(enable_event_discovery=False, max_plausible_speed_mps=35.0),
    )

    assert artifact["summary"]["segment_count"] == 1
    assert artifact["summary"]["fit_segment_count"] == 1
    assert artifact["segments"][0]["initial_speed_mps"] <= 35.0
    assert artifact["event_selection"]["selected_optional_count"] == 0
    assert artifact["event_selection"]["rejected_optional_count"] == len(false_contacts)
    assert {event["status"] for event in artifact["event_selection"]["rejected"]} == {"candidate_prediction"}


def test_solve_track_discovers_missing_bounce_and_refits_two_segments() -> None:
    calibration = _projection_calibration()
    physics = PhysicsParameters.no_drag()
    contact0 = _anchor("contact-0", "contact", 0.0, (0.0, -2.0, 1.05), sigma_m=0.06)
    bounce_t = 0.5
    contact1_t = 1.0
    v_before = (2.0, 3.0, _vz_for_endpoint(contact0.world_xyz[2], BALL_RADIUS_M, bounce_t))
    bounce_xyz = _no_drag_position(contact0.world_xyz, v_before, bounce_t)
    v_after = (
        2.0,
        3.0,
        _vz_for_endpoint(BALL_RADIUS_M, 1.05, contact1_t - bounce_t),
    )
    contact1 = _anchor("contact-1", "contact", contact1_t, _no_drag_position(bounce_xyz, v_after, contact1_t - bounce_t), sigma_m=0.06)
    frames = _piecewise_track_frames(
        calibration=calibration,
        p0=contact0.world_xyz,
        v0=v_before,
        bounce_t=bounce_t,
        v1=v_after,
        t_end=contact1_t,
        fps=60.0,
    )

    artifact = solve_ball_arc_track(
        ball_track={"schema_version": 1, "fps": 60.0, "source": "synthetic", "frames": frames, "bounces": []},
        calibration=calibration,
        extra_anchors=[contact0, contact1],
        physics=physics,
        config=BallArcSolverConfig(
            enable_event_discovery=True,
            discovery_reprojection_px=28.0,
            max_reprojection_inlier_px=8.0,
            robust_pixel_sigma=2.0,
        ),
    )

    assert artifact["status"] == "ran"
    assert artifact["summary"]["discovered_bounce_count"] == 1
    assert len(artifact["segments"]) == 2
    proposed = [anchor for anchor in artifact["anchors"] if anchor["status"] == "solver_proposed"]
    assert len(proposed) == 1
    assert proposed[0]["frame"] == pytest.approx(round(bounce_t * 60.0), abs=1)
    assert artifact["summary"]["coverage_world_xyz_count"] == len(frames)


def test_solve_ball_arcs_cli_writes_render_only_reference_artifact(tmp_path: Path) -> None:
    calibration = _projection_calibration()
    physics = PhysicsParameters.no_drag()
    contact0 = (0.0, -1.8, 1.0)
    bounce_t = 0.45
    contact1_t = 0.9
    v_before = (1.8, 3.2, _vz_for_endpoint(contact0[2], BALL_RADIUS_M, bounce_t))
    bounce_xyz = _no_drag_position(contact0, v_before, bounce_t)
    v_after = (1.8, 3.2, _vz_for_endpoint(BALL_RADIUS_M, 1.0, contact1_t - bounce_t))
    contact1 = _no_drag_position(bounce_xyz, v_after, contact1_t - bounce_t)
    frames = _piecewise_track_frames(
        calibration=calibration,
        p0=contact0,
        v0=v_before,
        bounce_t=bounce_t,
        v1=v_after,
        t_end=contact1_t,
        fps=60.0,
    )
    ball_path = _write_json(
        tmp_path / "ball_track.json",
        {"schema_version": 1, "fps": 60.0, "source": "synthetic", "frames": frames, "bounces": []},
    )
    calibration_path = _write_json(tmp_path / "court_calibration.json", calibration)
    reviewed_bounces_path = _write_json(
        tmp_path / "reviewed_ball_bounces.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_reviewed_ball_bounces",
            "status": "human_reviewed",
            "source": "human_review",
            "bounces": [{"frame": int(round(bounce_t * 60.0)), "t": bounce_t, "review_id": "bounce_0001"}],
        },
    )
    contact_path = _write_json(
        tmp_path / "contact_windows.json",
        {
            "schema_version": 1,
            "events": [
                _contact_event(0.0, 7),
                _contact_event(contact1_t, 7),
            ],
        },
    )
    skeleton_path = _write_json(
        tmp_path / "skeleton3d.json",
        _skeleton_payload([(0.0, contact0), (contact1_t, contact1)], player_id=7),
    )
    out_dir = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/solve_ball_arcs.py",
            "--clip",
            "synthetic_clip",
            "--ball-track",
            str(ball_path),
            "--court-calibration",
            str(calibration_path),
            "--contact-windows",
            str(contact_path),
            "--skeleton3d",
            str(skeleton_path),
            "--reviewed-bounces",
            str(reviewed_bounces_path),
            "--out-dir",
            str(out_dir),
            "--ball-type",
            "no_drag_test",
            "--contact-reach-offset-m",
            "0.0",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    stdout = json.loads(completed.stdout)
    artifact_path = out_dir / "synthetic_clip" / "ball_track_arc_solved.json"
    report_path = out_dir / "synthetic_clip" / "REPORT.md"
    assert stdout["artifacts"][0]["ball_track_arc_solved"] == str(artifact_path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "racketsport_ball_track_arc_solved"
    assert payload["render_only"] is True
    assert payload["not_for_detection_metrics"] is True
    assert len(payload["segments"]) == 2
    assert payload["summary"]["coverage_world_xyz_count"] == len(frames)
    assert payload["summary"]["human_reviewed_bounce_count"] == 1
    assert "process_video.py integration is intentionally not applied" in report_path.read_text(encoding="utf-8")


def test_solve_ball_arcs_cli_accepts_size_sidecar_and_reports_size_residuals(tmp_path: Path) -> None:
    calibration = _projection_calibration()
    physics = PhysicsParameters.no_drag()
    start = (0.0, -1.8, 1.0)
    bounce_t = 0.45
    v0 = (1.8, 3.2, _vz_for_endpoint(start[2], BALL_RADIUS_M, bounce_t))
    frames = _observations_from_arc(
        calibration,
        start,
        v0,
        t0=0.0,
        t1=bounce_t,
        fps=60.0,
        include_size=False,
    )
    track_frames = [
        {"t": obs.t, "xy": list(obs.xy), "conf": obs.confidence, "visible": True, "world_xyz": None, "approx": False}
        for obs in frames
    ]
    size_frames = [
        {
            "frame": obs.frame,
            "t": obs.t,
            "diameter_px": _apparent_diameter_px(calibration, _no_drag_position(start, v0, obs.t)),
            "confidence": 0.9,
            "source": "synthetic_test",
        }
        for obs in frames
    ]
    ball_path = _write_json(
        tmp_path / "ball_track.json",
        {"schema_version": 1, "fps": 60.0, "source": "synthetic", "frames": track_frames, "bounces": []},
    )
    calibration_path = _write_json(tmp_path / "court_calibration.json", calibration)
    size_path = _write_json(
        tmp_path / "ball_size_observations.json",
        {"schema_version": 1, "artifact_type": "racketsport_ball_size_observations", "frames": size_frames},
    )
    reviewed_bounces_path = _write_json(
        tmp_path / "reviewed_ball_bounces.json",
        {
            "schema_version": 1,
            "status": "human_reviewed",
            "bounces": [
                {"frame": 0, "t": 0.0, "review_id": "bounce_start"},
                {"frame": int(round(bounce_t * 60.0)), "t": bounce_t, "review_id": "bounce_end"},
            ],
        },
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/solve_ball_arcs.py",
            "--clip",
            "synthetic_size_clip",
            "--ball-track",
            str(ball_path),
            "--court-calibration",
            str(calibration_path),
            "--reviewed-bounces",
            str(reviewed_bounces_path),
            "--ball-sizes",
            str(size_path),
            "--out-dir",
            str(tmp_path / "out"),
            "--ball-type",
            "no_drag_test",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads((tmp_path / "out" / "synthetic_size_clip" / "ball_track_arc_solved.json").read_text(encoding="utf-8"))
    assert payload["summary"]["size_observation_count"] == len(size_frames)
    assert payload["validation"]["size_depth_residuals_m"]["enabled"] is True
    assert payload["validation"]["size_depth_residuals_m"]["with_size"]["count"] == len(size_frames)


def test_solve_ball_arcs_cli_measures_size_sidecar_from_video(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    video_path = tmp_path / "synthetic_ball.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (96, 72))
    for index in range(4):
        image = np.zeros((72, 96, 3), dtype=np.uint8)
        radius = 4 + index
        cv2.circle(image, (40 + index * 3, 36), radius, (0, 255, 255), -1)
        writer.write(image)
    writer.release()
    track = {
        "schema_version": 1,
        "fps": 30.0,
        "frames": [
            {"t": index / 30.0, "xy": [40 + index * 3, 36], "conf": 0.9, "visible": True}
            for index in range(4)
        ],
    }
    track_path = _write_json(tmp_path / "ball_track.json", track)
    out_path = tmp_path / "ball_size_observations.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/solve_ball_arcs.py",
            "--measure-size-video",
            str(video_path),
            "--ball-track",
            str(track_path),
            "--ball-sizes-out",
            str(out_path),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "racketsport_ball_size_observations"
    assert payload["summary"]["measured_count"] >= 3
    diameters = [frame["diameter_px"] for frame in payload["frames"]]
    assert min(diameters) > 0.0
    assert max(diameters) > min(diameters)
    assert all(0.0 <= frame["confidence"] <= 1.0 for frame in payload["frames"])


def test_clip_dir_default_reviewed_bounces_uses_inferred_clip_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clip = "wolverine_mixed_0200_mid_steep_corner"
    run_dir = tmp_path / "manager_rebuild_wolverine_20260702T23Z"
    run_dir.mkdir()
    _write_json(run_dir / "ball_track.json", {"frames": [], "fps": 30.0})
    _write_json(run_dir / "court_calibration.json", _projection_calibration())
    _write_json(run_dir / "pipeline_run.json", {"clip": clip})
    review_path = (
        tmp_path
        / "runs"
        / "ball_bounce_inout_review_packets_ground_contact_only_20260701T200001Z"
        / clip
        / "reviewed_ball_bounces.json"
    )
    review_path.parent.mkdir(parents=True)
    _write_json(review_path, {"bounces": [{"frame": 12, "t": 0.4}]})
    monkeypatch.chdir(tmp_path)

    tasks = solve_ball_arcs_cli._tasks(
        SimpleNamespace(
            ball_track=None,
            court_calibration=None,
            clip_dir=[run_dir],
            prototype_root=None,
            clips=None,
        )
    )

    assert tasks[0]["clip"] == clip
    assert Path(tasks[0]["reviewed_bounces"]).resolve() == review_path


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _anchor(
    anchor_id: str,
    kind: str,
    t: float,
    world_xyz: tuple[float, float, float],
    *,
    sigma_m: float = 0.05,
    status: str = "contact_prior",
    player_id: int | None = None,
) -> AnchorEvent:
    return AnchorEvent(
        anchor_id=anchor_id,
        kind=kind,
        t=t,
        frame=int(round(t * 60.0)),
        world_xyz=world_xyz,
        sigma_m=sigma_m,
        status=status,
        player_id=player_id,
        immovable=status == "human_reviewed",
    )


def _projection_calibration(*, with_gsd: bool = False) -> dict:
    calibration = {
        "schema_version": 1,
        "sport": "pickleball",
        "homography": [[1000.0 / 12.0, 0.0, 960.0], [0.0, 1000.0 / 12.0, 540.0], [0.0, 0.0, 1.0]],
        "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0, "dist": [], "source": "test"},
        "extrinsics": {
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "t": [0.0, 0.0, 12.0],
            "camera_height_m": 12.0,
        },
        "reprojection_error_px": {"median": 0.0, "p95": 0.0},
        "capture_quality": {"grade": "good", "reasons": []},
        "image_pts": [[876.6666667, -18.8], [1043.3333333, -18.8], [1043.3333333, 1098.8], [876.6666667, 1098.8]],
        "world_pts": [[-1.0, -6.7056, 0.0], [1.0, -6.7056, 0.0], [1.0, 6.7056, 0.0], [-1.0, 6.7056, 0.0]],
        "image_size": [1920, 1080],
    }
    if with_gsd:
        calibration["gsd_model"] = {
            "type": "analytic_ray_plane",
            "plane_sigma_m": 0.02,
            "calibration_sigma_m": 0.08,
            "samples": [{"court_xy": [0.0, 0.0], "gsd_m_per_px": 0.03, "sigma_p_m": 0.09}],
        }
    return calibration


def _project(calibration: dict, world_xyz: tuple[float, float, float]) -> tuple[float, float]:
    intrinsics = calibration["intrinsics"]
    translation = calibration["extrinsics"]["t"]
    camera_x = world_xyz[0] + translation[0]
    camera_y = world_xyz[1] + translation[1]
    camera_z = world_xyz[2] + translation[2]
    return (
        intrinsics["fx"] * camera_x / camera_z + intrinsics["cx"],
        intrinsics["fy"] * camera_y / camera_z + intrinsics["cy"],
    )


def _vz_for_endpoint(z0: float, z1: float, dt: float) -> float:
    return (z1 - z0 + 0.5 * GRAVITY * dt * dt) / dt


def _no_drag_position(
    p0: tuple[float, float, float],
    v0: tuple[float, float, float],
    dt: float,
) -> tuple[float, float, float]:
    return (
        p0[0] + v0[0] * dt,
        p0[1] + v0[1] * dt,
        p0[2] + v0[2] * dt - 0.5 * GRAVITY * dt * dt,
    )


def _observations_from_arc(
    calibration: dict,
    p0: tuple[float, float, float],
    v0: tuple[float, float, float],
    *,
    t0: float,
    t1: float,
    fps: float,
    include_size: bool = False,
) -> list[BallObservation]:
    observations: list[BallObservation] = []
    start = int(round(t0 * fps))
    end = int(round(t1 * fps))
    for frame in range(start, end + 1):
        t = frame / fps
        world = _no_drag_position(p0, v0, t - t0)
        diameter = _apparent_diameter_px(calibration, world) if include_size else None
        observations.append(
            BallObservation(
                frame=frame,
                t=t,
                xy=_project(calibration, world),
                confidence=0.95,
                visible=True,
                diameter_px=diameter,
                size_confidence=0.95 if diameter is not None else None,
            )
        )
    return observations


def _piecewise_track_frames(
    *,
    calibration: dict,
    p0: tuple[float, float, float],
    v0: tuple[float, float, float],
    bounce_t: float,
    v1: tuple[float, float, float],
    t_end: float,
    fps: float,
) -> list[dict]:
    frames: list[dict] = []
    bounce_xyz = _no_drag_position(p0, v0, bounce_t)
    for frame in range(int(round(t_end * fps)) + 1):
        t = frame / fps
        if t <= bounce_t:
            world = _no_drag_position(p0, v0, t)
        else:
            world = _no_drag_position(bounce_xyz, v1, t - bounce_t)
        frames.append(
            {
                "t": t,
                "xy": list(_project(calibration, world)),
                "conf": 0.95,
                "visible": True,
                "world_xyz": None,
                "approx": False,
            }
        )
    return frames


def _contact_event(t: float, player_id: int) -> dict:
    return {
        "type": "contact",
        "t": t,
        "frame": int(round(t * 60.0)),
        "player_id": player_id,
        "confidence": 1.0,
        "sources": {"human_review": 1.0, "wrist_vel": 0.0, "ball_inflection": 0.0},
        "window": {"t0": max(0.0, t - 0.03), "t1": t + 0.03, "importance": 1.0},
    }


def _apparent_diameter_px(calibration: dict, world_xyz: tuple[float, float, float]) -> float:
    intrinsics = calibration["intrinsics"]
    translation = calibration["extrinsics"]["t"]
    camera_z = world_xyz[2] + translation[2]
    return ((intrinsics["fx"] + intrinsics["fy"]) * 0.5) * (2.0 * BALL_RADIUS_M) / camera_z


def _skeleton_payload(samples: list[tuple[float, tuple[float, float, float]]], *, player_id: int) -> dict:
    frames = []
    for t, wrist in samples:
        elbow = (wrist[0] - 0.2, wrist[1], wrist[2])
        frames.append(
            {
                "t": t,
                "frame_idx": int(round(t * 60.0)),
                "joints_world": [
                    [elbow[0], elbow[1], elbow[2]],
                    [wrist[0], wrist[1], wrist[2]],
                    [elbow[0], elbow[1] + 0.3, elbow[2]],
                    [wrist[0], wrist[1] + 0.3, wrist[2]],
                ],
                "joint_conf": [0.95, 0.95, 0.95, 0.95],
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_skeleton3d",
        "fps": 60.0,
        "joint_names": ["right_elbow", "right_wrist", "left_elbow", "left_wrist"],
        "players": [{"id": player_id, "frames": frames}],
        "preview_only": True,
        "source_model": "synthetic",
        "world_frame": "court_netcenter_z_up_m",
        "provenance": {},
    }
