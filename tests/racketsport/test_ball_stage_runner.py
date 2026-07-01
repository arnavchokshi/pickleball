from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.racketsport.calibration_fixtures import minimal_calibration_image_pts, minimal_calibration_world_pts
from threed.racketsport.ball_stage_runner import BallStageRunner
from threed.racketsport.orchestrator import StageRun, run_pipeline
from threed.racketsport.body_compute import build_body_compute_execution
from threed.racketsport.frame_rating import build_frame_compute_plan
from threed.racketsport.schemas import BallTrack, ContactWindows, validate_artifact_file


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ball_track_payload() -> dict:
    return {
        "schema_version": 1,
        "fps": 30.0,
        "source": "tracknet",
        "frames": [
            {"t": 0.0, "xy": [120.0, 240.0], "conf": 0.82, "visible": True, "approx": False},
            {"t": 1.0 / 30.0, "xy": [124.0, 242.0], "conf": 0.76, "visible": True, "approx": False},
            {"t": 2.0 / 30.0, "xy": [0.0, 0.0], "conf": 0.0, "visible": False, "approx": False},
        ],
        "bounces": [],
    }


def _ball_track_payload_with_world_bounce() -> dict:
    payload = _ball_track_payload()
    payload["frames"] = [
        {
            "t": 0.0,
            "xy": [120.0, 240.0],
            "conf": 0.88,
            "visible": True,
            "world_xyz": [0.0, 0.0, 1.2],
            "approx": False,
        },
        {
            "t": 1.0 / 30.0,
            "xy": [124.0, 242.0],
            "conf": 0.86,
            "visible": True,
            "world_xyz": [0.2, 0.0, 0.48],
            "approx": False,
        },
        {
            "t": 2.0 / 30.0,
            "xy": [128.0, 244.0],
            "conf": 0.85,
            "visible": True,
            "world_xyz": [0.4, 0.0, 0.02],
            "approx": False,
        },
        {
            "t": 3.0 / 30.0,
            "xy": [132.0, 246.0],
            "conf": 0.84,
            "visible": True,
            "world_xyz": [0.6, 0.0, 0.50],
            "approx": False,
        },
        {
            "t": 4.0 / 30.0,
            "xy": [136.0, 248.0],
            "conf": 0.82,
            "visible": True,
            "world_xyz": [0.8, 0.0, 1.15],
            "approx": False,
        },
    ]
    payload["bounces"] = []
    return payload


def _ball_track_payload_with_image_bounce() -> dict:
    calibration = _ballistic_projection_calibration_payload()
    times = [0.00, 0.05, 0.10, 0.15, 0.20]
    return {
        "schema_version": 1,
        "fps": 20.0,
        "source": "tracknet",
        "frames": [
            {
                "t": t,
                "xy": _project_with_ballistic_calibration(calibration, _synthetic_bounce_world_xyz(t)),
                "conf": 0.9,
                "visible": True,
                "approx": False,
            }
            for t in times
        ],
        "bounces": [],
    }


def _ball_track_payload_with_image_contact_turn() -> dict:
    payload = _ball_track_payload()
    payload["frames"] = [
        {"t": 0.0, "xy": [100.0, 100.0], "conf": 0.92, "visible": True, "approx": False},
        {"t": 1.0 / 30.0, "xy": [112.0, 100.0], "conf": 0.91, "visible": True, "approx": False},
        {"t": 2.0 / 30.0, "xy": [112.0, 112.0], "conf": 0.9, "visible": True, "approx": False},
    ]
    return payload


def _ballistic_projection_calibration_payload() -> dict:
    return {
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


def _synthetic_bounce_world_xyz(t: float) -> tuple[float, float, float]:
    bounce_t = 0.10
    dt = t - bounce_t
    x = 1.20 + 4.0 * dt
    y = 2.04 + 0.4 * dt
    if dt <= 0.0:
        z = 0.04 + 3.0 * (-dt) - 0.5 * 9.81 * dt * dt
    else:
        z = 0.04 + 2.6 * dt - 0.5 * 9.81 * dt * dt
    return x, y, z


def _project_with_ballistic_calibration(calibration: dict, world_xyz: tuple[float, float, float]) -> list[float]:
    intrinsics = calibration["intrinsics"]
    translation = calibration["extrinsics"]["t"]
    camera_x = world_xyz[0] + translation[0]
    camera_y = world_xyz[1] + translation[1]
    camera_z = world_xyz[2] + translation[2]
    return [
        intrinsics["fx"] * camera_x / camera_z + intrinsics["cx"],
        intrinsics["fy"] * camera_y / camera_z + intrinsics["cy"],
    ]


def _write_no_click_ball_source(inputs_dir: Path) -> Path:
    source = inputs_dir / "tracknet_smoke_0000_0010" / "ball_track_fusion_temporal_vball100_localtraj.json"
    _write_json(source, _ball_track_payload())
    return source


def _write_contact_cue_artifacts(inputs_dir: Path) -> None:
    _write_json(inputs_dir / "audio_onsets.json", {"schema_version": 1, "onsets": [{"time_s": 1.0 / 30.0, "score": 0.9}]})
    _write_json(
        inputs_dir / "wrist_velocity_peaks.json",
        {
            "schema_version": 1,
            "peaks": [
                {
                    "time_s": 1.0 / 30.0,
                    "player_id": 7,
                    "wrist_world_xyz": [0.0, 0.0, 1.0],
                    "speed_mps": 12.0,
                    "confidence": 0.8,
                }
            ],
        },
    )
    _write_json(
        inputs_dir / "ball_inflections.json",
        {
            "schema_version": 1,
            "candidates": [
                {
                    "time_s": 1.0 / 30.0,
                    "ball_world_xyz": [0.02, 0.0, 1.0],
                    "confidence": 0.7,
                }
            ],
        },
    )


def _write_wrist_velocity_peak_artifact(inputs_dir: Path) -> None:
    _write_json(
        inputs_dir / "wrist_velocity_peaks.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_wrist_velocity_peaks",
            "peaks": [
                {
                    "time_s": 1.0 / 30.0,
                    "player_id": 7,
                    "wrist_world_xyz": [0.0, 0.0, 0.0],
                    "speed_mps": 12.0,
                    "confidence": 0.8,
                }
            ],
        },
    )


def _write_model_ball_track(path: Path, *, source: str = "totnet") -> None:
    payload = _ball_track_payload()
    payload["source"] = source
    payload["frames"][0]["xy"] = [444.0, 222.0]
    _write_json(path, payload)


def _physics_refinement_payload() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_physics_refinement",
        "physics": "cpu_fallback_scaffold",
        "foot2_done": False,
        "must_not_mark_done_verified": True,
        "constraint_summary": {
            "contact_frames": 1,
            "max_contact_slide_m": 0.0,
            "max_floor_penetration_m": 0.0,
            "inter_player_penetration_frames": 0,
            "max_inter_player_penetration_m": 0.0,
        },
        "execution_plan": {
            "mode": "cpu_fallback",
            "will_run_mjx": False,
            "reason": "test fixture",
        },
    }


def _write_dependency_artifacts(run_dir: Path) -> None:
    _write_json(
        run_dir / "court_calibration.json",
        {
            "schema_version": 1,
            "sport": "pickleball",
            "homography": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0, "dist": [], "source": "manual"},
            "extrinsics": {
                "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "t": [0.0, 0.0, 12.0],
                "camera_height_m": 12.0,
            },
            "reprojection_error_px": {"median": 0.0, "p95": 0.0},
            "capture_quality": {"grade": "good", "reasons": []},
            "image_pts": minimal_calibration_image_pts(),
            "world_pts": minimal_calibration_world_pts(),
        },
    )
    _write_json(run_dir / "court_zones.json", {"schema_version": 1, "zones": {}})
    _write_json(
        run_dir / "net_plane.json",
        {
            "schema_version": 1,
            "plane": {"point": [0.0, 0.0, 0.0], "normal": [0.0, 1.0, 0.0]},
            "endpoints": [[-3.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            "center_height_in": 34.0,
            "post_height_in": 36.0,
        },
    )
    _write_json(
        run_dir / "court_line_evidence.json",
        {
            "schema_version": 1,
            "sport": "pickleball",
            "source": "test",
            "line_observations": [],
            "keypoint_observations": [],
            "net_observations": [],
            "aggregate": {
                "accepted_line_ids": [],
                "rejected_line_ids": [],
                "missing_required_line_ids": [],
                "missing_required_net_ids": [],
                "mean_residual_px": 0.0,
                "p95_residual_px": 0.0,
                "temporal_stability_px": 0.0,
                "auto_calibration_ready": False,
                "reasons": ["test_dependency_artifact"],
            },
        },
    )
    _write_json(
        run_dir / "tracks.json",
        {
            "schema_version": 1,
            "fps": 30.0,
            "players": [
                {
                    "id": 7,
                    "side": "near",
                    "role": "near_left",
                    "frames": [
                        {"t": 0.0, "bbox": [1.0, 2.0, 3.0, 4.0], "world_xy": [0.0, 0.0], "conf": 0.9},
                        {
                            "t": 1.0 / 30.0,
                            "bbox": [1.2, 2.0, 3.2, 4.0],
                            "world_xy": [0.05, 0.0],
                            "conf": 0.88,
                        },
                    ],
                }
            ],
            "rally_spans": [],
        },
    )
    _write_json(
        run_dir / "smpl_motion.json",
        {
            "schema_version": 1,
            "model": "smplx",
            "fps": 30.0,
            "world_frame": "court_Z0",
            "players": [
                {
                    "id": 7,
                    "betas": [0.0] * 10,
                    "frames": [
                        {
                            "t": 0.0,
                            "global_orient": [0.0, 0.0, 0.0],
                            "body_pose": [0.0] * 63,
                            "left_hand_pose": [],
                            "right_hand_pose": [],
                            "transl_world": [0.0, 0.0, 0.0],
                            "joints_world": [[0.0, 0.0, 0.0]],
                            "joint_conf": [0.9],
                            "foot_contact": {"left": True, "right": True},
                            "grf": None,
                        }
                    ],
                    "skate_free": False,
                    "physics": "test",
                }
            ],
        },
    )
    _write_json(
        run_dir / "skeleton3d.json",
        {
            "schema_version": 1,
            "joint_names": ["root"],
            "preview_only": True,
            "players": [{"id": 7, "frames": [{"t": 0.0, "joints_world": [[0.0, 0.0, 0.0]], "joint_conf": [0.9]}]}],
        },
    )
    _write_json(
        run_dir / "body_compute_execution.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_compute_execution",
            "mode": "adaptive_frame_compute_plan",
            "scheduled_frames": [
                {
                    "frame_idx": 0,
                    "player_targets": [
                        {
                            "player_id": 7,
                            "track_conf": 0.9,
                            "score": 0.8,
                            "recommended_tier": "deep_mesh",
                            "target_representation": "world_mesh",
                            "reasons": ["test_fixture"],
                        }
                    ],
                }
            ],
            "skipped_frames": [],
            "summary": {
                "scheduled_frame_count": 1,
                "scheduled_player_frame_count": 1,
                "scheduled_by_target_representation": {"world_mesh": 1},
                "skipped_frame_count": 0,
            },
        },
    )
    _write_json(
        run_dir / "body_mesh_readiness.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_mesh_readiness",
            "clip": "clip_001",
            "status": "verified",
            "world_mesh_available": True,
            "representation_decision": "world_mesh_required_available_verified",
            "trusted_for_body_promotion": True,
            "summary": {
                "player_count": 1,
                "mesh_player_count": 1,
                "mesh_frame_count": 1,
                "mesh_vertex_count_min": 3,
                "mesh_vertex_count_max": 3,
                "joints_player_count": 1,
                "joints_frame_count": 1,
            },
            "blockers": [],
            "warnings": [],
        },
    )
    _write_json(run_dir / "physics_refinement.json", _physics_refinement_payload())


class NoopDependencyRunner:
    real_model = False
    source_mode = "prewritten_test_artifacts"

    def __init__(self, stage: str, produced_artifacts: tuple[str, ...]) -> None:
        self.stage = stage
        self.produced_artifacts = produced_artifacts

    def run(self, context) -> StageRun:
        return StageRun(
            stage=self.stage,
            status="ran",
            real_model=False,
            source_mode=self.source_mode,
            produced_artifacts=self.produced_artifacts,
            notes=("test runner reused prewritten dependency artifacts",),
        )


def _noop_dependency_runners() -> dict[str, NoopDependencyRunner]:
    return {
        "calibration": NoopDependencyRunner(
            "calibration",
            ("court_calibration.json", "court_zones.json", "net_plane.json", "court_line_evidence.json"),
        ),
        "tracking": NoopDependencyRunner("tracking", ("tracks.json",)),
        "body": NoopDependencyRunner(
            "body",
            ("smpl_motion.json", "skeleton3d.json", "body_compute_execution.json", "body_mesh_readiness.json"),
        ),
        "physics": NoopDependencyRunner("physics", ("smpl_motion.json", "physics_refinement.json")),
    }


def test_ball_stage_runner_uses_no_click_localtraj_artifact_without_reading_ball_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = tmp_path / "inputs" / "clip_001"
    run_dir = tmp_path / "runs" / "clip_001"
    source = _write_no_click_ball_source(inputs)
    _write_dependency_artifacts(run_dir)
    click_labels = inputs / "ball_points.json"
    click_labels.write_text("{ this would explode if read", encoding="utf-8")

    original_read_text = Path.read_text

    def guard_click_reads(path: Path, *args, **kwargs):
        if path.name == "ball_points.json":
            raise AssertionError("BALL StageRunner must not read ball_points.json")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guard_click_reads)

    summary = run_pipeline(
        clip="clip_001",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="ball_events",
        runners=_noop_dependency_runners(),
    )

    assert summary["status"] == "blocked"
    ball_stage = summary["stages"][-1]
    assert ball_stage["stage"] == "ball_events"
    assert ball_stage["status"] == "blocked"
    assert ball_stage["source_mode"] == "no_click_fusion_temporal_vball100_localtraj"
    assert ball_stage["metrics"]["source_ball_track"] == str(source)
    assert ball_stage["metrics"]["uses_human_clicks"] is False
    assert ball_stage["produced_artifacts"] == ["ball_track.json", "contact_windows.json"]

    emitted = json.loads((run_dir / "ball_track.json").read_text(encoding="utf-8"))
    assert emitted == _ball_track_payload()


def test_ball_stage_runner_real_totnet_mode_runs_model_inference_before_contact_fusion(tmp_path: Path) -> None:
    clip = "clip_001"
    inputs = tmp_path / "inputs" / clip
    run_dir = tmp_path / "runs" / clip
    video = inputs / "input.mp4"
    checkpoint = tmp_path / "totnet.ckpt"
    totnet_repo = tmp_path / "TOTNet"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"fake video bytes")
    checkpoint.write_bytes(b"fake checkpoint")
    (totnet_repo / "src").mkdir(parents=True)
    _write_dependency_artifacts(run_dir)
    _write_contact_cue_artifacts(inputs)

    calls: list[dict[str, object]] = []

    def fake_totnet_runner(**kwargs):
        calls.append(kwargs)
        _write_model_ball_track(Path(kwargs["out"]))
        _write_json(Path(kwargs["predictions_out"]), {"artifact_type": "fake_totnet_predictions"})
        if kwargs.get("metadata_out") is not None:
            _write_json(Path(kwargs["metadata_out"]), {"artifact_type": "fake_totnet_run"})
        return {
            "schema_version": 1,
            "artifact_type": "racketsport_totnet_ball_run",
            "frame_count": 3,
            "visible_frame_count": 2,
            "runtime": {"effective_fps": 42.0},
        }

    runners = _noop_dependency_runners()
    runners["ball_events"] = BallStageRunner(
        totnet_repo=totnet_repo,
        totnet_checkpoint=checkpoint,
        video_path=video,
        model_runner=fake_totnet_runner,
    )

    summary = run_pipeline(
        clip=clip,
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="ball_events",
        runners=runners,
    )

    assert calls
    assert calls[0]["video"] == video
    assert calls[0]["totnet_repo"] == totnet_repo
    assert calls[0]["checkpoint"] == checkpoint
    assert calls[0]["out"] == run_dir / "ball_track.json"
    assert calls[0]["predictions_out"] == run_dir / "totnet_predictions.json"
    ball_stage = summary["stages"][-1]
    assert ball_stage["stage"] == "ball_events"
    assert ball_stage["real_model"] is True
    assert ball_stage["source_mode"] == "totnet_inference"
    assert ball_stage["metrics"]["model_family"] == "TOTNet"
    assert ball_stage["metrics"]["source_ball_track"] == str(run_dir / "ball_track.json")
    assert ball_stage["metrics"]["contact_event_count"] == 1
    assert "ran TOTNet video inference locally" in ball_stage["notes"]
    emitted = json.loads((run_dir / "ball_track.json").read_text(encoding="utf-8"))
    assert emitted["source"] == "totnet"
    assert emitted["frames"][0]["xy"] == [444.0, 222.0]


def test_ball_stage_runner_real_tracknet_mode_runs_model_inference_before_contact_fusion(tmp_path: Path) -> None:
    clip = "clip_001"
    inputs = tmp_path / "inputs" / clip
    run_dir = tmp_path / "runs" / clip
    video = inputs / "input.mp4"
    tracknet_file = tmp_path / "TrackNet_best.pt"
    inpaintnet_file = tmp_path / "InpaintNet_best.pt"
    tracknet_repo = tmp_path / "TrackNetV3"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"fake video bytes")
    tracknet_file.write_bytes(b"fake tracknet checkpoint")
    inpaintnet_file.write_bytes(b"fake inpaintnet checkpoint")
    tracknet_repo.mkdir(parents=True)
    (tracknet_repo / "predict.py").write_text("print('fake')\n", encoding="utf-8")
    _write_dependency_artifacts(run_dir)
    _write_contact_cue_artifacts(inputs)

    calls: list[dict[str, object]] = []

    def fake_tracknet_runner(**kwargs):
        calls.append(kwargs)
        _write_model_ball_track(Path(kwargs["out"]), source="tracknet")
        return {
            "schema_version": 1,
            "artifact_type": "racketsport_tracknet_ball_run",
            "source_mode": "tracknet_predict",
            "frame_count": 3,
            "visible_frame_count": 2,
            "runtime": {"effective_fps": 24.0},
        }

    runners = _noop_dependency_runners()
    runners["ball_events"] = BallStageRunner(
        tracknet_repo=tracknet_repo,
        tracknet_file=tracknet_file,
        inpaintnet_file=inpaintnet_file,
        video_path=video,
        tracknet_runner=fake_tracknet_runner,
        tracknet_fps=30.0,
    )

    summary = run_pipeline(
        clip=clip,
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="ball_events",
        runners=runners,
    )

    assert calls
    assert calls[0]["video"] == video
    assert calls[0]["tracknet_repo"] == tracknet_repo
    assert calls[0]["tracknet_file"] == tracknet_file
    assert calls[0]["inpaintnet_file"] == inpaintnet_file
    assert calls[0]["out"] == run_dir / "ball_track.json"
    assert calls[0]["metadata_out"] == run_dir / "tracknet_metadata.json"
    ball_stage = summary["stages"][-1]
    assert ball_stage["stage"] == "ball_events"
    assert ball_stage["real_model"] is True
    assert ball_stage["source_mode"] == "tracknetv3_inference"
    assert ball_stage["metrics"]["model_family"] == "TrackNetV3"
    assert ball_stage["metrics"]["source_ball_track"] == str(run_dir / "ball_track.json")
    assert ball_stage["metrics"]["contact_event_count"] == 1
    assert "ran TrackNetV3 video inference locally" in ball_stage["notes"]
    emitted = json.loads((run_dir / "ball_track.json").read_text(encoding="utf-8"))
    assert emitted["source"] == "tracknet"
    assert emitted["frames"][0]["xy"] == [444.0, 222.0]


def test_ball_stage_runner_derives_ball_inflections_from_current_tracknet_output(tmp_path: Path) -> None:
    clip = "clip_001"
    inputs = tmp_path / "inputs" / clip
    run_dir = tmp_path / "runs" / clip
    video = inputs / "input.mp4"
    tracknet_file = tmp_path / "TrackNet_best.pt"
    inpaintnet_file = tmp_path / "InpaintNet_best.pt"
    tracknet_repo = tmp_path / "TrackNetV3"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"fake video bytes")
    tracknet_file.write_bytes(b"fake tracknet checkpoint")
    inpaintnet_file.write_bytes(b"fake inpaintnet checkpoint")
    tracknet_repo.mkdir(parents=True)
    (tracknet_repo / "predict.py").write_text("print('fake')\n", encoding="utf-8")
    _write_dependency_artifacts(run_dir)
    _write_wrist_velocity_peak_artifact(inputs)

    def fake_tracknet_runner(**kwargs):
        _write_json(Path(kwargs["out"]), _ball_track_payload_with_image_contact_turn())
        return {
            "schema_version": 1,
            "artifact_type": "racketsport_tracknet_ball_run",
            "source_mode": "tracknet_predict",
            "frame_count": 3,
            "visible_frame_count": 3,
        }

    runners = _noop_dependency_runners()
    runners["ball_events"] = BallStageRunner(
        tracknet_repo=tracknet_repo,
        tracknet_file=tracknet_file,
        inpaintnet_file=inpaintnet_file,
        video_path=video,
        tracknet_runner=fake_tracknet_runner,
        tracknet_fps=30.0,
    )

    summary = run_pipeline(
        clip=clip,
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="ball_events",
        runners=runners,
    )

    ball_inflections = json.loads((run_dir / "ball_inflections.json").read_text(encoding="utf-8"))
    assert ball_inflections["source"] == "ball_track_image_motion"
    assert ball_inflections["summary"]["candidate_count"] == 1
    ball_stage = summary["stages"][-1]
    assert ball_stage["status"] == "ran"
    assert ball_stage["metrics"]["contact_event_count"] == 1
    assert ball_stage["metrics"]["ball_inflection_candidate_count"] == 1
    assert "ball_inflections.json" in ball_stage["produced_artifacts"]
    assert "derived ball_inflections.json from current ball_track.json image motion" in ball_stage["notes"]


def test_ball_stage_runner_tracknet_local_search_keeps_raw_and_outputs_filtered_track(tmp_path: Path) -> None:
    clip = "clip_001"
    inputs = tmp_path / "inputs" / clip
    run_dir = tmp_path / "runs" / clip
    video = inputs / "input.mp4"
    tracknet_file = tmp_path / "TrackNet_best.pt"
    inpaintnet_file = tmp_path / "InpaintNet_best.pt"
    tracknet_repo = tmp_path / "TrackNetV3"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"fake video bytes")
    tracknet_file.write_bytes(b"fake tracknet checkpoint")
    inpaintnet_file.write_bytes(b"fake inpaintnet checkpoint")
    tracknet_repo.mkdir(parents=True)
    (tracknet_repo / "predict.py").write_text("print('fake')\n", encoding="utf-8")
    _write_dependency_artifacts(run_dir)
    _write_contact_cue_artifacts(inputs)

    tracknet_calls: list[dict[str, object]] = []
    local_search_calls: list[dict[str, object]] = []

    def fake_tracknet_runner(**kwargs):
        tracknet_calls.append(kwargs)
        _write_model_ball_track(Path(kwargs["out"]), source="tracknet")
        return {
            "schema_version": 1,
            "artifact_type": "racketsport_tracknet_ball_run",
            "source_mode": "tracknet_predict",
            "frame_count": 3,
            "visible_frame_count": 2,
        }

    def fake_local_search_runner(**kwargs):
        local_search_calls.append(kwargs)
        payload = _ball_track_payload()
        payload["source"] = "tracknet"
        payload["frames"][0]["xy"] = [555.0, 333.0]
        payload["frames"][2]["xy"] = [560.0, 336.0]
        payload["frames"][2]["conf"] = 0.61
        payload["frames"][2]["visible"] = True
        _write_json(Path(kwargs["out_path"]), payload)
        summary = {
            "schema_version": 1,
            "artifact_type": "racketsport_ball_local_search_filter",
            "visible_before": 2,
            "visible_after": 3,
            "recovered_count": 1,
            "relocated_off_path_count": 1,
            "suppressed_off_path_count": 0,
            "uses_human_clicks": False,
        }
        _write_json(Path(kwargs["summary_path"]), summary)
        return summary

    runners = _noop_dependency_runners()
    runners["ball_events"] = BallStageRunner(
        tracknet_repo=tracknet_repo,
        tracknet_file=tracknet_file,
        inpaintnet_file=inpaintnet_file,
        video_path=video,
        tracknet_runner=fake_tracknet_runner,
        tracknet_fps=30.0,
        tracknet_local_search=True,
        local_search_runner=fake_local_search_runner,
    )

    summary = run_pipeline(
        clip=clip,
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="ball_events",
        runners=runners,
    )

    assert tracknet_calls
    assert local_search_calls
    assert local_search_calls[0]["video_path"] == video
    assert local_search_calls[0]["ball_track_path"] == run_dir / "ball_track_tracknet_raw.json"
    assert local_search_calls[0]["out_path"] == run_dir / "ball_track.json"
    assert local_search_calls[0]["summary_path"] == run_dir / "ball_local_search_summary.json"
    assert json.loads((run_dir / "ball_track_tracknet_raw.json").read_text(encoding="utf-8"))["frames"][0]["xy"] == [
        444.0,
        222.0,
    ]
    assert json.loads((run_dir / "ball_track.json").read_text(encoding="utf-8"))["frames"][0]["xy"] == [555.0, 333.0]
    ball_stage = summary["stages"][-1]
    assert ball_stage["metrics"]["source_ball_track"] == str(run_dir / "ball_track.json")
    assert ball_stage["metrics"]["visible_frame_count"] == 3
    assert ball_stage["metrics"]["tracknet_raw_visible_frame_count"] == 2
    assert ball_stage["metrics"]["raw_tracknet_ball_track"] == str(run_dir / "ball_track_tracknet_raw.json")
    assert ball_stage["metrics"]["local_search"]["recovered_count"] == 1
    assert "ball_track_tracknet_raw.json" in ball_stage["produced_artifacts"]
    assert "ball_local_search_summary.json" in ball_stage["produced_artifacts"]
    assert "applied TrackNetV3 local-search postprocess" in ball_stage["notes"]


def test_ball_stage_runner_applies_physics3d_bounces_from_world_xyz(tmp_path: Path) -> None:
    clip = "clip_001"
    inputs = tmp_path / "inputs" / clip
    run_dir = tmp_path / "runs" / clip
    video = inputs / "input.mp4"
    tracknet_file = tmp_path / "TrackNet_best.pt"
    inpaintnet_file = tmp_path / "InpaintNet_best.pt"
    tracknet_repo = tmp_path / "TrackNetV3"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"fake video bytes")
    tracknet_file.write_bytes(b"fake tracknet checkpoint")
    inpaintnet_file.write_bytes(b"fake inpaintnet checkpoint")
    tracknet_repo.mkdir(parents=True)
    (tracknet_repo / "predict.py").write_text("print('fake')\n", encoding="utf-8")
    _write_dependency_artifacts(run_dir)
    _write_contact_cue_artifacts(inputs)

    def fake_tracknet_runner(**kwargs):
        _write_json(Path(kwargs["out"]), _ball_track_payload_with_world_bounce())
        return {
            "schema_version": 1,
            "artifact_type": "racketsport_tracknet_ball_run",
            "source_mode": "tracknet_predict",
            "frame_count": 5,
            "visible_frame_count": 5,
        }

    runners = _noop_dependency_runners()
    runners["ball_events"] = BallStageRunner(
        tracknet_repo=tracknet_repo,
        tracknet_file=tracknet_file,
        inpaintnet_file=inpaintnet_file,
        video_path=video,
        tracknet_runner=fake_tracknet_runner,
        tracknet_fps=30.0,
        ball_physics3d=True,
    )

    summary = run_pipeline(
        clip=clip,
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="ball_events",
        runners=runners,
    )

    emitted = json.loads((run_dir / "ball_track.json").read_text(encoding="utf-8"))
    assert emitted["bounces"] == [{"t": 2.0 / 30.0, "world_xy": [0.4, 0.0]}]
    physics_summary = json.loads((run_dir / "ball_physics3d_summary.json").read_text(encoding="utf-8"))
    assert physics_summary["sample_count"] == 5
    assert physics_summary["bounce_count"] == 1
    assert physics_summary["uses_human_clicks"] is False
    ball_stage = summary["stages"][-1]
    assert ball_stage["metrics"]["bounce_count"] == 1
    assert ball_stage["metrics"]["physics3d"]["bounce_count"] == 1
    assert "ball_physics3d_summary.json" in ball_stage["produced_artifacts"]
    assert "applied BALL 3D bounce physics from existing world_xyz samples" in ball_stage["notes"]


def test_ball_stage_runner_reconstructs_physics3d_bounces_from_image_track_and_calibration(tmp_path: Path) -> None:
    clip = "clip_001"
    inputs = tmp_path / "inputs" / clip
    run_dir = tmp_path / "runs" / clip
    video = inputs / "input.mp4"
    tracknet_file = tmp_path / "TrackNet_best.pt"
    inpaintnet_file = tmp_path / "InpaintNet_best.pt"
    tracknet_repo = tmp_path / "TrackNetV3"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"fake video bytes")
    tracknet_file.write_bytes(b"fake tracknet checkpoint")
    inpaintnet_file.write_bytes(b"fake inpaintnet checkpoint")
    tracknet_repo.mkdir(parents=True)
    (tracknet_repo / "predict.py").write_text("print('fake')\n", encoding="utf-8")
    _write_dependency_artifacts(run_dir)
    _write_json(run_dir / "court_calibration.json", _ballistic_projection_calibration_payload())
    _write_contact_cue_artifacts(inputs)

    def fake_tracknet_runner(**kwargs):
        _write_json(Path(kwargs["out"]), _ball_track_payload_with_image_bounce())
        return {
            "schema_version": 1,
            "artifact_type": "racketsport_tracknet_ball_run",
            "source_mode": "tracknet_predict",
            "frame_count": 5,
            "visible_frame_count": 5,
        }

    runners = _noop_dependency_runners()
    runners["ball_events"] = BallStageRunner(
        tracknet_repo=tracknet_repo,
        tracknet_file=tracknet_file,
        inpaintnet_file=inpaintnet_file,
        video_path=video,
        tracknet_runner=fake_tracknet_runner,
        tracknet_fps=20.0,
        ball_physics3d=True,
    )

    summary = run_pipeline(
        clip=clip,
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="ball_events",
        runners=runners,
    )

    emitted = json.loads((run_dir / "ball_track.json").read_text(encoding="utf-8"))
    assert emitted["frames"][2]["world_xyz"][2] == pytest.approx(0.04, abs=0.05)
    assert emitted["bounces"] == [{"t": pytest.approx(0.10), "world_xy": pytest.approx([1.20, 2.04])}]
    physics_summary = json.loads((run_dir / "ball_physics3d_summary.json").read_text(encoding="utf-8"))
    assert physics_summary["status"] == "ran"
    assert physics_summary["sample_count"] == 5
    assert physics_summary["bounce_count"] == 1
    assert physics_summary["image_reconstruction"]["status"] == "ran"
    ball_stage = summary["stages"][-1]
    assert ball_stage["metrics"]["physics3d"]["bounce_count"] == 1
    assert "reconstructed BALL 3D bounce physics from image track and court calibration" in ball_stage["notes"]


def test_ball_stage_runner_fails_closed_on_renamed_tap_track_even_with_cues(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs" / "clip_001"
    run_dir = tmp_path / "runs" / "clip_001"
    renamed_tap_source = inputs / "renamed_no_click_candidate.json"
    tap_payload = _ball_track_payload()
    tap_payload["source"] = "tap"
    _write_json(renamed_tap_source, tap_payload)
    _write_contact_cue_artifacts(inputs)
    _write_dependency_artifacts(run_dir)

    runners = _noop_dependency_runners()
    runners["ball_events"] = BallStageRunner(source_path=renamed_tap_source)

    summary = run_pipeline(
        clip="clip_001",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="ball_events",
        runners=runners,
    )

    assert summary["status"] == "fail"
    ball_stage = summary["stages"][-1]
    assert ball_stage["status"] == "fail"
    assert any("refuses to consume tap/manual ball tracks" in note for note in ball_stage["notes"])
    assert not (run_dir / "ball_track.json").exists()
    assert not (run_dir / "contact_windows.json").exists()


def test_ball_stage_runner_prefers_eval_suite_selected_track_over_legacy_prototype(tmp_path: Path) -> None:
    clip = "clip_001"
    inputs = tmp_path / "inputs" / clip
    run_dir = tmp_path / "runs" / clip
    eval_root = tmp_path / "eval_suite"
    selected_source = eval_root / "selected_tracks" / clip / "ball_track.json"
    legacy_source = eval_root / clip / "tracknet_smoke_0000_0010" / "ball_track_fusion_temporal_vball100_localtraj.json"
    selected_payload = _ball_track_payload()
    selected_payload["frames"][0]["xy"] = [321.0, 123.0]
    selected_payload["source"] = "pbmat"
    legacy_payload = _ball_track_payload()
    legacy_payload["frames"][0]["xy"] = [999.0, 999.0]
    _write_json(selected_source, selected_payload)
    _write_json(
        selected_source.parent / "ball_track_selection.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_ball_track_selection",
            "status": "selected_not_gate_verified",
            "clip": clip,
            "candidate": "pbmat_v0_motion_composite",
            "candidate_category": "composite_alias_not_trained_model",
            "candidate_score": 0.12,
            "candidate_rank": 2,
            "eligible_for_model_ranking": False,
            "trained_pbmat_checkpoint": False,
            "source_ball_track": str(legacy_source),
            "out": str(selected_source),
            "not_ground_truth": True,
        },
    )
    _write_json(legacy_source, legacy_payload)
    _write_dependency_artifacts(run_dir)

    runners = _noop_dependency_runners()
    runners["ball_events"] = BallStageRunner(prototype_root=eval_root, allow_prototype_root_fallback=True)

    summary = run_pipeline(
        clip=clip,
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="ball_events",
        runners=runners,
    )

    assert summary["status"] == "blocked"
    ball_stage = summary["stages"][-1]
    assert ball_stage["status"] == "blocked"
    assert ball_stage["metrics"]["source_ball_track"] == str(selected_source)
    assert ball_stage["source_mode"] == "selected_ball_track_prototype"
    assert ball_stage["metrics"]["selection"]["candidate"] == "pbmat_v0_motion_composite"
    assert ball_stage["metrics"]["selection"]["candidate_category"] == "composite_alias_not_trained_model"
    assert ball_stage["metrics"]["selection"]["trained_pbmat_checkpoint"] is False
    emitted = json.loads((run_dir / "ball_track.json").read_text(encoding="utf-8"))
    assert emitted["frames"][0]["xy"] == [321.0, 123.0]


def test_ball_stage_runner_prefers_current_inputs_over_stale_run_dir_track(tmp_path: Path) -> None:
    clip = "clip_001"
    inputs = tmp_path / "inputs" / clip
    run_dir = tmp_path / "runs" / clip
    current_source = inputs / "tracknet_smoke_0000_0010" / "ball_track_fusion_temporal_vball100_localtraj.json"
    stale_source = run_dir / "tracknet_smoke_0000_0010" / "ball_track_fusion_temporal_vball100_localtraj.json"
    current_payload = _ball_track_payload()
    current_payload["frames"][0]["xy"] = [111.0, 222.0]
    stale_payload = _ball_track_payload()
    stale_payload["frames"][0]["xy"] = [999.0, 999.0]
    _write_json(current_source, current_payload)
    _write_json(stale_source, stale_payload)
    _write_dependency_artifacts(run_dir)

    runners = _noop_dependency_runners()

    summary = run_pipeline(
        clip=clip,
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="ball_events",
        runners=runners,
    )

    ball_stage = summary["stages"][-1]
    assert ball_stage["metrics"]["source_ball_track"] == str(current_source)
    emitted = json.loads((run_dir / "ball_track.json").read_text(encoding="utf-8"))
    assert emitted["frames"][0]["xy"] == [111.0, 222.0]


def test_ball_stage_runner_fails_closed_when_selected_track_has_no_selection_sidecar(tmp_path: Path) -> None:
    clip = "clip_001"
    inputs = tmp_path / "inputs" / clip
    run_dir = tmp_path / "runs" / clip
    eval_root = tmp_path / "eval_suite"
    selected_source = eval_root / "selected_tracks" / clip / "ball_track.json"
    legacy_source = eval_root / clip / "tracknet_smoke_0000_0010" / "ball_track_fusion_temporal_vball100_localtraj.json"
    _write_json(selected_source, _ball_track_payload())
    _write_json(legacy_source, _ball_track_payload())
    _write_dependency_artifacts(run_dir)

    runners = _noop_dependency_runners()
    runners["ball_events"] = BallStageRunner(prototype_root=eval_root, allow_prototype_root_fallback=True)

    summary = run_pipeline(
        clip=clip,
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="ball_events",
        runners=runners,
    )

    assert summary["status"] == "fail"
    ball_stage = summary["stages"][-1]
    assert any("missing selected-track metadata sidecar" in note for note in ball_stage["notes"])
    assert not (run_dir / "ball_track.json").exists()


def test_ball_stage_runner_prefers_current_input_cues_over_stale_run_dir_cues(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs" / "clip_001"
    run_dir = tmp_path / "runs" / "clip_001"
    _write_no_click_ball_source(inputs)
    _write_dependency_artifacts(run_dir)
    _write_contact_cue_artifacts(inputs)
    _write_json(run_dir / "audio_onsets.json", {"schema_version": 1, "onsets": []})
    _write_json(run_dir / "wrist_velocity_peaks.json", {"schema_version": 1, "peaks": []})
    _write_json(run_dir / "ball_inflections.json", {"schema_version": 1, "candidates": []})

    summary = run_pipeline(
        clip="clip_001",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="ball_events",
        runners=_noop_dependency_runners(),
    )

    ball_stage = summary["stages"][-1]
    assert ball_stage["metrics"]["contact_event_count"] == 1
    assert "fused contact_windows.json from wrist and ball cue artifacts" in ball_stage["notes"]


def test_ball_stage_runner_fails_closed_when_no_click_source_artifact_is_missing(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs" / "clip_001"
    inputs.mkdir(parents=True)
    run_dir = tmp_path / "runs" / "clip_001"
    _write_dependency_artifacts(run_dir)

    summary = run_pipeline(
        clip="clip_001",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="ball_events",
        runners=_noop_dependency_runners(),
    )

    assert summary["status"] == "fail"
    ball_stage = summary["stages"][-1]
    assert ball_stage["stage"] == "ball_events"
    assert ball_stage["status"] == "fail"
    assert any("missing no-click BALL source artifact" in note for note in ball_stage["notes"])
    assert not (run_dir / "ball_track.json").exists()
    assert not (run_dir / "contact_windows.json").exists()


def test_ball_stage_runner_does_not_read_prototype_root_without_explicit_opt_in(tmp_path: Path) -> None:
    clip = "clip_001"
    inputs = tmp_path / "inputs" / clip
    inputs.mkdir(parents=True)
    run_dir = tmp_path / "runs" / clip
    prototype_root = tmp_path / "prototype_gate_h100_v2"
    stale_source = prototype_root / clip / "tracknet_smoke_0000_0010" / "ball_track_fusion_temporal_vball100_localtraj.json"
    _write_json(stale_source, _ball_track_payload())
    _write_dependency_artifacts(run_dir)

    runners = _noop_dependency_runners()
    runners["ball_events"] = BallStageRunner(prototype_root=prototype_root)

    summary = run_pipeline(
        clip=clip,
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="ball_events",
        runners=runners,
    )

    assert summary["status"] == "fail"
    ball_stage = summary["stages"][-1]
    assert any("missing no-click BALL source artifact" in note for note in ball_stage["notes"])
    assert str(stale_source) not in ball_stage["notes"][0]
    assert not (run_dir / "ball_track.json").exists()


def test_ball_stage_runner_blocks_empty_contact_windows_even_when_artifacts_are_schema_valid(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs" / "clip_001"
    run_dir = tmp_path / "runs" / "clip_001"
    _write_no_click_ball_source(inputs)
    _write_dependency_artifacts(run_dir)

    summary = run_pipeline(
        clip="clip_001",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="ball_events",
        runners=_noop_dependency_runners(),
    )

    assert summary["status"] == "blocked"
    ball_stage = summary["stages"][-1]
    assert ball_stage["status"] == "blocked"
    assert ball_stage["metrics"]["contact_event_count"] == 0
    assert any("BALL contact windows are empty" in note for note in ball_stage["notes"])
    ball_track = validate_artifact_file("ball_track", run_dir / "ball_track.json")
    contact_windows = validate_artifact_file("contact_windows", run_dir / "contact_windows.json")
    assert isinstance(ball_track, BallTrack)
    assert isinstance(contact_windows, ContactWindows)
    assert len(ball_track.frames) == 3
    assert contact_windows.events == []


def test_ball_stage_runner_fuses_trusted_contact_windows_from_required_cue_artifacts(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs" / "clip_001"
    run_dir = tmp_path / "runs" / "clip_001"
    _write_no_click_ball_source(inputs)
    _write_dependency_artifacts(run_dir)
    _write_contact_cue_artifacts(inputs)

    runners = _noop_dependency_runners()
    runners["ball_events"] = BallStageRunner(contact_fusion_mode="audio_wrist_ball")

    summary = run_pipeline(
        clip="clip_001",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="ball_events",
        runners=runners,
    )

    assert summary["status"] == "blocked"
    assert summary["readiness"]["status"] == "not_ready"
    assert "calibration:court_line_evidence_not_ready" in summary["readiness"]["semantic_blockers"]
    assert "body_compute_execution.json" in summary["review_artifacts"]["reused_artifacts"]
    ball_stage = summary["stages"][-1]
    assert ball_stage["metrics"]["contact_event_count"] == 1
    assert "fused contact_windows.json from audio, wrist, and ball cue artifacts" in ball_stage["notes"]
    contact_windows = validate_artifact_file("contact_windows", run_dir / "contact_windows.json")
    tracks = validate_artifact_file("tracks", run_dir / "tracks.json")
    ball_track = validate_artifact_file("ball_track", run_dir / "ball_track.json")
    assert isinstance(contact_windows, ContactWindows)
    assert isinstance(ball_track, BallTrack)
    assert len(contact_windows.events) == 1
    assert contact_windows.events[0].player_id == 7

    frame_plan = build_frame_compute_plan(
        tracks,
        ball_track=ball_track,
        contact_windows=contact_windows,
        expected_players=1,
    )
    assert frame_plan["summary"]["deep_mesh_window_count"] == 1
    assert frame_plan["deep_mesh_windows"][0]["target_player_ids"] == [7]
    _write_json(run_dir / "frame_compute_plan.json", frame_plan)
    execution = build_body_compute_execution(tracks, frame_plan_path=run_dir / "frame_compute_plan.json")
    assert execution["summary"]["scheduled_player_frame_count"] == 2


def test_ball_stage_runner_can_opt_into_wrist_ball_contact_fusion_without_audio(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs" / "clip_001"
    run_dir = tmp_path / "runs" / "clip_001"
    _write_no_click_ball_source(inputs)
    _write_dependency_artifacts(run_dir)
    _write_contact_cue_artifacts(inputs)
    (inputs / "audio_onsets.json").unlink()

    runners = _noop_dependency_runners()
    runners["ball_events"] = BallStageRunner(contact_fusion_mode="wrist_ball")

    summary = run_pipeline(
        clip="clip_001",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="ball_events",
        runners=runners,
    )

    ball_stage = summary["stages"][-1]
    assert ball_stage["metrics"]["contact_event_count"] == 1
    assert "fused contact_windows.json from wrist and ball cue artifacts" in ball_stage["notes"]
    contact_windows = validate_artifact_file("contact_windows", run_dir / "contact_windows.json")
    assert isinstance(contact_windows, ContactWindows)
    assert len(contact_windows.events) == 1
    assert contact_windows.events[0].player_id == 7
    assert contact_windows.events[0].sources.audio is None


def test_ball_stage_runner_default_fuses_wrist_ball_contacts_without_audio(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs" / "clip_001"
    run_dir = tmp_path / "runs" / "clip_001"
    _write_no_click_ball_source(inputs)
    _write_dependency_artifacts(run_dir)
    _write_contact_cue_artifacts(inputs)
    (inputs / "audio_onsets.json").unlink()

    summary = run_pipeline(
        clip="clip_001",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="ball_events",
        runners=_noop_dependency_runners(),
    )

    ball_stage = summary["stages"][-1]
    assert ball_stage["metrics"]["contact_fusion_mode"] == "wrist_ball"
    assert ball_stage["metrics"]["contact_event_count"] == 1
    assert "fused contact_windows.json from wrist and ball cue artifacts" in ball_stage["notes"]
    contact_windows = validate_artifact_file("contact_windows", run_dir / "contact_windows.json")
    assert isinstance(contact_windows, ContactWindows)
    assert len(contact_windows.events) == 1
    assert contact_windows.events[0].sources.audio is None
