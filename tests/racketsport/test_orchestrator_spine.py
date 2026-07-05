from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS
from threed.racketsport.court_positioning_artifacts import build_court_keypoints_artifact
from threed.racketsport.orchestrator import ExternalCalibrationRunner, StageRun, run_pipeline
from threed.racketsport.court_line_evidence import aggregate_court_line_evidence
from threed.racketsport.schemas import CourtCalibration, CourtLineEvidence, CourtLineObservation, NetLineObservation, Tracks, VirtualWorld, validate_artifact_file


def _sidecar_payload() -> dict:
    return {
        "schema_version": 1,
        "device_tier": "B_standard",
        "device_model": "iPhone16,2",
        "fps": 30,
        "format": "hevc",
        "resolution": [1920, 1080],
        "orientation": "landscape",
        "locked": {"exposure_s": 0.001, "iso": 320, "focus": 0.7, "wb_locked": True},
        "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0, "dist": [], "source": "manual"},
        "arkit_camera_pose": {"R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], "t": [0.0, 0.0, 15.0]},
        "court_plane": {"point": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, 1.0]},
        "manual_court_taps": [[756.8, 88.4896], [1163.2, 88.4896], [1163.2, 991.5104], [756.8, 991.5104]],
        "gravity": [0.0, -1.0, 0.0],
        "lidar_depth_refs": [],
        "ondevice_pose_track": None,
        "capture_quality": {"grade": "good", "reasons": []},
    }


def _metric_sidecar_payload() -> dict:
    payload = _sidecar_payload()
    payload["intrinsics"] = {"fx": 100.0, "fy": 100.0, "cx": 500.0, "cy": 500.0, "dist": [], "source": "arkit"}
    payload["arkit_camera_pose"] = {
        "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
        "t": [0.0, 0.0, 10.0],
    }
    payload["court_plane"] = {"point": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, 1.0]}
    payload["manual_court_taps"] = []
    return payload


def _metric_court_keypoints_payload(sidecar_payload: dict) -> dict:
    intrinsics = sidecar_payload["intrinsics"]
    return build_court_keypoints_artifact(
        frame_indexes=[0, 15, 29],
        keypoints={
            point.name: {
                "uv": [
                    intrinsics["cx"] + intrinsics["fx"] * point.world_xyz_m[0] / 10.0,
                    intrinsics["cy"] + intrinsics["fy"] * point.world_xyz_m[1] / 10.0,
                ],
                "confidence": 0.9,
                "inlier_frames": [0, 15, 29],
                "recovered": False,
            }
            for point in PICKLEBALL_KEYPOINTS
        },
        target_court_score=0.95,
        source="test_synthetic_aggregate",
    )


def _write_inputs(inputs_dir: Path) -> None:
    inputs_dir.mkdir(parents=True)
    (inputs_dir / "capture_sidecar.json").write_text(json.dumps(_sidecar_payload()), encoding="utf-8")
    (inputs_dir / "detections.json").write_text(
        json.dumps(
            {
                "fps": 30.0,
                "frames": [
                    {"frame": 0, "detections": [{"bbox": [940.0, 440.0, 980.0, 540.0], "conf": 0.91, "class": "person", "player_id": 7}]},
                    {"frame": 1, "detections": [{"bbox": [942.0, 440.0, 982.0, 540.0], "conf": 0.89, "class": "person", "player_id": 7}]},
                ],
            }
        ),
        encoding="utf-8",
    )
    frames = inputs_dir / "body_frames"
    frames.mkdir()
    (frames / "frame_000000.jpg").write_bytes(b"not decoded by default pose runtime")
    (frames / "frame_000001.jpg").write_bytes(b"not decoded by default pose runtime")


def _write_missing_sam3d_manifest(path: Path) -> Path:
    manifest = {
        "schema_version": 1,
        "models": [
            {
                "id": "unused_validation_model",
                "stage": "validation_only",
                "use": "Deliberately incomplete manifest for fail-closed BODY tests",
                "source": "local_test_fixture",
                "license": "Apache-2.0",
                "commercial_posture": "ok",
                "status": "missing",
                "local_path": str(path.parent / "missing-validation-model.pth"),
                "sha256": "0" * 64,
                "fallbacks": [],
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_calibration_runner_prefers_metric_court_keypoints_when_available(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs" / "clip_001"
    run_dir = tmp_path / "runs" / "phase1" / "clip_001"
    inputs.mkdir(parents=True)
    sidecar_payload = _metric_sidecar_payload()
    (inputs / "capture_sidecar.json").write_text(json.dumps(sidecar_payload), encoding="utf-8")
    (inputs / "court_keypoints.json").write_text(
        json.dumps(_metric_court_keypoints_payload(sidecar_payload)),
        encoding="utf-8",
    )

    summary = run_pipeline(clip="clip_001", inputs_dir=inputs, run_dir=run_dir, stage="calibration")

    assert summary["status"] == "blocked"
    assert summary["stages"][0]["status"] == "ran"
    assert summary["stages"][0]["wall_seconds"] >= 0.0
    assert summary["stages"][0]["source_mode"] == "arkit_plane_keypoints"
    calibration = validate_artifact_file("court_calibration", run_dir / "court_calibration.json")
    assert isinstance(calibration, CourtCalibration)
    assert calibration.source == "arkit_plane_keypoint_metric_solve_v1"
    assert calibration.metric_confidence == "high"


def test_pipeline_runs_body_without_legacy_pose_dependency_and_fails_without_sam3d_frames(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs" / "clip_001"
    run_dir = tmp_path / "runs" / "phase11" / "clip_001"
    _write_inputs(inputs)
    manifest = _write_missing_sam3d_manifest(tmp_path / "models" / "MANIFEST.json")

    summary = run_pipeline(clip="clip_001", inputs_dir=inputs, run_dir=run_dir, stage="body", tracking_mode="precomputed", manifest_path=manifest)

    assert summary["status"] == "fail"
    assert [item["stage"] for item in summary["stages"]] == ["calibration", "tracking", "body"]
    assert summary["stages"][0]["status"] == "ran"
    assert summary["stages"][0]["real_model"] is False
    assert summary["stages"][1]["status"] == "ran"
    assert summary["stages"][1]["real_model"] is False
    assert summary["stages"][1]["source_mode"] == "precomputed_detections"
    assert summary["stages"][2]["status"] == "fail"
    assert summary["stages"][2]["real_model"] is True
    assert summary["stages"][2]["source_mode"] == "fast_sam_3d_body"
    assert any("adaptive BODY schedule contains no SAM3D body-mode frames" in note for note in summary["stages"][2]["notes"])
    assert validate_artifact_file("court_calibration", run_dir / "court_calibration.json")
    evidence = validate_artifact_file("court_line_evidence", run_dir / "court_line_evidence.json")
    assert isinstance(evidence, CourtLineEvidence)
    assert evidence.aggregate.auto_calibration_ready is False
    tracks = validate_artifact_file("tracks", run_dir / "tracks.json")
    assert isinstance(tracks, Tracks)
    assert not (run_dir / "skeleton3d.json").exists()
    assert not (run_dir / "smpl_motion.json").exists()
    frame_plan = json.loads((run_dir / "frame_compute_plan.json").read_text(encoding="utf-8"))
    assert frame_plan["artifact_type"] == "racketsport_frame_compute_plan"
    assert frame_plan["summary"]["human_review_frame_count"] == 2
    virtual_world = validate_artifact_file("virtual_world", run_dir / "virtual_world.json")
    assert isinstance(virtual_world, VirtualWorld)
    assert virtual_world.summary.warnings == ["missing_mesh_vertices", "missing_ball_track", "missing_paddle_pose"]
    body_execution = json.loads((run_dir / "body_compute_execution.json").read_text(encoding="utf-8"))
    assert body_execution["artifact_type"] == "racketsport_body_compute_execution"
    assert body_execution["summary"]["scheduled_frame_count"] == 0
    assert summary["review_artifacts"]["produced_artifacts"] == [
        "frame_compute_plan.json",
        "body_compute_execution.json",
        "virtual_world.json",
    ]


def test_calibration_runner_samples_video_for_court_line_evidence(tmp_path: Path, monkeypatch) -> None:
    inputs = tmp_path / "inputs" / "clip_001"
    run_dir = tmp_path / "runs" / "phase1" / "clip_001"
    _write_inputs(inputs)
    video_path = inputs / "video.mp4"
    video_path.write_bytes(b"not-real-video-but-discovered-before-cv2")
    calls: list[Path] = []

    def fake_video_evidence(path, calibration, *, net_plane, sample_count):
        calls.append(Path(path))
        evidence = _ready_court_line_evidence(calibration.sport)
        evidence.source = "stub_video_sampler"
        return evidence

    monkeypatch.setattr(
        "threed.racketsport.orchestrator.build_auto_court_line_evidence_from_video",
        fake_video_evidence,
    )

    summary = run_pipeline(clip="clip_001", inputs_dir=inputs, run_dir=run_dir, stage="calibration")

    assert summary["status"] == "pass"
    assert calls == [video_path]
    evidence = validate_artifact_file("court_line_evidence", run_dir / "court_line_evidence.json")
    assert isinstance(evidence, CourtLineEvidence)
    assert evidence.source == "stub_video_sampler"
    assert evidence.aggregate.auto_calibration_ready is True


def test_pipeline_review_frame_plan_uses_explicit_ball_source_after_body_failure(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs" / "clip_001"
    run_dir = tmp_path / "runs" / "phase11" / "clip_001"
    _write_inputs(inputs)
    ball_source = inputs / "strict_ball_track.json"
    ball_source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fps": 30.0,
                "source": "tracknet",
                "frames": [
                    {"t": 0.0, "xy": [300.0, 200.0], "conf": 0.1, "visible": False},
                    {"t": 1.0 / 30.0, "xy": [310.0, 210.0], "conf": 0.9, "visible": True},
                ],
                "bounces": [],
            }
        ),
        encoding="utf-8",
    )

    summary = run_pipeline(
        clip="clip_001",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="body",
        tracking_mode="precomputed",
        ball_source_path=ball_source,
    )

    assert summary["status"] == "fail"
    frame_plan = json.loads((run_dir / "frame_compute_plan.json").read_text(encoding="utf-8"))
    assert frame_plan["summary"]["by_reason"]["ball_uncertain"] == 1
    assert summary["review_artifacts"]["produced_artifacts"] == [
        "frame_compute_plan.json",
        "body_compute_execution.json",
        "virtual_world.json",
    ]


def test_pipeline_summary_blocks_when_schema_valid_artifacts_are_not_semantically_ready(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs" / "clip_001"
    run_dir = tmp_path / "runs" / "phase2" / "clip_001"
    _write_inputs(inputs)

    summary = run_pipeline(clip="clip_001", inputs_dir=inputs, run_dir=run_dir, stage="tracking", tracking_mode="precomputed")

    assert summary["status"] == "blocked"
    assert [item["stage"] for item in summary["stages"]] == ["calibration", "tracking"]
    assert all(item["status"] == "ran" for item in summary["stages"])
    assert summary["readiness"]["status"] == "not_ready"
    assert "calibration:court_line_evidence_not_ready" in summary["readiness"]["semantic_blockers"]


def test_pipeline_precomputed_tracks_mode_uses_tracks_without_detections(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs" / "clip_001"
    run_dir = tmp_path / "runs" / "phase2" / "clip_001"
    inputs.mkdir(parents=True)
    (inputs / "capture_sidecar.json").write_text(json.dumps(_sidecar_payload()), encoding="utf-8")
    (inputs / "tracks.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fps": 30.0,
                "players": [
                    {
                        "id": 1,
                        "side": "near",
                        "role": "left",
                        "frames": [
                            {"t": 0.0, "bbox": [940.0, 440.0, 980.0, 540.0], "world_xy": [0.0, 0.0], "conf": 0.91},
                            {
                                "t": 1.0 / 30.0,
                                "bbox": [942.0, 440.0, 982.0, 540.0],
                                "world_xy": [0.1, 0.0],
                                "conf": 0.89,
                            },
                        ],
                    }
                ],
                "rally_spans": [],
            }
        ),
        encoding="utf-8",
    )

    summary = run_pipeline(clip="clip_001", inputs_dir=inputs, run_dir=run_dir, stage="tracking", tracking_mode="precomputed_tracks")

    assert summary["status"] == "blocked"
    assert [item["stage"] for item in summary["stages"]] == ["calibration", "tracking"]
    assert summary["stages"][1]["source_mode"] == "precomputed_tracks"
    assert summary["stages"][1]["metrics"]["track_frame_count"] == 2
    tracks = validate_artifact_file("tracks", run_dir / "tracks.json")
    assert isinstance(tracks, Tracks)
    assert len(tracks.players) == 1
    assert len(tracks.players[0].frames) == 2


def test_pipeline_overwrites_existing_review_frame_plan_from_current_inputs(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs" / "clip_001"
    run_dir = tmp_path / "runs" / "phase11" / "clip_001"
    _write_inputs(inputs)
    run_dir.mkdir(parents=True)
    stale_plan = {
        "schema_version": 1,
        "artifact_type": "racketsport_frame_compute_plan",
        "fps": 30.0,
        "expected_players": 4,
        "frame_count": 0,
        "frames": [],
        "deep_mesh_windows": [],
        "summary": {"frame_count": 0, "deep_mesh_window_count": 0, "stale_marker": True},
    }
    frame_plan_path = run_dir / "frame_compute_plan.json"
    frame_plan_path.write_text(json.dumps(stale_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = run_pipeline(clip="clip_001", inputs_dir=inputs, run_dir=run_dir, stage="body", tracking_mode="precomputed")

    assert summary["status"] == "fail"
    rewritten_plan = json.loads(frame_plan_path.read_text(encoding="utf-8"))
    assert rewritten_plan["summary"].get("stale_marker") is None
    assert rewritten_plan["frame_count"] == 2
    assert "frame_compute_plan.json" in summary["review_artifacts"]["produced_artifacts"]
    assert summary["review_artifacts"]["reused_artifacts"] == []
    assert "body_compute_execution.json" in summary["review_artifacts"]["produced_artifacts"]


def test_video_backed_calibration_is_advisory_not_blocking_for_trusted_manual_taps(tmp_path: Path, monkeypatch) -> None:
    """Task #45 S1: ManualCalibrationRunner's manual-taps branch (real human-tapped
    corners against a declared image_size, via capture_sidecar.json) is a trusted
    calibration source, so a not-ready automatic court-line/net evidence result no
    longer hard-fails the calibration stage -- it becomes an advisory note and the
    pipeline proceeds past calibration into tracking. (The overall run still reports
    "blocked", not "pass": pipeline_contracts' readiness report separately and
    correctly still flags court_line_evidence as not semantically ready -- that
    downstream signal is untouched by this fix, only the hard *stage failure* is.)"""

    inputs = tmp_path / "inputs" / "clip_001"
    run_dir = tmp_path / "runs" / "phase1" / "clip_001"
    _write_inputs(inputs)
    (inputs / "video.mp4").write_bytes(b"not-real-video-but-discovered-before-cv2")

    def fake_video_evidence(path, calibration, *, net_plane, sample_count):
        evidence = aggregate_court_line_evidence(
            sport=calibration.sport,
            line_observations=[],
            net_observations=[],
            required_line_ids=("near_nvz", "far_nvz", "near_centerline", "far_centerline"),
            required_net_ids=("top_net",),
        )
        evidence.source = "stub_video_sampler_not_ready"
        return evidence

    monkeypatch.setattr(
        "threed.racketsport.orchestrator.build_auto_court_line_evidence_from_video",
        fake_video_evidence,
    )

    summary = run_pipeline(clip="clip_001", inputs_dir=inputs, run_dir=run_dir, stage="tracking", tracking_mode="precomputed")

    assert summary["status"] == "blocked"
    assert [item["stage"] for item in summary["stages"]] == ["calibration", "tracking"]
    calibration_stage = summary["stages"][0]
    assert calibration_stage["status"] == "ran"
    assert any("ADVISORY" in note and "automatic court evidence not ready" in note for note in calibration_stage["notes"])
    assert calibration_stage["metrics"]["calibration_confidence"]["reprojection_median_px"] == 0.0
    assert (run_dir / "court_calibration.json").is_file()
    assert (run_dir / "tracks.json").is_file()
    # readiness still honestly reports the evidence isn't semantically ready -- the fix
    # is that this no longer hard-fails the calibration *stage*.
    assert "calibration:court_line_evidence_not_ready" in summary["readiness"]["semantic_blockers"]


def test_video_backed_calibration_still_fails_closed_with_no_calibration_seed(tmp_path: Path, monkeypatch) -> None:
    """The untrusted/no-tap path (no capture_sidecar.json at all -- no calibration
    input whatsoever) must keep its exact fail-closed behavior; Task #45 S1 only makes
    the automatic-evidence gate advisory for an already-trusted calibration source, it
    does not weaken the "no calibration input at all" hard failure."""

    inputs = tmp_path / "inputs" / "clip_001"
    run_dir = tmp_path / "runs" / "phase1" / "clip_001"
    inputs.mkdir(parents=True)
    (inputs / "video.mp4").write_bytes(b"not-real-video-but-discovered-before-cv2")

    summary = run_pipeline(clip="clip_001", inputs_dir=inputs, run_dir=run_dir, stage="tracking", tracking_mode="precomputed")

    assert summary["status"] == "fail"
    assert [item["stage"] for item in summary["stages"]] == ["calibration"]
    assert any("no trusted no-tap calibration seed" in note for note in summary["stages"][0]["notes"])
    assert not (run_dir / "tracks.json").exists()
    assert not (run_dir / "court_calibration.json").exists()


def test_calibration_runner_attempts_video_only_court_evidence_before_missing_sidecar_failure(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs" / "clip_001"
    run_dir = tmp_path / "runs" / "phase1" / "clip_001"
    inputs.mkdir(parents=True)
    (inputs / "video.mp4").write_bytes(b"placeholder")

    summary = run_pipeline(clip="clip_001", inputs_dir=inputs, run_dir=run_dir, stage="calibration")

    assert summary["status"] == "fail"
    assert summary["stages"][0]["stage"] == "calibration"
    assert summary["stages"][0]["source_mode"] == "manual_sidecar"
    assert any("no trusted no-tap calibration seed" in note for note in summary["stages"][0]["notes"])
    assert (run_dir / "court_zones.json").is_file()
    assert (run_dir / "net_plane.json").is_file()
    evidence = validate_artifact_file("court_line_evidence", run_dir / "court_line_evidence.json")
    assert isinstance(evidence, CourtLineEvidence)
    assert evidence.source == "auto_video_no_calibration_seed"
    assert "missing_calibration_seed" in evidence.aggregate.reasons
    assert not (run_dir / "court_calibration.json").exists()


def test_pipeline_fails_closed_when_runner_outputs_invalid_artifact(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs" / "clip_001"
    run_dir = tmp_path / "runs" / "phase11" / "clip_001"
    _write_inputs(inputs)
    (inputs / "detections.json").write_text(json.dumps({"fps": 30.0, "frames": [{"detections": [{"bbox": [1, 2, 3, 4], "class": "person"}]}]}), encoding="utf-8")

    summary = run_pipeline(clip="clip_001", inputs_dir=inputs, run_dir=run_dir, stage="tracking", tracking_mode="precomputed")

    assert summary["status"] == "fail"
    assert summary["stages"][-1]["stage"] == "tracking"
    assert summary["stages"][-1]["status"] == "fail"
    assert any("tracking failed" in note for note in summary["stages"][-1]["notes"])


def test_orchestrator_cli_writes_summary_and_does_not_fake_e2e(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs" / "clip_001"
    run_dir = tmp_path / "runs" / "phase11" / "clip_001"
    _write_inputs(inputs)
    manifest = _write_missing_sam3d_manifest(tmp_path / "models" / "MANIFEST.json")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "threed.racketsport.orchestrator",
            "--clip",
            "clip_001",
            "--inputs",
            str(inputs),
            "--out",
            str(run_dir),
            "--stage",
            "e2e",
            "--tracking-mode",
            "precomputed",
            "--manifest",
            str(manifest),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    payload = json.loads((run_dir / "pipeline_run.json").read_text(encoding="utf-8"))
    assert payload["status"] == "fail"
    assert payload["stages"][2]["stage"] == "body"
    assert payload["stages"][2]["status"] == "fail"
    assert any("adaptive BODY schedule contains no SAM3D body-mode frames" in note for note in payload["stages"][2]["notes"])
    assert not (run_dir / "replay_scene.json").exists()


def _ready_court_line_evidence(sport: str) -> CourtLineEvidence:
    lines = [
        CourtLineObservation(
            line_id=line_id,
            image_segment=[[10.0, 10.0], [100.0, 10.0]],
            confidence=0.95,
            frame_indexes=[0, 1],
            residual_px={"mean": 1.0, "p95": 2.0},
            visible_fraction=0.9,
            source="test",
        )
        for line_id in ("near_nvz", "far_nvz", "near_centerline", "far_centerline")
    ]
    net = NetLineObservation(
        net_id="top_net",
        image_points=[[10.0, 8.0], [55.0, 8.0], [100.0, 8.0]],
        confidence=0.95,
        frame_indexes=[0, 1],
        residual_px={"mean": 1.0, "p95": 2.0},
        source="test",
    )
    return aggregate_court_line_evidence(
        sport=sport,  # type: ignore[arg-type]
        line_observations=lines,
        net_observations=[net],
        required_line_ids=("near_nvz", "far_nvz", "near_centerline", "far_centerline"),
        required_net_ids=("top_net",),
    )


# ---------------------------------------------------------------------------
# ExternalCalibrationRunner (Task #33 CAL-MIGRATION: --court-calibration consumption)
# ---------------------------------------------------------------------------


def _metric_calibration_payload(*, sport: str = "pickleball", source: str = "metric_15pt_reviewed", dist: list[float] | None = None) -> dict:
    # A "metric"-flavored CourtCalibration (coordinate_frame/T_world_court/
    # metric_confidence/gsd_model/source/solved_over_frames all set) requires
    # per_keypoint_residual_px to have exactly the 15 canonical pickleball
    # keypoints -- see schemas._point_lists_must_be_paired.
    world_pts = [list(point.world_xyz_m) for point in PICKLEBALL_KEYPOINTS]
    image_pts = [[100.0 + 50.0 * i, 300.0 - 10.0 * i] for i in range(len(world_pts))]
    return {
        "schema_version": 1,
        "sport": sport,
        "coordinate_frame": "court_netcenter_z_up_m",
        "T_world_court": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "homography": [[100.0, 0.0, 1000.0], [0.0, 100.0, 1000.0], [0.0, 0.0, 1.0]],
        "intrinsics": {"fx": 1391.18, "fy": 1391.18, "cx": 960.0, "cy": 540.0, "dist": dist or [], "source": source},
        "image_size": [1920, 1080],
        "extrinsics": {"R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], "t": [0.0, 0.0, 10.0], "camera_height_m": 1.73},
        "reprojection_error_px": {"median": 5.0, "p95": 19.9},
        "per_keypoint_residual_px": [2.0 for _ in image_pts],
        "metric_confidence": "low",
        "gsd_model": {"type": "analytic_ray_plane", "plane_sigma_m": 0.0, "calibration_sigma_m": 0.0, "samples": []},
        "capture_quality": {"grade": "warn", "reasons": ["reviewed_15pt_correspondences"]},
        "image_pts": image_pts,
        "world_pts": world_pts,
        "source": source,
        "solved_over_frames": [0, 30, 60],
    }


def _write_json_file(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _external_calibration_inputs(tmp_path: Path, clip: str) -> Path:
    inputs = tmp_path / "inputs" / clip
    inputs.mkdir(parents=True)
    (inputs / "source.mp4").write_bytes(b"not-real-video-but-discovered-before-cv2")
    return inputs


def test_external_calibration_runner_consumes_trusted_metric_source(tmp_path: Path, monkeypatch) -> None:
    inputs = _external_calibration_inputs(tmp_path, "clip_ext_ok")
    run_dir = tmp_path / "runs" / "clip_ext_ok"
    external_path = _write_json_file(tmp_path / "external" / "court_calibration_metric15pt.json", _metric_calibration_payload(dist=[0.0, 0.0, 0.0, 0.0]))

    monkeypatch.setattr(
        "threed.racketsport.orchestrator.build_auto_court_line_evidence_from_video",
        lambda path, calibration, *, net_plane, sample_count: _ready_court_line_evidence(calibration.sport),
    )

    summary = run_pipeline(
        clip="clip_ext_ok",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="calibration",
        runners={"calibration": ExternalCalibrationRunner(source_path=external_path)},
    )

    assert summary["status"] == "pass"
    stage = summary["stages"][0]
    assert stage["stage"] == "calibration"
    assert stage["status"] == "ran"
    assert stage["real_model"] is False
    assert stage["source_mode"] == "external_metric_calibration"
    assert any("consumed externally-provided" in note for note in stage["notes"])
    assert stage["metrics"]["intrinsics_source"] == "metric_15pt_reviewed"
    assert stage["metrics"]["intrinsics_dist_nonzero"] is False

    calibration = validate_artifact_file("court_calibration", run_dir / "court_calibration.json")
    assert isinstance(calibration, CourtCalibration)
    assert calibration.intrinsics.source == "metric_15pt_reviewed"
    assert calibration.intrinsics.fx == pytest.approx(1391.18)
    assert (run_dir / "court_zones.json").is_file()
    assert (run_dir / "net_plane.json").is_file()
    assert (run_dir / "court_line_evidence.json").is_file()


def test_external_calibration_runner_flags_nonzero_distortion(tmp_path: Path, monkeypatch) -> None:
    inputs = _external_calibration_inputs(tmp_path, "clip_ext_dist")
    run_dir = tmp_path / "runs" / "clip_ext_dist"
    dist = [-0.30035182958629364, 0.09861181595540636, 0.0, 0.0]
    external_path = _write_json_file(tmp_path / "external" / "court_calibration_metric15pt.json", _metric_calibration_payload(dist=dist))

    monkeypatch.setattr(
        "threed.racketsport.orchestrator.build_auto_court_line_evidence_from_video",
        lambda path, calibration, *, net_plane, sample_count: _ready_court_line_evidence(calibration.sport),
    )

    summary = run_pipeline(
        clip="clip_ext_dist",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="calibration",
        runners={"calibration": ExternalCalibrationRunner(source_path=external_path)},
    )

    assert summary["status"] == "pass"
    stage = summary["stages"][0]
    assert stage["metrics"]["intrinsics_dist_nonzero"] is True
    assert any("intrinsics.dist is nonzero" in note for note in stage["notes"])
    assert any("undistort" in note for note in stage["notes"])


def test_external_calibration_runner_rejects_untrusted_intrinsics_source(tmp_path: Path) -> None:
    inputs = _external_calibration_inputs(tmp_path, "clip_ext_untrusted")
    run_dir = tmp_path / "runs" / "clip_ext_untrusted"
    external_path = _write_json_file(
        tmp_path / "external" / "court_calibration.json",
        _metric_calibration_payload(source="estimated_from_declared_court_corners"),
    )

    summary = run_pipeline(
        clip="clip_ext_untrusted",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="calibration",
        runners={"calibration": ExternalCalibrationRunner(source_path=external_path)},
    )

    assert summary["status"] == "fail"
    assert any("not a trusted external calibration source" in note for note in summary["stages"][0]["notes"])
    assert not (run_dir / "court_calibration.json").is_file()


def test_external_calibration_runner_rejects_sport_mismatch(tmp_path: Path) -> None:
    inputs = _external_calibration_inputs(tmp_path, "clip_ext_sport")
    run_dir = tmp_path / "runs" / "clip_ext_sport"
    external_path = _write_json_file(tmp_path / "external" / "court_calibration.json", _metric_calibration_payload(sport="tennis"))

    summary = run_pipeline(
        clip="clip_ext_sport",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="calibration",
        sport="pickleball",
        runners={"calibration": ExternalCalibrationRunner(source_path=external_path)},
    )

    assert summary["status"] == "fail"
    assert any("does not match" in note for note in summary["stages"][0]["notes"])


def test_external_calibration_runner_rejects_schema_invalid_file(tmp_path: Path) -> None:
    inputs = _external_calibration_inputs(tmp_path, "clip_ext_badschema")
    run_dir = tmp_path / "runs" / "clip_ext_badschema"
    payload = _metric_calibration_payload()
    del payload["homography"]
    external_path = _write_json_file(tmp_path / "external" / "court_calibration.json", payload)

    summary = run_pipeline(
        clip="clip_ext_badschema",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="calibration",
        runners={"calibration": ExternalCalibrationRunner(source_path=external_path)},
    )

    assert summary["status"] == "fail"


def test_external_calibration_runner_missing_file_fails_closed(tmp_path: Path) -> None:
    inputs = _external_calibration_inputs(tmp_path, "clip_ext_missing")
    run_dir = tmp_path / "runs" / "clip_ext_missing"
    missing_path = tmp_path / "external" / "does_not_exist.json"

    summary = run_pipeline(
        clip="clip_ext_missing",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="calibration",
        runners={"calibration": ExternalCalibrationRunner(source_path=missing_path)},
    )

    assert summary["status"] == "fail"
    assert any("not found" in note for note in summary["stages"][0]["notes"])


def test_external_calibration_runner_advisory_when_video_evidence_not_ready(tmp_path: Path) -> None:
    # No monkeypatch of build_auto_court_line_evidence_from_video: the garbage
    # source.mp4 bytes cannot actually be decoded, so real automatic court-line
    # evidence detection fails. Task #45 S1: ExternalCalibrationRunner only ever
    # consumes calibrations whose intrinsics.source is already in
    # trusted_intrinsics_sources (rejected earlier otherwise -- see
    # test_external_calibration_runner_rejects_untrusted_intrinsics_source), so this no
    # longer hard-fails the calibration *stage* -- it becomes an advisory note. The
    # overall pipeline_run.json status is still "blocked" (not "pass"): the readiness
    # report separately and correctly flags court_line_evidence as not semantically
    # ready, which this fix intentionally does not touch.
    inputs = _external_calibration_inputs(tmp_path, "clip_ext_unready")
    run_dir = tmp_path / "runs" / "clip_ext_unready"
    external_path = _write_json_file(tmp_path / "external" / "court_calibration_metric15pt.json", _metric_calibration_payload())

    summary = run_pipeline(
        clip="clip_ext_unready",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="calibration",
        runners={"calibration": ExternalCalibrationRunner(source_path=external_path)},
    )

    assert summary["status"] == "blocked"
    stage = summary["stages"][0]
    assert stage["status"] == "ran"
    assert any("ADVISORY" in note and "automatic court evidence not ready" in note for note in stage["notes"])
    assert stage["metrics"]["calibration_confidence"]["reprojection_median_px"] == 5.0
    assert (run_dir / "court_calibration.json").is_file()
    assert "calibration:court_line_evidence_not_ready" in summary["readiness"]["semantic_blockers"]


def test_external_calibration_runner_untrusted_source_still_fails_closed_regardless_of_evidence(tmp_path: Path) -> None:
    # An untrusted intrinsics.source is rejected before the evidence check is ever
    # reached -- confirms Task #45 S1's advisory carve-out never applies to an
    # unreviewed/guessed external calibration, whatever the video evidence looks like.
    inputs = _external_calibration_inputs(tmp_path, "clip_ext_untrusted_unready")
    run_dir = tmp_path / "runs" / "clip_ext_untrusted_unready"
    external_path = _write_json_file(
        tmp_path / "external" / "court_calibration.json",
        _metric_calibration_payload(source="estimated_from_declared_court_corners"),
    )

    summary = run_pipeline(
        clip="clip_ext_untrusted_unready",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="calibration",
        runners={"calibration": ExternalCalibrationRunner(source_path=external_path)},
    )

    assert summary["status"] == "fail"


# ---------------------------------------------------------------------------
# run_pipeline(reuse_existing_stage_artifacts=...) (Task #45 S2)
# ---------------------------------------------------------------------------


class _RaisingCalibrationRunner:
    """A calibration runner that fails the test loudly if it is ever invoked -- used to
    prove a dependency stage's runner was (or was not) called."""

    stage = "calibration"
    real_model = False
    source_mode = "should_not_be_invoked"

    def run(self, context) -> StageRun:  # noqa: ANN001
        raise AssertionError("calibration runner should not have been invoked -- artifacts already valid on disk")


def _run_real_external_calibration(tmp_path: Path, monkeypatch, clip: str) -> tuple[Path, Path]:
    inputs = _external_calibration_inputs(tmp_path, clip)
    run_dir = tmp_path / "runs" / clip
    external_path = _write_json_file(
        tmp_path / "external" / f"{clip}_court_calibration_metric15pt.json",
        _metric_calibration_payload(dist=[0.0, 0.0, 0.0, 0.0]),
    )
    monkeypatch.setattr(
        "threed.racketsport.orchestrator.build_auto_court_line_evidence_from_video",
        lambda path, calibration, *, net_plane, sample_count: _ready_court_line_evidence(calibration.sport),
    )
    first = run_pipeline(
        clip=clip,
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="calibration",
        runners={"calibration": ExternalCalibrationRunner(source_path=external_path)},
    )
    assert first["status"] == "pass"
    return inputs, run_dir


def test_run_pipeline_reuses_already_valid_dependency_stage_artifacts_when_opted_in(tmp_path: Path, monkeypatch) -> None:
    inputs, run_dir = _run_real_external_calibration(tmp_path, monkeypatch, "clip_reuse_ok")

    second = run_pipeline(
        clip="clip_reuse_ok",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="tracking",
        tracking_mode="precomputed",
        runners={"calibration": _RaisingCalibrationRunner()},
        reuse_existing_stage_artifacts=True,
    )

    calibration_stage = next(s for s in second["stages"] if s["stage"] == "calibration")
    assert calibration_stage["status"] == "ran"
    assert calibration_stage["source_mode"] == "reused_existing_run_artifacts"
    assert any("reuse_existing_stage_artifacts=True" in note for note in calibration_stage["notes"])
    # tracking was attempted for real (proving the pipeline proceeded past calibration
    # instead of stopping there) -- it fails here only because this fixture's inputs_dir
    # has no detections.json, which is unrelated to calibration reuse.
    tracking_stage = next(s for s in second["stages"] if s["stage"] == "tracking")
    assert "should not have been invoked" not in " ".join(tracking_stage["notes"])


def test_run_pipeline_default_still_rederives_dependency_stage_artifacts(tmp_path: Path, monkeypatch) -> None:
    """Regression guard: reuse_existing_stage_artifacts defaults to False, so every
    existing caller/test keeps getting run_pipeline's original full re-derivation
    behavior on every call unless it explicitly opts in."""

    inputs, run_dir = _run_real_external_calibration(tmp_path, monkeypatch, "clip_reuse_default_off")

    second = run_pipeline(
        clip="clip_reuse_default_off",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="tracking",
        tracking_mode="precomputed",
        runners={"calibration": _RaisingCalibrationRunner()},
    )

    assert second["status"] == "fail"
    calibration_stage = second["stages"][0]
    assert calibration_stage["stage"] == "calibration"
    assert calibration_stage["status"] == "fail"
    assert "should not have been invoked" in calibration_stage["notes"][0]


def test_run_pipeline_reuse_never_skips_the_explicitly_requested_stage(tmp_path: Path, monkeypatch) -> None:
    """reuse_existing_stage_artifacts only ever applies to *dependency* stages -- the
    stage a caller explicitly asked to run always gets invoked for real, even if valid
    artifacts for it already happen to be sitting on disk from a prior call."""

    inputs, run_dir = _run_real_external_calibration(tmp_path, monkeypatch, "clip_reuse_target_stage")

    second = run_pipeline(
        clip="clip_reuse_target_stage",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="calibration",
        runners={"calibration": _RaisingCalibrationRunner()},
        reuse_existing_stage_artifacts=True,
    )

    assert second["status"] == "fail"
    assert "should not have been invoked" in second["stages"][0]["notes"][0]


# ---------------------------------------------------------------------------
# calibration stage contract: soft-required artifacts (Task #45 S4)
# ---------------------------------------------------------------------------


class _MinimalCalibrationRunner:
    """Writes only court_calibration.json + court_line_evidence.json, deliberately
    skipping court_zones.json/net_plane.json, to exercise Task #45 S4's relaxed
    contract validation -- no process_video.py stage reads those two files back as
    input (court_zones.json is read only by the offline
    threed.racketsport.court_keypoint_eval eval script; net_plane.json only by
    review/visualization tooling), so a calibration runner that does not happen to
    (re)write them must not hard-fail the stage."""

    stage = "calibration"
    real_model = False
    source_mode = "minimal_test_stub"

    def __init__(self, calibration_payload: dict, evidence) -> None:
        self._calibration_payload = calibration_payload
        self._evidence = evidence

    def run(self, context) -> StageRun:  # noqa: ANN001
        calibration_path = context.run_dir / "court_calibration.json"
        calibration_path.parent.mkdir(parents=True, exist_ok=True)
        calibration_path.write_text(json.dumps(self._calibration_payload), encoding="utf-8")
        evidence_payload = self._evidence.model_dump(mode="json") if hasattr(self._evidence, "model_dump") else self._evidence
        (context.run_dir / "court_line_evidence.json").write_text(json.dumps(evidence_payload), encoding="utf-8")
        return StageRun(
            stage=self.stage,
            status="ran",
            real_model=False,
            source_mode=self.source_mode,
            produced_artifacts=("court_calibration.json", "court_line_evidence.json"),
            notes=("wrote only calibration + evidence, no court_zones.json/net_plane.json",),
        )


def test_calibration_stage_does_not_hard_fail_when_court_zones_or_net_plane_are_missing(tmp_path: Path) -> None:
    inputs = _external_calibration_inputs(tmp_path, "clip_soft_artifacts")
    run_dir = tmp_path / "runs" / "clip_soft_artifacts"
    evidence = _ready_court_line_evidence("pickleball")

    summary = run_pipeline(
        clip="clip_soft_artifacts",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="calibration",
        runners={"calibration": _MinimalCalibrationRunner(_metric_calibration_payload(dist=[0.0, 0.0, 0.0, 0.0]), evidence)},
    )

    # The calibration *stage* itself succeeds (S4's fix) even though court_zones.json/
    # net_plane.json never got written -- the overall run status is still "blocked", not
    # "pass": pipeline_contracts.py's separate readiness report (out of this lane's
    # scope) still lists those two filenames as contract-required and reports them
    # missing there. S4 only stops that from hard-failing the stage.
    assert summary["status"] == "blocked"
    assert summary["stages"][0]["status"] == "ran"
    assert not (run_dir / "court_zones.json").exists()
    assert not (run_dir / "net_plane.json").exists()
