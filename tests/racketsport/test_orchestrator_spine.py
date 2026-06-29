from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from threed.racketsport.orchestrator import run_pipeline
from threed.racketsport.court_line_evidence import aggregate_court_line_evidence
from threed.racketsport.schemas import CourtLineEvidence, CourtLineObservation, NetLineObservation, Tracks, VirtualWorld, validate_artifact_file


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


def test_pipeline_runs_manual_calibration_and_precomputed_tracking_then_fails_missing_body_runtime(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs" / "clip_001"
    run_dir = tmp_path / "runs" / "phase11" / "clip_001"
    _write_inputs(inputs)

    summary = run_pipeline(clip="clip_001", inputs_dir=inputs, run_dir=run_dir, stage="body", tracking_mode="precomputed")

    assert summary["status"] == "fail"
    assert [item["stage"] for item in summary["stages"]] == ["calibration", "tracking", "body"]
    assert summary["stages"][0]["status"] == "ran"
    assert summary["stages"][0]["real_model"] is False
    assert summary["stages"][1]["status"] == "ran"
    assert summary["stages"][1]["real_model"] is False
    assert summary["stages"][1]["source_mode"] == "precomputed_detections"
    assert summary["stages"][2]["status"] == "fail"
    assert summary["stages"][2]["real_model"] is True
    assert any("missing checkpoint for fast_sam_3d_body_dinov3" in note for note in summary["stages"][2]["notes"])
    assert validate_artifact_file("court_calibration", run_dir / "court_calibration.json")
    evidence = validate_artifact_file("court_line_evidence", run_dir / "court_line_evidence.json")
    assert isinstance(evidence, CourtLineEvidence)
    assert evidence.aggregate.auto_calibration_ready is False
    tracks = validate_artifact_file("tracks", run_dir / "tracks.json")
    assert isinstance(tracks, Tracks)
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


def test_video_backed_calibration_fails_closed_when_semantic_court_evidence_not_ready(tmp_path: Path, monkeypatch) -> None:
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

    assert summary["status"] == "fail"
    assert [item["stage"] for item in summary["stages"]] == ["calibration"]
    assert any("automatic court evidence not ready" in note for note in summary["stages"][0]["notes"])
    assert not (run_dir / "tracks.json").exists()


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
    assert any("missing checkpoint for fast_sam_3d_body_dinov3" in note for note in payload["stages"][2]["notes"])
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
