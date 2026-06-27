from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from threed.racketsport.orchestrator import run_pipeline
from threed.racketsport.schemas import Tracks, validate_artifact_file


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


def test_pipeline_runs_real_calibration_and_tracking_then_blocks_missing_body(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs" / "clip_001"
    run_dir = tmp_path / "runs" / "phase11" / "clip_001"
    _write_inputs(inputs)

    summary = run_pipeline(clip="clip_001", inputs_dir=inputs, run_dir=run_dir, stage="body")

    assert summary["status"] == "blocked"
    assert [item["stage"] for item in summary["stages"]] == ["calibration", "tracking", "body"]
    assert summary["stages"][0]["status"] == "ran"
    assert summary["stages"][0]["real_model"] is False
    assert summary["stages"][1]["status"] == "ran"
    assert summary["stages"][1]["real_model"] is False
    assert summary["stages"][1]["source_mode"] == "precomputed_detections"
    assert summary["stages"][2]["status"] == "blocked"
    assert "no runner registered for stage: body" in summary["stages"][2]["notes"]
    assert validate_artifact_file("court_calibration", run_dir / "court_calibration.json")
    tracks = validate_artifact_file("tracks", run_dir / "tracks.json")
    assert isinstance(tracks, Tracks)
    assert not (run_dir / "smpl_motion.json").exists()


def test_pipeline_fails_closed_when_runner_outputs_invalid_artifact(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs" / "clip_001"
    run_dir = tmp_path / "runs" / "phase11" / "clip_001"
    _write_inputs(inputs)
    (inputs / "detections.json").write_text(json.dumps({"fps": 30.0, "frames": [{"detections": [{"bbox": [1, 2, 3, 4], "class": "person"}]}]}), encoding="utf-8")

    summary = run_pipeline(clip="clip_001", inputs_dir=inputs, run_dir=run_dir, stage="tracking")

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
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    payload = json.loads((run_dir / "pipeline_run.json").read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert payload["stages"][2]["stage"] == "body"
    assert payload["stages"][2]["status"] == "blocked"
    assert not (run_dir / "replay_scene.json").exists()
