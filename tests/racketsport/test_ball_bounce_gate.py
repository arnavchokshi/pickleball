from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.ball_bounce_gate import build_ball_bounce_gate_report


def _write_test_video(path: Path, *, size: str = "1920x1080", fps: int = 60, audio: bool = True) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required to synthesize bounce-gate test video")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size={size}:rate={fps}:duration=0.2",
    ]
    if audio:
        command.extend(["-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000:duration=0.2"])
    command.extend(["-pix_fmt", "yuv420p", "-shortest", str(path)])
    subprocess.run(command, check=True)


def _ball_track_payload(*, bounces: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "schema_version": 1,
        "fps": 60.0,
        "source": "fused",
        "frames": [
            {"t": 0.966667, "xy": [118.0, 204.0], "conf": 0.71, "visible": True},
            {"t": 1.000000, "xy": [120.0, 210.0], "conf": 0.83, "visible": True},
            {"t": 1.033333, "xy": [124.0, 205.0], "conf": 0.76, "visible": True},
        ],
        "bounces": bounces
        if bounces is not None
        else [
            {
                "t": 1.0,
                "frame": 60,
                "world_xy": [1.2, 2.4],
                "contact_xy_img": [120.0, 210.0],
                "p_bounce": 0.82,
                "audio_delta_ms": 20.0,
                "source": "catboost_audio_fusion_v1",
            }
        ],
    }


def _classifier_payload(
    *,
    model_path: str | None = None,
    model_sha256: str | None = None,
    input_ball_track_path: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_bounce_classifier_output",
        "model_family": "catboost",
        "trained_on_real_labels": True,
        "probability_threshold": 0.5,
        "candidate_count": 1,
        "accepted_bounces": [
            {
                "t": 1.0,
                "frame": 60,
                "p_bounce": 0.82,
                "world_xy": [1.2, 2.4],
                "contact_xy_img": [120.0, 210.0],
                "source": "catboost_audio_fusion_v1",
            }
        ],
    }
    if model_path is not None:
        payload.update(
            {
                "model_path": model_path,
                "model_sha256": model_sha256,
                "feature_window_frames": 20,
                "training_label_count": 6,
                "validation_label_count": 2,
                "training_command": "python scripts/racketsport/train_ball_bounce_classifier.py --real-labels ...",
                "inference_command": "python scripts/racketsport/run_ball_bounce_classifier.py --ball-track ...",
                "input_ball_track_path": input_ball_track_path,
            }
        )
    return payload


def _audio_onsets_payload(*, onsets: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_audio_onsets",
        "trusted_for_contact": False,
        "summary": {"onset_count": len(onsets or [])},
        "onsets": onsets
        if onsets is not None
        else [{"time_s": 1.02, "score": 0.91, "source": "audio_energy_onset"}],
    }


def _reviewed_bounces_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_reviewed_ball_bounces",
        "fps": 60.0,
        "bounces": [{"frame": 60, "t": 1.0}],
    }


def test_ball_bounce_gate_passes_classifier_audio_review_and_projection(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    track = tmp_path / "ball_track.json"
    classifier = tmp_path / "bounce_classifier.json"
    audio = tmp_path / "audio_onsets.json"
    reviewed = tmp_path / "reviewed_bounces.json"
    model = tmp_path / "bounce_model.pkl"
    _write_test_video(video)
    model.write_bytes(b"real gbm model bytes for bounce gate fixture")
    track.write_text(json.dumps(_ball_track_payload()), encoding="utf-8")
    classifier.write_text(
        json.dumps(
            _classifier_payload(
                model_path=model.name,
                model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                input_ball_track_path=str(track),
            )
        ),
        encoding="utf-8",
    )
    audio.write_text(json.dumps(_audio_onsets_payload()), encoding="utf-8")
    reviewed.write_text(json.dumps(_reviewed_bounces_payload()), encoding="utf-8")

    report = build_ball_bounce_gate_report(
        ball_track_path=track,
        video_path=video,
        classifier_path=classifier,
        audio_onsets_path=audio,
        reviewed_bounces_path=reviewed,
    )

    assert report["status"] == "TESTED-ON-REAL-DATA"
    assert report["milestone"] == "M4 Bounce"
    assert report["gate_result"] == "pass"
    assert report["blocked_reason"] is None
    assert report["bounce_count"] == 1
    assert report["audio_alignment"]["max_abs_delta_ms"] == pytest.approx(20.0)
    assert report["review_timing"]["max_abs_delta_frames"] == pytest.approx(0.0)
    assert report["violations"] == []
    assert report["not_ground_truth"] is True


def test_ball_bounce_gate_fails_closed_when_classifier_audio_and_bounces_are_missing(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    track = tmp_path / "ball_track.json"
    audio = tmp_path / "audio_onsets.json"
    _write_test_video(video, audio=False)
    track.write_text(json.dumps(_ball_track_payload(bounces=[])), encoding="utf-8")
    audio.write_text(json.dumps(_audio_onsets_payload(onsets=[])), encoding="utf-8")

    report = build_ball_bounce_gate_report(
        ball_track_path=track,
        video_path=video,
        classifier_path=tmp_path / "missing_classifier.json",
        audio_onsets_path=audio,
        reviewed_bounces_path=tmp_path / "missing_reviewed_bounces.json",
    )

    assert report["status"] == "TESTED-ON-REAL-DATA"
    assert report["gate_result"] == "fail"
    assert report["blocked_reason"] == "ball_bounce_gate_failed"
    assert set(report["violations"]) >= {
        "ball_track_has_no_bounces",
        "missing_bounce_classifier_output",
        "missing_reviewed_bounce_labels",
        "audio_onsets_empty",
        "video_audio_missing",
    }


def test_ball_bounce_gate_rejects_classifier_output_without_run_provenance(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    track = tmp_path / "ball_track.json"
    classifier = tmp_path / "bounce_classifier.json"
    audio = tmp_path / "audio_onsets.json"
    reviewed = tmp_path / "reviewed_bounces.json"
    _write_test_video(video)
    track.write_text(json.dumps(_ball_track_payload()), encoding="utf-8")
    classifier.write_text(json.dumps(_classifier_payload()), encoding="utf-8")
    audio.write_text(json.dumps(_audio_onsets_payload()), encoding="utf-8")
    reviewed.write_text(json.dumps(_reviewed_bounces_payload()), encoding="utf-8")

    report = build_ball_bounce_gate_report(
        ball_track_path=track,
        video_path=video,
        classifier_path=classifier,
        audio_onsets_path=audio,
        reviewed_bounces_path=reviewed,
    )

    assert report["gate_result"] == "fail"
    assert set(report["violations"]) >= {
        "bounce_classifier_model_path_missing",
        "bounce_classifier_model_sha256_missing",
        "bounce_classifier_feature_window_frames_missing",
        "bounce_classifier_training_label_count_missing",
        "bounce_classifier_training_command_missing",
        "bounce_classifier_inference_command_missing",
        "bounce_classifier_input_track_missing",
    }


def test_ball_bounce_gate_cli_writes_failed_report(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    track = tmp_path / "ball_track.json"
    out = tmp_path / "m4_report.json"
    _write_test_video(video, audio=False)
    track.write_text(json.dumps(_ball_track_payload(bounces=[])), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/validate_ball_bounce.py",
            "--ball-track",
            str(track),
            "--video",
            str(video),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert json.loads(completed.stdout)["gate_result"] == "fail"
    assert json.loads(out.read_text(encoding="utf-8"))["blocked_reason"] == "ball_bounce_gate_failed"
