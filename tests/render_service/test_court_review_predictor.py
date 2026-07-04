import json
import subprocess
from pathlib import Path

from server import court_review


def test_public_court_predictor_defaults_to_template_seed(monkeypatch) -> None:
    def fake_run(*args, **kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args[0],
            0,
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "width": 1000,
                            "height": 600,
                            "avg_frame_rate": "60/1",
                            "duration": "10.0",
                            "nb_frames": "600",
                        }
                    ]
                }
            ),
            stderr="",
        )

    monkeypatch.delenv("PICKLEBALL_COURT_PREDICTOR_MODE", raising=False)
    monkeypatch.setattr(court_review.subprocess, "run", fake_run)

    prediction = court_review.predict_court_layout_from_video(video_path=Path("/tmp/clip.mp4"), clip="clip_1")

    assert prediction["prediction_source"] == "template_projection_seed:ffprobe_metadata"
    assert prediction["verified"] is False
    assert prediction["not_cal3_verified"] is True
    assert prediction["image_size"] == [1000, 600]
    assert len(prediction["points"]) == 15
    assert "template_seed_not_automatic_detection" in prediction["promotion_blockers"]


def test_detector_mode_falls_back_to_template_seed_when_detector_fails(monkeypatch) -> None:
    def fake_run(*args, **kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args[0], 1, stdout="", stderr="ffprobe failed")

    def fake_detector(**kwargs):
        raise RuntimeError("detector unavailable")

    monkeypatch.setenv("PICKLEBALL_COURT_PREDICTOR_MODE", "detector")
    monkeypatch.setattr(court_review.subprocess, "run", fake_run)
    monkeypatch.setattr(court_review, "_predict_with_detector", fake_detector)

    prediction = court_review.predict_court_layout_from_video(video_path=Path("/tmp/clip.mp4"), clip="clip_1")

    assert prediction["prediction_source"] == "template_projection_seed:ffprobe_metadata"
    assert prediction["image_size"] == [1280, 720]
    assert "court_detector_v2_failed_fell_back_to_template:RuntimeError" in prediction["warnings"]
