from __future__ import annotations

import pytest

from scripts.racketsport.run_totnet_ball import _normalize_max_frames, _runtime_metrics


def test_runtime_metrics_report_effective_fps_and_realtime_factor() -> None:
    metrics = _runtime_metrics(
        seconds=2.0,
        decoded_frame_count=60,
        source_fps=30.0,
        max_frames=60,
        timing_breakdown={
            "decode_preprocess_seconds": 0.7,
            "host_to_device_seconds": 0.1,
            "inference_seconds": 1.0,
            "postprocess_seconds": 0.2,
        },
    )

    assert metrics["seconds"] == pytest.approx(2.0)
    assert metrics["decoded_frame_count"] == 60
    assert metrics["effective_fps"] == pytest.approx(30.0)
    assert metrics["video_seconds_processed"] == pytest.approx(2.0)
    assert metrics["realtime_factor"] == pytest.approx(1.0)
    assert metrics["max_frames"] == 60
    assert metrics["timing_breakdown"]["decode_preprocess_seconds"] == pytest.approx(0.7)
    assert metrics["timing_breakdown"]["inference_seconds"] == pytest.approx(1.0)
    assert metrics["timing_breakdown"]["accounted_seconds"] == pytest.approx(2.0)


def test_normalize_max_frames_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="max_frames must be positive"):
        _normalize_max_frames(0)
