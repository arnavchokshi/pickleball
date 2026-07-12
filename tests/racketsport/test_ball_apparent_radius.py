from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from threed.racketsport.ball_apparent_radius import (
    RadiusSkipReason,
    estimate_apparent_radius,
)


def _textured_frame() -> np.ndarray:
    yy, xx = np.mgrid[:128, :128]
    frame = np.empty((128, 128, 3), dtype=np.uint8)
    frame[..., 0] = np.clip(35 + xx // 4, 0, 255)
    frame[..., 1] = np.clip(45 + yy // 5, 0, 255)
    frame[..., 2] = np.clip(55 + (xx + yy) // 10, 0, 255)
    return frame


def _moving_ball_pair(radius: int) -> tuple[np.ndarray, np.ndarray]:
    previous = _textured_frame()
    current = _textured_frame()
    color = (20, 235, 245)
    cv2.circle(previous, (48, 64), radius, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(current, (64, 64), radius, color, -1, lineType=cv2.LINE_AA)
    return previous, current


@pytest.mark.parametrize("radius", [4, 7, 11])
def test_known_synthetic_radii(radius: int) -> None:
    previous, current = _moving_ball_pair(radius)
    result = estimate_apparent_radius(
        current,
        (64.0, 64.0),
        previous_frame=previous,
        previous_ball_center_xy=(48.0, 64.0),
    )

    assert result.status == "measured"
    assert result.radius_px == pytest.approx(radius, abs=2.0)
    assert result.confidence > 0.0
    assert result.provenance["heatmap_used"] is False


def test_deterministic_measurement_and_provenance() -> None:
    previous, current = _moving_ball_pair(7)
    kwargs = {
        "previous_frame": previous,
        "previous_ball_center_xy": (48.0, 64.0),
    }
    first = estimate_apparent_radius(current, (64.0, 64.0), **kwargs)
    second = estimate_apparent_radius(current, (64.0, 64.0), **kwargs)

    assert first.to_dict() == second.to_dict()
    assert first.method == "hough_circle_temporal_blob_gated_v1"
    assert first.provenance["radius_source"].startswith("current-frame grayscale gradient Hough")


def test_motion_blur_has_typed_skip_reason() -> None:
    frame = _textured_frame()
    cv2.ellipse(frame, (64, 64), (18, 4), 0, 0, 360, (20, 235, 245), -1)

    result = estimate_apparent_radius(frame, (64.0, 64.0))

    assert result.status == "skipped"
    assert result.radius_px is None
    assert result.skip_reason == RadiusSkipReason.MOTION_BLUR


def test_truncated_crop_abstains() -> None:
    result = estimate_apparent_radius(_textured_frame(), (4.0, 4.0))
    assert result.status == "skipped"
    assert result.skip_reason == RadiusSkipReason.CROP_TRUNCATED


def test_real_wolverine_frame_smoke() -> None:
    root = Path(__file__).resolve().parents[2]
    video = root / "eval_clips/ball/wolverine_mixed_0200_mid_steep_corner/source.mp4"
    track_path = root / "runs/lanes/ball_sizeobs_20260712/wolverine_offline/ball_track.json"
    if not video.is_file() or not track_path.is_file():
        pytest.skip("internal Wolverine smoke artifact is unavailable")
    frames = json.loads(track_path.read_text(encoding="utf-8"))["frames"]
    frame_index = 160
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index - 1)
    ok_previous, previous = cap.read()
    ok_current, current = cap.read()
    cap.release()
    assert ok_previous and ok_current

    result = estimate_apparent_radius(
        current,
        tuple(frames[frame_index]["xy"]),
        previous_frame=previous,
        previous_ball_center_xy=tuple(frames[frame_index - 1]["xy"]),
    )

    assert result.status == "measured"
    assert result.radius_px is not None and 2.0 <= result.radius_px <= 20.0
    assert result.center_xy_px is not None
    assert result.gates["center_refinement_px"] <= 20.0
