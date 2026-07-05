from __future__ import annotations

import numpy as np

from threed.racketsport.court_line_bank import build_line_bank_from_image, normalize_hough_lines_p


def test_normalize_hough_lines_p_accepts_opencv_3d_shape() -> None:
    raw = np.array([[[1, 2, 3, 4]], [[5, 6, 7, 8]]], dtype=np.int32)

    assert normalize_hough_lines_p(raw) == [
        (1.0, 2.0, 3.0, 4.0),
        (5.0, 6.0, 7.0, 8.0),
    ]


def test_normalize_hough_lines_p_accepts_opencv_2d_shape() -> None:
    raw = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)

    assert normalize_hough_lines_p(raw) == [
        (1.0, 2.0, 3.0, 4.0),
        (5.0, 6.0, 7.0, 8.0),
    ]


def test_normalize_hough_lines_p_accepts_none() -> None:
    assert normalize_hough_lines_p(None) == []


def test_line_bank_recovers_synthetic_court_lines() -> None:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[40:43, 20:300] = 255
    image[120:123, 20:300] = 255
    image[200:203, 20:300] = 255
    image[40:200, 20:23] = 255
    image[40:200, 300:303] = 255

    bank = build_line_bank_from_image(image, max_segments=32)

    assert len(bank["segments"]) >= 4
    assert bank["metadata"]["detectors"]["hough"]["available"] is True
