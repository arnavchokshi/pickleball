from __future__ import annotations

import math
from pathlib import Path

import pytest

from threed.racketsport.event_head.datasets import (
    DatasetFormatError,
    WindowSpec,
    dense_class_counts,
    sqrt_frequency_class_weights,
)


def test_sqrt_frequency_weights_use_exact_background_normalized_formula() -> None:
    counts = (100.0, 25.0, 4.0)

    weights = sqrt_frequency_class_weights(counts)

    assert weights == (
        1.0,
        math.sqrt(counts[0] / counts[1]),
        math.sqrt(counts[0] / counts[2]),
    )
    assert weights == (1.0, 2.0, 5.0)


@pytest.mark.parametrize(
    "counts",
    [
        (100.0, 25.0),
        (100.0, 0.0, 4.0),
        (100.0, -1.0, 4.0),
        (100.0, float("nan"), 4.0),
        (100.0, float("inf"), 4.0),
    ],
    ids=("missing-class", "zero", "negative", "nan", "infinite"),
)
def test_sqrt_frequency_weights_reject_missing_zero_or_nonfinite_counts(
    counts: tuple[float, ...],
) -> None:
    with pytest.raises(DatasetFormatError):
        sqrt_frequency_class_weights(counts)


def _window(
    *, class_id: int, unknown_frame: int,
) -> WindowSpec:
    unknown_mask = [False] * 5
    unknown_mask[unknown_frame] = True
    return WindowSpec(
        video_path=Path("unused-by-dense-counts.avi"),
        start_frame=0,
        num_frames=5,
        fps=30.0,
        events=((2, class_id),),
        validity_mask=(True, True, True),
        source="fixture",
        license_posture="RD_ONLY",
        unknown_frame_mask=tuple(unknown_mask),
    )


def test_dense_class_counts_apply_soft_dilation_without_counting_unknown_neighbors() -> None:
    # Each five-frame window has one UNKNOWN neighbor beside its event center:
    # HIT's left neighbor and BOUNCE's right neighbor. With soft weight 0.5,
    # only the known neighbor gains 0.5 positive mass. Per window this yields
    # background=2.5 and its event class=1.5.
    windows = (
        _window(class_id=1, unknown_frame=1),
        _window(class_id=2, unknown_frame=3),
    )

    undilated = dense_class_counts(
        windows,
        label_dilation_frames=0,
        neighbor_positive_weight=0.5,
    )
    dilated = dense_class_counts(
        windows,
        label_dilation_frames=1,
        neighbor_positive_weight=0.5,
    )

    assert undilated == (6.0, 1.0, 1.0)
    assert dilated == (5.0, 1.5, 1.5)
    assert sum(undilated) == sum(dilated) == 8.0
