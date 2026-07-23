from __future__ import annotations

import json

import numpy as np
import pytest

from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS
from threed.racketsport.court_structured_evidence import (
    CANONICAL_FLOOR_KEYPOINT_NAMES,
    NET_TOP_KEYPOINT_NAMES,
    extract_court_structured_evidence,
)


def _single_channel(
    heatmap: np.ndarray,
    *,
    visibility: float = 0.9,
    image_size: tuple[int, int] | None = None,
    source_size: tuple[int, int] | None = None,
) -> dict[str, object]:
    return extract_court_structured_evidence(
        {"near_left_corner": heatmap},
        {"near_left_corner": visibility},
        image_size=image_size,
        source_size=source_size,
    )[0]


def test_extracts_two_separated_subpixel_peaks_with_probabilities() -> None:
    heatmap = np.full((7, 7), 0.001, dtype=np.float64)
    heatmap[2, 2] = 0.40
    heatmap[2, 1] = 0.20
    heatmap[2, 3] = 0.30
    heatmap[1, 2] = 0.10
    heatmap[3, 2] = 0.25
    heatmap[5, 5] = 0.19
    heatmap[5, 4] = 0.08
    heatmap[5, 6] = 0.12
    heatmap[4, 5] = 0.07
    heatmap[6, 5] = 0.10

    record = _single_channel(heatmap)
    primary = record["primary_peak"]
    secondary = record["secondary_peak"]

    assert primary["heatmap_discrete_xy"] == [2, 2]
    assert secondary["heatmap_discrete_xy"] == [5, 5]
    assert primary["heatmap_xy"] != [2.0, 2.0]
    assert secondary["heatmap_xy"] != [5.0, 5.0]
    assert primary["probability"] > secondary["probability"] > 0.0
    assert record["peak_separation_heatmap_px"] >= 2.0
    assert record["peak_margin"] == pytest.approx(
        primary["probability"] - secondary["probability"]
    )


def test_entropy_and_calibrated_confidence_order_sharp_above_diffuse() -> None:
    sharp = np.full((5, 5), 0.001, dtype=np.float64)
    sharp[1, 1] = 0.95
    sharp[4, 4] = 0.02

    diffuse = np.ones((5, 5), dtype=np.float64)

    sharp_record = _single_channel(sharp, visibility=0.8)
    diffuse_record = _single_channel(diffuse, visibility=0.8)

    assert sharp_record["normalized_entropy"] < diffuse_record["normalized_entropy"]
    assert sharp_record["peak_margin"] > diffuse_record["peak_margin"]
    assert sharp_record["raw_confidence"] > diffuse_record["raw_confidence"]
    assert sharp_record["calibrated_raw_confidence"] > diffuse_record["calibrated_raw_confidence"]
    assert diffuse_record["normalized_entropy"] == pytest.approx(1.0)
    assert diffuse_record["calibrated_raw_confidence"] == pytest.approx(0.0)


def test_scales_peaks_and_covariance_to_image_and_source_pixels() -> None:
    heatmap = np.full((3, 4), 0.001, dtype=np.float64)
    heatmap[1, 2] = 0.90
    heatmap[1, 1] = 0.10
    heatmap[1, 3] = 0.10
    heatmap[0, 2] = 0.10
    heatmap[2, 2] = 0.10
    heatmap[2, 0] = 0.20

    image_record = _single_channel(heatmap, image_size=(40, 30))
    source_record = _single_channel(
        heatmap,
        image_size=(40, 30),
        source_size=(80, 60),
    )

    assert image_record["coordinate_space"] == "image_pixels"
    assert image_record["observation_xy"] == pytest.approx([20.0, 10.0])
    assert source_record["coordinate_space"] == "source_pixels"
    assert source_record["primary_peak"]["image_xy"] == pytest.approx([20.0, 10.0])
    assert source_record["primary_peak"]["source_xy"] == pytest.approx([40.0, 20.0])
    assert source_record["observation_xy"] == pytest.approx([40.0, 20.0])

    image_covariance = np.asarray(image_record["covariance_px2"], dtype=np.float64)
    source_covariance = np.asarray(source_record["covariance_px2"], dtype=np.float64)
    assert image_covariance.shape == (2, 2)
    assert np.all(np.linalg.eigvalsh(image_covariance) > 0.0)
    assert source_covariance == pytest.approx(image_covariance * 4.0)


def test_output_is_deterministic_json_serializable_and_canonical_ordered() -> None:
    heatmaps: dict[str, np.ndarray] = {}
    visibility: dict[str, float] = {}
    # Deliberately insert channels in reverse taxonomy order. Output ordering must not depend on
    # mapping insertion order or tied secondary-peak traversal.
    for index, point in reversed(list(enumerate(PICKLEBALL_KEYPOINTS))):
        heatmap = np.full((5, 5), 0.01, dtype=np.float64)
        heatmap[1, 1] = 0.60 + index * 0.001
        heatmap[4, 4] = 0.20
        heatmaps[point.name] = heatmap
        visibility[point.name] = 0.75

    first = extract_court_structured_evidence(heatmaps, visibility)
    second = extract_court_structured_evidence(heatmaps, visibility)

    assert first == second
    assert [row["keypoint_name"] for row in first] == list(CANONICAL_FLOOR_KEYPOINT_NAMES)
    assert json.loads(json.dumps(first, sort_keys=True)) == first


def test_all_three_net_top_channels_are_structurally_excluded() -> None:
    heatmaps: dict[str, np.ndarray] = {}
    visibility: dict[str, float] = {}
    for point in PICKLEBALL_KEYPOINTS:
        heatmap = np.full((5, 5), 0.001, dtype=np.float64)
        # Give excluded net channels stronger evidence than every floor channel. They must still
        # never appear in planar structured observations.
        heatmap[1, 1] = 0.99 if point.name in NET_TOP_KEYPOINT_NAMES else 0.50
        heatmap[4, 4] = 0.20
        heatmaps[point.name] = heatmap
        visibility[point.name] = 1.0

    records = extract_court_structured_evidence(heatmaps, visibility)
    observed = {row["keypoint_name"] for row in records}

    assert len(records) == 12
    assert observed == set(CANONICAL_FLOOR_KEYPOINT_NAMES)
    assert observed.isdisjoint(NET_TOP_KEYPOINT_NAMES)
    assert all(row["provenance"]["net_top_channels_excluded"] is True for row in records)
