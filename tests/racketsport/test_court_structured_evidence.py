from __future__ import annotations

import json

import numpy as np
import pytest

from threed.racketsport.court_keypoint_net import ALL_PICKLEBALL_KEYPOINTS, PICKLEBALL_KEYPOINTS
from threed.racketsport.court_structured_evidence import (
    CANONICAL_FLOOR_KEYPOINT_NAMES,
    CourtEvidenceBundle,
    EVIDENCE_FLOOR_KEYPOINT_NAMES,
    NET_TOP_KEYPOINT_NAMES,
    build_court_evidence_bundle,
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


def test_v3_auxiliary_floor_channels_are_emitted_in_taxonomy_order() -> None:
    heatmaps: dict[str, np.ndarray] = {}
    visibility: dict[str, float] = {}
    for point in reversed(ALL_PICKLEBALL_KEYPOINTS):
        heatmap = np.full((5, 5), 0.001, dtype=np.float64)
        heatmap[1, 1] = 0.7
        heatmap[4, 4] = 0.2
        heatmaps[point.name] = heatmap
        visibility[point.name] = 0.9

    records = extract_court_structured_evidence(heatmaps, visibility)

    assert [row["keypoint_name"] for row in records] == list(EVIDENCE_FLOOR_KEYPOINT_NAMES)
    assert len(records) == 30
    assert records[0]["canonical_floor_index"] == 0
    assert records[-1]["canonical_floor_index"] is None
    assert {row["keypoint_name"] for row in records}.isdisjoint(NET_TOP_KEYPOINT_NAMES)


def test_dark_decoder_and_udp_coordinate_mapping_are_explicit() -> None:
    yy, xx = np.mgrid[:9, :9]
    heatmap = np.exp(-((xx - 3.35) ** 2 + (yy - 4.20) ** 2) / (2.0 * 0.9**2))
    heatmap[1, 7] += 0.2
    record = extract_court_structured_evidence(
        {"near_left_corner": heatmap},
        {"near_left_corner": 1.0},
        source_size=(81, 81),
        decoder="dark",
        coordinate_transform="udp",
    )[0]

    assert record["decoder"] == "dark"
    assert record["coordinate_transform"] == "udp"
    assert record["primary_peak"]["heatmap_xy"] == pytest.approx([3.35, 4.20], abs=0.08)
    assert record["primary_peak"]["source_xy"] == pytest.approx([33.5, 42.0], abs=0.8)


def test_evidence_bundle_validates_shapes_and_freezes_dense_arrays() -> None:
    distance = np.ones((12, 20), dtype=np.float64)
    surface = np.full((12, 20), 0.75, dtype=np.float64)
    bundle = build_court_evidence_bundle(
        [{"semantic": "near_left_corner", "xy": [1.0, 2.0], "confidence": 0.9}],
        image_size=(20, 12),
        line_distance_maps={"near_baseline": distance},
        surface_probability=surface,
        temporal_support=0.8,
    )

    assert isinstance(bundle, CourtEvidenceBundle)
    assert bundle.line_distance_maps["near_baseline"].flags.writeable is False
    assert bundle.surface_probability.flags.writeable is False
    distance[:] = 9.0
    assert float(bundle.line_distance_maps["near_baseline"][0, 0]) == 1.0
    with pytest.raises(ValueError, match="12x20"):
        build_court_evidence_bundle([], image_size=(20, 12), line_distance_maps={"bad": np.ones((2, 2))})
