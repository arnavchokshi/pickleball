from __future__ import annotations

import math

import pytest

from threed.racketsport.court_auto_evidence import detect_image_line_segments
from threed.racketsport.court_keypoint_net import keypoint_labels_from_court_corners
from threed.racketsport.court_line_bank import (
    LegacyPaintEvidenceSample,
    detect_paint_centerline_candidates,
    refine_legacy_paint_samples,
    refine_legacy_paint_segments,
)
from threed.racketsport.court_line_keypoints import detect_court_keypoints_from_image


cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


def _bright_band_image(
    *,
    height: int = 220,
    width: int = 420,
    center_y: float = 100.35,
    band_width_px: float = 8.0,
    edge_softness_px: float = 0.7,
) -> object:
    y = np.arange(height, dtype=np.float32)[:, None]
    left = center_y - band_width_px / 2.0
    right = center_y + band_width_px / 2.0
    band = 0.5 * (np.tanh((y - left) / edge_softness_px) - np.tanh((y - right) / edge_softness_px))
    gray = np.broadcast_to(35.0 + 205.0 * band, (height, width)).astype(np.uint8).copy()
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _horizontal_center(candidate: object) -> float:
    return sum(point[1] for point in candidate.endpoints) / 2.0


def _closest_horizontal(candidates: list[object], expected_y: float) -> object:
    horizontal = [candidate for candidate in candidates if abs(candidate.angle_deg) <= 3.0]
    assert horizontal
    return min(horizontal, key=lambda candidate: abs(_horizontal_center(candidate) - expected_y))


def test_clean_band_center_is_subpixel_and_covariance_is_normal_only() -> None:
    expected_y = 100.35
    image = _bright_band_image(center_y=expected_y)

    candidate = _closest_horizontal(detect_paint_centerline_candidates(image), expected_y)

    assert abs(_horizontal_center(candidate) - expected_y) < 0.3
    assert candidate.band_width_px == pytest.approx(8.0, abs=0.4)
    assert candidate.polarity == "bright_band_on_darker_sides"
    assert candidate.support_length_px > 390.0
    assert candidate.family_hint == "cross_court_candidate"
    for sample in candidate.sampled_points:
        covariance = np.asarray(sample.normal_covariance_px2)
        eigenvalues = np.linalg.eigvalsh(covariance)
        assert eigenvalues[0] >= -1e-12
        assert eigenvalues[1] > 0.0
        tangent = np.asarray([sample.normal[1], -sample.normal[0]])
        assert float(tangent @ covariance @ tangent) == pytest.approx(0.0, abs=1e-10)
    assert candidate.as_dict()["sampled_points"][0]["normal_variance_px2"] > 0.0


def test_raw_arm_stays_subpixel_under_blur_and_broad_shadow_gradient() -> None:
    expected_y = 108.65
    image = _bright_band_image(center_y=expected_y, edge_softness_px=1.2)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    x = np.arange(gray.shape[1], dtype=np.float32)[None, :]
    illumination = 0.58 + 0.42 * x / float(gray.shape[1] - 1)
    illumination[:, 145:285] *= 0.72
    gray = cv2.GaussianBlur(np.clip(gray * illumination, 0, 255).astype(np.uint8), (0, 0), 1.7)
    shadowed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    candidate = _closest_horizontal(
        detect_paint_centerline_candidates(shadowed, preprocessing="raw"),
        expected_y,
    )

    assert abs(_horizontal_center(candidate) - expected_y) < 0.65
    assert candidate.preprocessing == "raw"
    assert candidate.coverage_fraction < 1.01


@pytest.mark.parametrize("provider", ["classical_paired_edges", "opencv_lsd_paired_edges"])
def test_dual_parallel_thin_distractors_do_not_replace_the_paint_band(provider: str) -> None:
    expected_y = 130.35
    image = _bright_band_image(center_y=expected_y)
    cv2.line(image, (0, 58), (419, 58), (245, 245, 245), 1, cv2.LINE_AA)
    cv2.line(image, (0, 70), (419, 70), (245, 245, 245), 1, cv2.LINE_AA)
    pixels_per_metre = 8.0 / 0.0508
    seed = {
        "homography": [
            [pixels_per_metre, 0.0, 0.0],
            [0.0, pixels_per_metre, 0.0],
            [0.0, 0.0, 1.0],
        ]
    }

    candidates = detect_paint_centerline_candidates(image, seed_calibration=seed, provider=provider)
    candidate = _closest_horizontal(candidates, expected_y)

    assert abs(_horizontal_center(candidate) - expected_y) < 0.3
    assert all(abs(_horizontal_center(row) - 64.0) > 3.0 for row in candidates if abs(row.angle_deg) <= 3.0)


def test_polarity_inversion_is_rejected() -> None:
    image = _bright_band_image(center_y=100.0)
    inverted = 255 - image

    candidates = detect_paint_centerline_candidates(inverted)

    assert not [candidate for candidate in candidates if abs(candidate.angle_deg) <= 3.0]


def test_occlusion_gap_loses_coverage_instead_of_spanning_the_gap() -> None:
    full = _bright_band_image(center_y=100.0)
    occluded = full.copy()
    occluded[:, 170:270] = (35, 35, 35)

    full_candidate = _closest_horizontal(detect_paint_centerline_candidates(full), 100.0)
    occluded_candidates = [
        candidate
        for candidate in detect_paint_centerline_candidates(occluded)
        if abs(candidate.angle_deg) <= 3.0
    ]

    assert occluded_candidates
    assert max(candidate.support_length_px for candidate in occluded_candidates) < full_candidate.support_length_px - 70.0
    assert all(
        not (min(point[0] for point in candidate.endpoints) < 170.0 and max(point[0] for point in candidate.endpoints) > 270.0)
        for candidate in occluded_candidates
    )


def test_opencv_lsd_arm_uses_the_same_paired_edge_center() -> None:
    expected_y = 96.4
    image = _bright_band_image(center_y=expected_y)

    candidate = _closest_horizontal(
        detect_paint_centerline_candidates(image, provider="opencv_lsd_paired_edges"),
        expected_y,
    )

    assert abs(_horizontal_center(candidate) - expected_y) < 0.3
    assert candidate.provider == "opencv_lsd_paired_edges"


def test_auto_evidence_segment_provider_is_explicit_and_default_is_unchanged() -> None:
    expected_y = 104.3
    image = _bright_band_image(center_y=expected_y)

    implicit_legacy = detect_image_line_segments(image)
    explicit_legacy = detect_image_line_segments(image, evidence_provider="legacy_hough")
    centerlines = detect_image_line_segments(image, evidence_provider="paint_centerline")

    assert implicit_legacy == explicit_legacy
    assert min(abs((segment[0][1] + segment[1][1]) / 2.0 - expected_y) for segment in centerlines) < 0.3


def test_keypoint_detector_consumes_opt_in_centerlines() -> None:
    labels = keypoint_labels_from_court_corners(
        {
            "near_left": [120.0, 360.0],
            "near_right": [620.0, 300.0],
            "far_right": [380.0, 90.0],
            "far_left": [80.0, 130.0],
        }
    )
    image = np.zeros((420, 720, 3), dtype=np.uint8)
    image[:, :] = (36, 42, 45)
    for start, end in (
        ("near_left_corner", "near_right_corner"),
        ("far_left_corner", "far_right_corner"),
        ("near_left_corner", "far_left_corner"),
        ("near_right_corner", "far_right_corner"),
        ("near_nvz_left", "near_nvz_right"),
        ("far_nvz_left", "far_nvz_right"),
        ("net_left_sideline", "net_right_sideline"),
        ("near_baseline_center", "far_baseline_center"),
    ):
        cv2.line(
            image,
            tuple(int(round(value)) for value in labels[start]),
            tuple(int(round(value)) for value in labels[end]),
            (245, 245, 245),
            8,
            cv2.LINE_AA,
        )

    predictions = detect_court_keypoints_from_image(image, line_evidence_provider="paint_centerline")
    errors = [math.dist(predictions.keypoints[name]["xy"], expected) for name, expected in labels.items()]

    assert float(np.median(errors)) < 1.0
    assert max(errors) < 2.0


def test_hybrid_refines_confirmed_band_and_preserves_declined_samples_with_covariance() -> None:
    center_y = 100.35
    image = _bright_band_image(center_y=center_y)
    samples = [
        LegacyPaintEvidenceSample(xy=(100.0, center_y - 3.8), normal=(0.0, 1.0), source_id="paint"),
        LegacyPaintEvidenceSample(xy=(100.0, 30.0), normal=(0.0, 1.0), source_id="blank"),
    ]

    refined = refine_legacy_paint_samples(image, samples)

    assert len(refined) == len(samples)
    assert refined[0].provenance == "band_refined"
    assert refined[0].xy[1] == pytest.approx(center_y, abs=0.3)
    assert refined[0].normal_variance_px2 < refined[1].normal_variance_px2
    assert refined[0].as_dict()["source_id"] == "paint"
    assert refined[1].provenance == "legacy_raw"
    assert refined[1].xy == samples[1].xy
    assert refined[1].legacy_xy == samples[1].xy


def test_hybrid_dual_line_distractor_falls_back_without_dropping_visibility() -> None:
    center_y = 130.35
    image = _bright_band_image(center_y=center_y)
    cv2.line(image, (0, 58), (419, 58), (245, 245, 245), 1, cv2.LINE_AA)
    cv2.line(image, (0, 70), (419, 70), (245, 245, 245), 1, cv2.LINE_AA)
    pixels_per_metre = 8.0 / 0.0508
    seed = {
        "homography": [
            [pixels_per_metre, 0.0, 0.0],
            [0.0, pixels_per_metre, 0.0],
            [0.0, 0.0, 1.0],
        ]
    }
    legacy = [
        LegacyPaintEvidenceSample(xy=(200.0, center_y - 3.8), normal=(0.0, 1.0), source_id="paint"),
        LegacyPaintEvidenceSample(xy=(200.0, 58.0), normal=(0.0, 1.0), source_id="thin_distractor"),
    ]

    hybrid = refine_legacy_paint_samples(image, legacy, seed_calibration=seed)

    assert len(hybrid) == 2
    assert hybrid[0].provenance == "band_refined"
    assert hybrid[0].xy[1] == pytest.approx(center_y, abs=0.3)
    assert hybrid[1].provenance == "legacy_raw"
    assert hybrid[1].xy == legacy[1].xy


def test_hybrid_segments_and_auto_provider_preserve_legacy_cardinality() -> None:
    image = _bright_band_image(center_y=100.35)
    legacy = detect_image_line_segments(image, evidence_provider="legacy_hough")

    hybrid_segments = refine_legacy_paint_segments(image, legacy)
    hybrid_auto = detect_image_line_segments(image, evidence_provider="hybrid_paint_refinement")

    assert len(hybrid_segments) == len(legacy)
    assert len(hybrid_auto) == len(legacy)
    assert all(len(segment.sampled_points) >= 2 for segment in hybrid_segments)
    assert all(
        len(segment.sampled_points)
        == sum(sample.provenance in {"band_refined", "legacy_raw"} for sample in segment.sampled_points)
        for segment in hybrid_segments
    )


def test_hybrid_polarity_inversion_keeps_original_visibility_as_legacy_raw() -> None:
    image = 255 - _bright_band_image(center_y=100.0)
    legacy = [LegacyPaintEvidenceSample(xy=(120.0, 100.0), normal=(0.0, 1.0), source_id="dark_band")]

    [hybrid] = refine_legacy_paint_samples(image, legacy)

    assert hybrid.provenance == "legacy_raw"
    assert hybrid.xy == legacy[0].xy
