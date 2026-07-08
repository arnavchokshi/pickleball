from __future__ import annotations

import itertools
import math
import time

import numpy as np
import pytest

from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS
from threed.racketsport.court_synth_scenes import (
    KEYPOINT_VIS_CLASSES,
    LINE_FAMILY_CLASSES,
    SCENARIO_NAMES,
    SURFACE_CLASSES,
)
from threed.racketsport.court_synth_stream import iter_synthetic_court_samples, reproject_canonical_keypoints

CANONICAL_NAMES = [point.name for point in PICKLEBALL_KEYPOINTS]


def _take(config, seed, count):
    return list(itertools.islice(iter_synthetic_court_samples(config, seed=seed), count))


def test_contract_keys_shapes_and_dtypes_hold_for_every_scenario() -> None:
    for scenario in SCENARIO_NAMES:
        samples = _take({"scenarios": [scenario], "image_size": [640, 360]}, seed=1, count=1)
        sample = samples[0]
        assert set(sample) == {
            "image_bgr",
            "keypoints_xy",
            "keypoints_vis",
            "line_family_mask",
            "surface_mask",
            "meta",
        }
        height, width = 360, 640
        assert sample["image_bgr"].shape == (height, width, 3)
        assert sample["image_bgr"].dtype == np.uint8
        assert sample["keypoints_xy"].shape == (15, 2)
        assert sample["keypoints_xy"].dtype == np.float32
        assert sample["keypoints_vis"].shape == (15,)
        assert set(sample["keypoints_vis"].tolist()) <= {0, 1, 2}
        assert sample["line_family_mask"].shape == (height, width)
        assert sample["line_family_mask"].dtype == np.uint8
        assert set(np.unique(sample["line_family_mask"]).tolist()) <= set(LINE_FAMILY_CLASSES.values())
        assert sample["surface_mask"].shape == (height, width)
        assert sample["surface_mask"].dtype == np.uint8
        assert set(np.unique(sample["surface_mask"]).tolist()) <= set(SURFACE_CLASSES.values())
        meta = sample["meta"]
        for key in ("homography", "distortion", "scenario", "image_size"):
            assert key in meta
        assert meta["scenario"] == scenario
        assert meta["image_size"] == [width, height]


def test_deterministic_for_same_config_and_seed() -> None:
    config = {"count": 5, "scenario_weights": {"dedicated_indoor": 1.0, "tennis_overlay": 1.0}}
    samples_a = _take(config, seed=42, count=5)
    samples_b = _take(config, seed=42, count=5)
    for a, b in zip(samples_a, samples_b, strict=True):
        assert np.array_equal(a["image_bgr"], b["image_bgr"])
        assert np.array_equal(a["keypoints_xy"], b["keypoints_xy"])
        assert np.array_equal(a["keypoints_vis"], b["keypoints_vis"])
        assert np.array_equal(a["line_family_mask"], b["line_family_mask"])
        assert np.array_equal(a["surface_mask"], b["surface_mask"])
        assert a["meta"] == b["meta"]


def test_different_seed_changes_output() -> None:
    config = {"count": 3}
    samples_a = _take(config, seed=1, count=3)
    samples_b = _take(config, seed=2, count=3)
    assert any(
        not np.array_equal(a["image_bgr"], b["image_bgr"]) for a, b in zip(samples_a, samples_b, strict=True)
    )


def test_self_consistency_below_half_pixel_across_all_scenarios_and_distortion() -> None:
    """The core CAL-SYNTH acceptance bar: reprojecting the court template through the emitted
    homography+distortion must reproduce every emitted keypoint (including elevated net keypoints
    and nonzero-distortion portrait samples) to < 0.5px."""

    max_seen = 0.0
    for scenario in SCENARIO_NAMES:
        samples = _take({"scenarios": [scenario], "count": 6}, seed=7, count=6)
        for sample in samples:
            reprojected = reproject_canonical_keypoints(sample["meta"])
            for idx, name in enumerate(CANONICAL_NAMES):
                rx, ry = reprojected[name]
                ex, ey = sample["keypoints_xy"][idx]
                err = math.hypot(rx - ex, ry - ey)
                max_seen = max(max_seen, err)
                assert err < 0.5, (scenario, name, err)
    assert max_seen < 0.5


def test_net_keypoints_are_elevated_and_self_consistent() -> None:
    sample = _take({"scenarios": ["dedicated_indoor"]}, seed=3, count=1)[0]
    meta = sample["meta"]
    net_names = [name for name in CANONICAL_NAMES if name.startswith("net_")]
    assert net_names
    for name in net_names:
        assert meta["keypoint_world_xyz_m"][name][2] > 0.0
    reprojected = reproject_canonical_keypoints(meta)
    for idx, name in enumerate(CANONICAL_NAMES):
        if not name.startswith("net_"):
            continue
        rx, ry = reprojected[name]
        ex, ey = sample["keypoints_xy"][idx]
        assert math.hypot(rx - ex, ry - ey) < 0.5


def test_tennis_overlay_emits_both_line_families_and_net() -> None:
    samples = _take({"scenarios": ["tennis_overlay"], "count": 8}, seed=11, count=8)
    seen_classes = set()
    for sample in samples:
        seen_classes |= set(np.unique(sample["line_family_mask"]).tolist())
    assert LINE_FAMILY_CLASSES["pickleball_line"] in seen_classes
    assert LINE_FAMILY_CLASSES["tennis_line"] in seen_classes
    assert LINE_FAMILY_CLASSES["net"] in seen_classes


def test_portrait_phone_exercises_off_frame_visibility() -> None:
    samples = _take({"scenarios": ["portrait_phone"], "count": 12}, seed=5, count=12)
    vis_values = {int(v) for sample in samples for v in sample["keypoints_vis"].tolist()}
    assert KEYPOINT_VIS_CLASSES["off_frame"] in vis_values


def test_portable_net_clutter_occasionally_occludes_a_net_keypoint() -> None:
    samples = _take({"scenarios": ["portable_net_clutter"], "count": 20}, seed=9, count=20)
    occluded_net_point_seen = False
    for sample in samples:
        for idx, name in enumerate(CANONICAL_NAMES):
            if name.startswith("net_") and sample["keypoints_vis"][idx] == KEYPOINT_VIS_CLASSES["occluded"]:
                occluded_net_point_seen = True
    assert occluded_net_point_seen


def test_adjacent_multi_court_instances_span_two_to_four_courts() -> None:
    samples = _take({"scenarios": ["adjacent_multi_court"], "count": 10}, seed=13, count=10)
    for sample in samples:
        instances = sample["meta"]["court_instances"]
        assert 2 <= len(instances) <= 4
        assert sum(1 for inst in instances if inst["is_primary"]) == 1


def test_unknown_scenario_name_fails_closed() -> None:
    with pytest.raises(ValueError):
        _take({"scenarios": ["not_a_real_scenario"]}, seed=0, count=1)


def test_forced_image_size_overrides_portrait_default_shape() -> None:
    sample = _take({"scenarios": ["portrait_phone"], "image_size": [640, 360]}, seed=2, count=1)[0]
    assert sample["image_bgr"].shape == (360, 640, 3)
    assert sample["meta"]["image_size"] == [640, 360]


def test_stream_is_lazy_and_supports_infinite_iteration() -> None:
    stream = iter_synthetic_court_samples({}, seed=0)
    first_five = [next(stream) for _ in range(5)]
    assert len(first_five) == 5


def test_throughput_smoke_floor() -> None:
    """Coarse regression guard against catastrophic slowdowns; the authoritative >=25/s bar is
    measured separately (see runs/lanes/cal_synth_20260705/report.md) since CI/dev machines vary."""

    config = {"image_size": [640, 360]}
    stream = iter_synthetic_court_samples(config, seed=123)
    warmup = 5
    for _ in range(warmup):
        next(stream)
    n = 40
    start = time.time()
    for _ in range(n):
        next(stream)
    elapsed = time.time() - start
    rate = n / elapsed
    assert rate >= 8.0, f"synthetic court streaming throughput regressed badly: {rate:.1f} samples/s"
