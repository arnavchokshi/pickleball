from __future__ import annotations

import hashlib
import itertools
import json
import math
import time

import numpy as np
import pytest

from threed.racketsport.court_keypoint_net import ALL_PICKLEBALL_KEYPOINTS, AUX_PICKLEBALL_KEYPOINTS, PICKLEBALL_KEYPOINTS
from threed.racketsport.court_synth_scenes import (
    ALL_KEYPOINT_NAMES,
    KEYPOINT_VIS_CLASSES,
    LINE_FAMILY_CLASSES,
    SCENARIO_NAMES,
    SURFACE_CLASSES,
)
from threed.racketsport.court_synth_stream import (
    iter_synthetic_court_samples,
    reproject_canonical_keypoints,
    reproject_scene_keypoints,
)

CANONICAL_NAMES = [point.name for point in PICKLEBALL_KEYPOINTS]


def _take(config, seed, count):
    return list(itertools.islice(iter_synthetic_court_samples(config, seed=seed), count))


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def test_default_off_matches_pre_a2_golden_and_explicit_false_is_identical() -> None:
    """Stamp the legacy six-key sample before any A2 RNG draw or target-channel change."""

    config = {
        "count": 1,
        "image_size": [160, 90],
        "scenarios": ["dedicated_indoor"],
        "focal_px_range": [125, 500],
        "apply_jpeg_roundtrip": False,
    }
    implicit = _take(config, seed=20260722, count=1)[0]
    explicit = _take(
        {
            **config,
            "aux_keypoints": False,
            "paint_texture_randomization": False,
            "aux_partial_visibility": False,
        },
        seed=20260722,
        count=1,
    )[0]

    assert set(implicit) == {
        "image_bgr",
        "keypoints_xy",
        "keypoints_vis",
        "line_family_mask",
        "surface_mask",
        "meta",
    }
    expected_hashes = {
        "image_bgr": "085e0e6d31440d1bd5c3b887a700b4c0e69c74dfd465c5f188661ad5bb9f5038",
        "keypoints_xy": "42b46e78b6f2b5f2981bf03d766fbe1811fee5bcfa51aa016eb072540bd0f5f5",
        "keypoints_vis": "a1c8eb44beeb0c6e4ba983416ef7a86807c8f77110974fd4979c28c315f28ea0",
        "line_family_mask": "4e03417b660f9dcf13c466eb29cc1866b161f00d62935db728ed343fcb4d8694",
        "surface_mask": "e02f8f85ecc7d17e692dec5205739e0656888525353fab2e071b41bbf5c56559",
    }
    for name, expected_hash in expected_hashes.items():
        assert _array_sha256(implicit[name]) == expected_hash
        assert np.array_equal(implicit[name], explicit[name])
    implicit_meta_sha256 = hashlib.sha256(
        json.dumps(implicit["meta"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert implicit_meta_sha256 == "62ebcb39ea91f26512647f2c4fc11d2104b101682b9d1e909429dd270d383f44"
    assert implicit["meta"] == explicit["meta"]


def test_aux_contract_is_canonical_first_and_emits_stride_four_masked_targets() -> None:
    sample = _take(
        {
            "count": 1,
            "image_size": [160, 92],
            "scenarios": ["dedicated_outdoor"],
            "aux_keypoints": True,
        },
        seed=17,
        count=1,
    )[0]

    assert len(PICKLEBALL_KEYPOINTS) == 15
    assert len(AUX_PICKLEBALL_KEYPOINTS) == 18
    assert len(ALL_PICKLEBALL_KEYPOINTS) == 33
    assert sample["meta"]["keypoint_names"] == list(ALL_KEYPOINT_NAMES)
    assert [point.name for point in ALL_PICKLEBALL_KEYPOINTS] == list(ALL_KEYPOINT_NAMES)
    assert sample["keypoints_xy"].shape == (33, 2)
    assert sample["keypoints_vis"].shape == (33,)
    assert sample["keypoint_heatmaps"].shape == (33, 23, 40)
    assert sample["keypoint_heatmap_mask"].shape == (33, 23, 40)
    assert sample["keypoint_heatmaps"].dtype == np.float32
    assert sample["keypoint_heatmap_mask"].dtype == np.float32

    reprojected = reproject_scene_keypoints(sample["meta"])
    for index, name in enumerate(ALL_KEYPOINT_NAMES):
        assert np.linalg.norm(np.asarray(reprojected[name]) - sample["keypoints_xy"][index]) < 1e-4
        channel_mask = sample["keypoint_heatmap_mask"][index]
        channel_target = sample["keypoint_heatmaps"][index]
        if sample["keypoints_vis"][index] == KEYPOINT_VIS_CLASSES["off_frame"]:
            assert not np.any(channel_mask)
            assert not np.any(channel_target)
        elif index >= len(PICKLEBALL_KEYPOINTS) and not np.any(channel_mask):
            assert not np.any(channel_target)
        else:
            assert np.all(channel_mask == 1.0)
            assert float(channel_target.max()) > 0.0


def test_aux_partial_visibility_sampler_reaches_six_point_margin() -> None:
    samples = _take(
        {
            "count": 12,
            "image_size": [640, 360],
            "scenarios": ["harsh_shadow"],
            "focal_px_range": [1400.0, 2000.0],
            "aux_keypoints": True,
            "paint_texture_randomization": False,
        },
        seed=20260722,
        count=12,
    )
    in_frame_counts = [
        (
            int(np.count_nonzero(sample["keypoints_vis"][:15])),
            int(np.count_nonzero(sample["keypoints_vis"])),
        )
        for sample in samples
    ]

    assert min(canonical for canonical, _ in in_frame_counts) < 9
    assert min(combined for _, combined in in_frame_counts) >= 6


def test_paint_texture_randomization_is_opt_in_deterministic_and_changes_pixels() -> None:
    base = {
        "count": 1,
        "image_size": [160, 92],
        "scenarios": ["dedicated_outdoor"],
        "aux_keypoints": True,
    }
    dormant = _take({**base, "paint_texture_randomization": False}, seed=31, count=1)[0]
    active_a = _take({**base, "paint_texture_randomization": True}, seed=31, count=1)[0]
    active_b = _take({**base, "paint_texture_randomization": True}, seed=31, count=1)[0]

    assert np.array_equal(dormant["keypoints_xy"], active_a["keypoints_xy"])
    assert not np.array_equal(dormant["image_bgr"], active_a["image_bgr"])
    assert np.array_equal(active_a["image_bgr"], active_b["image_bgr"])
    assert active_a["meta"] == active_b["meta"]
    audit = active_a["meta"]["paint_texture_randomization"]
    assert audit["line_wear_range"] == [0.0, 0.55]
    assert audit["line_fade_alpha_range"] == [0.2, 1.0]
    assert audit["surface_texture_strength_range"] == [0.0, 0.18]
    assert audit["visual_mask_erosion_coupled"] is True


def test_worn_away_aux_paint_is_unsupervised() -> None:
    sample = _take(
        {
            "count": 1,
            "image_size": [320, 180],
            "scenarios": ["dedicated_indoor"],
            "aux_keypoints": True,
            "aux_partial_visibility": False,
            "paint_texture_randomization": True,
            "line_wear_range": [1.0, 1.0],
        },
        seed=20260722,
        count=1,
    )[0]

    aux_vis = sample["keypoints_vis"][len(PICKLEBALL_KEYPOINTS) :]
    aux_targets = sample["keypoint_heatmaps"][len(PICKLEBALL_KEYPOINTS) :]
    aux_masks = sample["keypoint_heatmap_mask"][len(PICKLEBALL_KEYPOINTS) :]
    in_frame = aux_vis != KEYPOINT_VIS_CLASSES["off_frame"]
    assert np.any(in_frame)
    assert not np.any(aux_targets[in_frame])
    assert not np.any(aux_masks[in_frame])
    assert sample["meta"]["aux_keypoints"]["paint_unsupported_in_frame_count"] == int(in_frame.sum())
    assert sample["meta"]["aux_keypoints"]["target_mask_rule"] == (
        "off_frame_or_no_local_pickleball_paint_support_is_unsupervised"
    )


def test_aux_randomization_config_fails_closed_on_invalid_values() -> None:
    with pytest.raises(ValueError, match="aux_partial_visibility requires"):
        _take({"aux_partial_visibility": True}, seed=0, count=1)
    with pytest.raises(ValueError, match="line_wear_range"):
        _take(
            {
                "aux_keypoints": True,
                "line_wear_range": [-0.1, 0.2],
            },
            seed=0,
            count=1,
        )
    with pytest.raises(ValueError, match="requires heatmap_stride=4"):
        _take(
            {
                "aux_keypoints": True,
                "heatmap_stride": 2,
            },
            seed=0,
            count=1,
        )


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
