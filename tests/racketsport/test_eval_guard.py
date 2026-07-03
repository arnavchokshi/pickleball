from __future__ import annotations

from pathlib import Path

import pytest

from threed.racketsport.eval_guard import (
    INTERNAL_VAL_ONLY_CLIP_IDS,
    PROTECTED_EVAL_CLIPS,
    PROTECTED_EVAL_CLIP_IDS,
    STRICT_HOLDOUT_CLIP_IDS,
    EvalClipLeakError,
    assert_not_training_on_eval_clip,
)


def test_registry_has_exactly_the_four_protected_clips_with_expected_roles() -> None:
    assert PROTECTED_EVAL_CLIP_IDS == (
        "outdoor_webcam_iynbd_1500_long_high_baseline",
        "indoor_doubles_fwuks_0500_long_mid_baseline",
        "burlington_gold_0300_low_steep_corner",
        "wolverine_mixed_0200_mid_steep_corner",
    )
    assert set(STRICT_HOLDOUT_CLIP_IDS) == {
        "outdoor_webcam_iynbd_1500_long_high_baseline",
        "indoor_doubles_fwuks_0500_long_mid_baseline",
    }
    assert set(INTERNAL_VAL_ONLY_CLIP_IDS) == {
        "burlington_gold_0300_low_steep_corner",
        "wolverine_mixed_0200_mid_steep_corner",
    }
    assert len(PROTECTED_EVAL_CLIPS) == 4
    for clip in PROTECTED_EVAL_CLIPS:
        assert clip.description


def test_outdoor_clip_in_training_paths_is_refused() -> None:
    paths = [
        "cvat_upload/03_outdoor_webcam_iynbd_1500_long_high_baseline_frames_0000_1150.mp4",
        "some/other/clean_clip/video.mp4",
    ]
    with pytest.raises(EvalClipLeakError, match="outdoor_webcam_iynbd_1500_long_high_baseline"):
        assert_not_training_on_eval_clip(paths, allow_internal_val=False)


def test_indoor_clip_is_refused_even_with_allow_internal_val_true() -> None:
    """Strict-holdout clips have no override -- allow_internal_val must not rescue them."""

    paths = ["eval_clips/ball/indoor_doubles_fwuks_0500_long_mid_baseline/source.mp4"]
    with pytest.raises(EvalClipLeakError, match="indoor_doubles_fwuks_0500_long_mid_baseline"):
        assert_not_training_on_eval_clip(paths, allow_internal_val=True)


def test_burlington_without_flag_is_refused() -> None:
    paths = ["cvat_upload/01_burlington_gold_0300_low_steep_corner_10s.mp4"]
    with pytest.raises(EvalClipLeakError, match="burlington_gold_0300_low_steep_corner"):
        assert_not_training_on_eval_clip(paths, allow_internal_val=False)


def test_burlington_with_flag_passes_and_is_logged() -> None:
    paths = ["cvat_upload/01_burlington_gold_0300_low_steep_corner_10s.mp4"]
    summary = assert_not_training_on_eval_clip(paths, allow_internal_val=True)

    assert summary["status"] == "internal_val_used"
    assert summary["allow_internal_val"] is True
    assert summary["internal_val_uses"] == [
        {
            "clip_id": "burlington_gold_0300_low_steep_corner",
            "matched_value": "cvat_upload/01_burlington_gold_0300_low_steep_corner_10s.mp4",
        }
    ]


def test_wolverine_with_flag_passes_and_is_logged() -> None:
    paths = [Path("cvat_upload/02_wolverine_mixed_0200_mid_steep_corner_10s.mp4")]
    summary = assert_not_training_on_eval_clip(paths, allow_internal_val=True)

    assert summary["status"] == "internal_val_used"
    assert summary["internal_val_uses"][0]["clip_id"] == "wolverine_mixed_0200_mid_steep_corner"


def test_clean_external_data_passes() -> None:
    paths = [
        "runs/roboflow_pickleball_dataset/images/train/frame_000123.jpg",
        Path("runs/roboflow_pickleball_dataset/labels/train/frame_000123.txt"),
        "clip_a",
        "clip_b",
    ]
    summary = assert_not_training_on_eval_clip(paths, allow_internal_val=False)

    assert summary["status"] == "clean"
    assert summary["internal_val_uses"] == []
    assert summary["checked_item_count"] == len(paths)


def test_clean_external_data_passes_with_internal_val_flag_too() -> None:
    # allow_internal_val=True should not create false positives on clean data.
    summary = assert_not_training_on_eval_clip(["clip_a", "clip_b"], allow_internal_val=True)
    assert summary["status"] == "clean"


def test_matches_bare_clip_id_strings_not_just_paths() -> None:
    with pytest.raises(EvalClipLeakError):
        assert_not_training_on_eval_clip(["wolverine_mixed_0200_mid_steep_corner"], allow_internal_val=False)


def test_matches_clip_ids_nested_inside_manifest_like_structures() -> None:
    manifest_fragment = {
        "clip_counts": {
            "outdoor_webcam_iynbd_1500_long_high_baseline": {"train": 0, "query": 12, "gallery": 12},
        },
        "splits": {"val": [{"clip": "clean_clip_a", "frame_count": 10}]},
    }
    with pytest.raises(EvalClipLeakError, match="outdoor_webcam_iynbd_1500_long_high_baseline"):
        assert_not_training_on_eval_clip([manifest_fragment], allow_internal_val=True)


def test_matches_clip_id_embedded_with_prefix_and_suffix_tokens() -> None:
    # Real repo filenames prefix/suffix the bare clip id (e.g. "01_..._10s.mp4"),
    # so matching must not require the clip id to be an isolated path segment.
    with pytest.raises(EvalClipLeakError, match="burlington_gold_0300_low_steep_corner"):
        assert_not_training_on_eval_clip(
            ["01_burlington_gold_0300_low_steep_corner_10s.mp4"], allow_internal_val=False
        )


def test_unknown_clip_id_in_clip_ids_argument_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unknown protected eval clip id"):
        assert_not_training_on_eval_clip(["anything"], clip_ids=("not_a_real_clip",))


def test_non_string_scalars_are_ignored_without_error() -> None:
    summary = assert_not_training_on_eval_clip([1, 2.5, True, None, "clean_clip"], allow_internal_val=False)
    assert summary["status"] == "clean"
