"""Tests for BODY-EXT-3's `grounding_consistent_mpjpe` variant.

See `runs/manager/heldout_eval_ledger.md` row BODY-EXT-3 for the pre-registered
formulation this implements, and `threed/racketsport/external_gt_alignment.py`'s module
docstring section 5 for the full rationale. In short: BODY-EXT-2 found that our pipeline's
own world-grounding step re-anchors every predicted frame with a per-frame *additive
constant* (floor-snap + externally supplied track position) that has no analogue in an
external dataset's ground truth, so raw `mpjpe` and the rigid-fit variants conflate a real
grounding-convention gap with genuine pose error. `grounding_consistent_mpjpe` removes that
gap by re-centering **each** sequence independently, per frame, on the mean position of its
own `floor_joint_names` (default: ankles) before measuring error -- no rotation/scale fit.
"""

from __future__ import annotations

import numpy as np
import pytest

from threed.racketsport.external_gt_alignment import (
    DEFAULT_FLOOR_JOINT_NAMES,
    grounding_consistent_mpjpe,
    per_joint_grounding_consistent_mpjpe,
    score_grounding_consistent_variant,
)

JOINT_NAMES = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def _synthetic_pose_sequence(*, n_frames: int, n_joints: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(scale=0.3, size=(n_joints, 3))
    frames = np.stack([base + rng.normal(scale=0.01, size=(n_joints, 3)) for _ in range(n_frames)])
    return frames


def test_identical_pose_under_different_grounding_conventions_scores_near_zero() -> None:
    """The core BODY-EXT-3 case: predicted and GT have the *same* underlying pose, but
    predicted has already been run through a convention-mismatched grounding step (a large,
    per-frame-*varying* constant standing in for our pipeline's floor-snap + externally
    supplied track-position offset, applied to axes that do not correspond to GT's own
    notion of "floor"). This must score ~0, unlike `mpjpe`/`clip_level_rigid_aligned_mpjpe`,
    which BODY-EXT-2 showed inherit most of this as apparent error."""

    gt = _synthetic_pose_sequence(n_frames=10, n_joints=12, seed=1)
    rng = np.random.default_rng(2)
    # Simulate our pipeline's per-frame grounding: a different, large additive constant on
    # every frame (as `_ground_fast_sam_sample`'s dx/dy/dz -- derived from a floor-snap value
    # and an externally supplied track position -- would produce), applied uniformly to
    # every joint in that frame (the defining property of a translation-only artifact).
    per_frame_offsets = rng.uniform(low=-20.0, high=20.0, size=(10, 1, 3))
    predicted = gt + per_frame_offsets

    # Raw MPJPE is dominated by the (large, arbitrary, per-frame-varying) grounding offset...
    assert grounding_consistent_mpjpe(predicted, gt, joint_names=JOINT_NAMES) == pytest.approx(0.0, abs=1e-9)

    from threed.racketsport.external_gt_alignment import mpjpe as raw_mpjpe

    assert raw_mpjpe(predicted, gt) > 1.0


def test_real_grounding_error_is_caught() -> None:
    """A genuine per-joint shape error (not a shared translation) must NOT be cancelled."""

    gt = _synthetic_pose_sequence(n_frames=8, n_joints=12, seed=3)
    predicted = gt.copy()
    wrist_index = JOINT_NAMES.index("left_wrist")
    predicted[:, wrist_index, :] += 0.25  # a real, non-floor-joint pose error
    error = grounding_consistent_mpjpe(predicted, gt, joint_names=JOINT_NAMES)
    assert error > 0.01


def test_error_on_floor_joints_themselves_is_still_detected() -> None:
    """A real error localized to the floor/ankle joints themselves must not be hidden by
    the ankle-mean re-centering (only a *shared* translation should cancel)."""

    gt = _synthetic_pose_sequence(n_frames=8, n_joints=12, seed=4)
    predicted = gt.copy()
    left_ankle = JOINT_NAMES.index("left_ankle")
    predicted[:, left_ankle, :] += np.array([0.3, 0.0, 0.0])  # only one ankle drifts
    error = grounding_consistent_mpjpe(predicted, gt, joint_names=JOINT_NAMES)
    assert error > 0.0


def test_matches_manual_ankle_recentering() -> None:
    gt = _synthetic_pose_sequence(n_frames=6, n_joints=12, seed=5)
    predicted = gt + np.array([2.0, -3.0, 5.0])
    left_ankle = JOINT_NAMES.index("left_ankle")
    right_ankle = JOINT_NAMES.index("right_ankle")

    predicted_anchor = predicted[:, [left_ankle, right_ankle], :].mean(axis=1, keepdims=True)
    gt_anchor = gt[:, [left_ankle, right_ankle], :].mean(axis=1, keepdims=True)
    expected = float(np.mean(np.linalg.norm((predicted - predicted_anchor) - (gt - gt_anchor), axis=-1)))

    assert grounding_consistent_mpjpe(predicted, gt, joint_names=JOINT_NAMES) == pytest.approx(expected, abs=1e-9)


def test_default_floor_joint_names_are_ankles() -> None:
    assert DEFAULT_FLOOR_JOINT_NAMES == ("left_ankle", "right_ankle")


def test_custom_floor_joint_names_are_respected() -> None:
    gt = _synthetic_pose_sequence(n_frames=5, n_joints=12, seed=6)
    predicted = gt + np.array([1.0, 1.0, 1.0])
    hip_names = ("left_hip", "right_hip")
    # A pure shared translation cancels regardless of which floor_joint_names are used.
    assert grounding_consistent_mpjpe(
        predicted, gt, joint_names=JOINT_NAMES, floor_joint_names=hip_names
    ) == pytest.approx(0.0, abs=1e-9)


def test_unknown_floor_joint_names_raises() -> None:
    gt = _synthetic_pose_sequence(n_frames=3, n_joints=3)
    predicted = gt.copy()
    with pytest.raises(ValueError):
        grounding_consistent_mpjpe(
            predicted, gt, joint_names=["a", "b", "c"], floor_joint_names=("not_present",)
        )


def test_shape_mismatch_raises() -> None:
    gt = _synthetic_pose_sequence(n_frames=4, n_joints=3)
    predicted = _synthetic_pose_sequence(n_frames=5, n_joints=3)
    with pytest.raises(ValueError):
        grounding_consistent_mpjpe(predicted, gt, joint_names=["a", "b", "c"])


def test_per_joint_grounding_consistent_mpjpe_averages_to_scalar() -> None:
    gt = _synthetic_pose_sequence(n_frames=7, n_joints=12, seed=7)
    predicted = gt + np.array([0.1, -0.2, 0.3])
    per_joint = per_joint_grounding_consistent_mpjpe(predicted, gt, joint_names=JOINT_NAMES)
    assert per_joint.shape == (12,)
    assert float(np.mean(per_joint)) == pytest.approx(
        grounding_consistent_mpjpe(predicted, gt, joint_names=JOINT_NAMES), abs=1e-9
    )


def test_per_joint_isolates_a_single_bad_joint() -> None:
    gt = _synthetic_pose_sequence(n_frames=6, n_joints=12, seed=8)
    predicted = gt.copy()
    elbow_index = JOINT_NAMES.index("right_elbow")
    predicted[:, elbow_index, :] += np.array([0.4, 0.0, 0.0])
    per_joint = per_joint_grounding_consistent_mpjpe(predicted, gt, joint_names=JOINT_NAMES)
    for index, name in enumerate(JOINT_NAMES):
        if name == "right_elbow":
            assert per_joint[index] == pytest.approx(0.4, abs=1e-6)
        else:
            # Small residual allowed only for the ankles themselves (their own mean shifts
            # slightly since only one joint in the whole 12-joint set moved -- they are
            # unaffected here since right_elbow is not a floor joint).
            assert per_joint[index] == pytest.approx(0.0, abs=1e-6)


def test_score_grounding_consistent_variant_reports_value_and_per_joint_breakdown() -> None:
    gt = _synthetic_pose_sequence(n_frames=6, n_joints=12, seed=9)
    predicted = gt + np.array([50.0, 20.0, -10.0])  # large shared "wrong axis" offset
    result = score_grounding_consistent_variant(
        predicted_joints=predicted, gt_joints=gt, joint_names=JOINT_NAMES
    )
    assert result["value_m"] == pytest.approx(0.0, abs=1e-9)
    assert result["floor_joint_names"] == list(DEFAULT_FLOOR_JOINT_NAMES)
    assert set(result["per_joint_m"].keys()) == set(JOINT_NAMES)
    assert "description" in result and result["description"]
    for value in result["per_joint_m"].values():
        assert value == pytest.approx(0.0, abs=1e-9)


def test_grounding_consistent_is_robust_to_per_frame_varying_arbitrary_offset() -> None:
    """The telescoping property the module docstring claims: re-centering the
    *already-grounded* predicted sequence on its own floor-joint mean must give the same
    answer regardless of what per-frame constant a prior (buggy or convention-mismatched)
    grounding step already applied -- because that constant is added uniformly to every
    joint in the frame, including the floor joints used as the new anchor."""

    gt = _synthetic_pose_sequence(n_frames=9, n_joints=12, seed=10)
    rng = np.random.default_rng(11)
    offset_a = rng.uniform(low=-5.0, high=5.0, size=(9, 1, 3))
    offset_b = rng.uniform(low=-500.0, high=500.0, size=(9, 1, 3))

    error_a = grounding_consistent_mpjpe(gt + offset_a, gt, joint_names=JOINT_NAMES)
    error_b = grounding_consistent_mpjpe(gt + offset_b, gt, joint_names=JOINT_NAMES)

    assert error_a == pytest.approx(0.0, abs=1e-8)
    assert error_b == pytest.approx(0.0, abs=1e-6)
