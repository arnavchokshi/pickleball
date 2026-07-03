from __future__ import annotations

import math

import numpy as np
import pytest

from threed.racketsport.external_gt_alignment import (
    clip_level_rigid_aligned_mpjpe,
    mpjpe,
    per_joint_breakdown,
    per_joint_clip_level_rigid_aligned_mpjpe,
    per_joint_mpjpe,
    per_joint_pa_mpjpe,
    per_joint_root_relative_mpjpe,
    procrustes_aligned_mpjpe,
    root_relative_mpjpe,
    score_external_gt_clip,
)


def _translate(points: np.ndarray, offset: tuple[float, float, float]) -> np.ndarray:
    return points + np.asarray(offset, dtype=np.float64)


def _rotate_z(points: np.ndarray, degrees: float) -> np.ndarray:
    theta = math.radians(degrees)
    rot = np.array(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return points @ rot.T


def _synthetic_pose_sequence(*, n_frames: int, n_joints: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # A stable "shape" per joint plus small per-frame jitter, so root-relative
    # geometry is consistent across frames (like a real short human motion clip).
    base = rng.normal(scale=0.3, size=(n_joints, 3))
    frames = np.stack([base + rng.normal(scale=0.01, size=(n_joints, 3)) for _ in range(n_frames)])
    return frames


def test_mpjpe_zero_for_identical_arrays() -> None:
    points = _synthetic_pose_sequence(n_frames=4, n_joints=5)
    assert mpjpe(points, points) == pytest.approx(0.0, abs=1e-12)


def test_mpjpe_matches_known_constant_offset() -> None:
    points = np.zeros((2, 3, 3))
    offset_points = points + np.array([0.03, 0.04, 0.0])
    # Every joint is displaced by exactly a 3-4-5 triangle -> distance 0.05m.
    assert mpjpe(points, offset_points) == pytest.approx(0.05, abs=1e-9)


def test_root_relative_mpjpe_ignores_pure_translation_error() -> None:
    gt = _synthetic_pose_sequence(n_frames=6, n_joints=17)
    predicted = _translate(gt, (5.0, -2.0, 1.0))
    # Raw MPJPE is huge (dominated by the translation offset)...
    assert mpjpe(predicted, gt) > 1.0
    # ...but root-relative MPJPE is ~0 once the shared translation is removed.
    assert root_relative_mpjpe(predicted, gt, root_index=0) == pytest.approx(0.0, abs=1e-9)


def test_root_relative_mpjpe_is_nonzero_for_real_shape_error() -> None:
    gt = _synthetic_pose_sequence(n_frames=6, n_joints=17)
    predicted = gt.copy()
    predicted[:, 3, :] += 0.2  # displace one non-root joint relative to the root
    error = root_relative_mpjpe(predicted, gt, root_index=0)
    # Only 1 of 17 joints moved, so the mean is diluted, but it must be clearly nonzero.
    assert error > 0.01


def test_procrustes_aligned_mpjpe_absorbs_rotation_translation_and_scale() -> None:
    gt = _synthetic_pose_sequence(n_frames=5, n_joints=17)
    predicted = np.stack([_rotate_z(frame, 37.0) * 1.4 for frame in gt])
    predicted = _translate(predicted, (10.0, 3.0, -1.0))
    # Raw and root-relative MPJPE are both large under this rotation+scale+translation...
    assert mpjpe(predicted, gt) > 1.0
    # ...but per-frame Procrustes alignment recovers ~0 error since the underlying
    # shape is identical up to a similarity transform.
    assert procrustes_aligned_mpjpe(predicted, gt) == pytest.approx(0.0, abs=1e-6)


def test_clip_level_rigid_aligned_mpjpe_uses_one_shared_transform() -> None:
    gt = _synthetic_pose_sequence(n_frames=8, n_joints=17)
    # Apply the SAME rotation/scale/translation to every frame (a single camera
    # miscalibration, not per-frame drift).
    predicted = np.stack([_rotate_z(frame, 12.0) * 0.9 for frame in gt])
    predicted = _translate(predicted, (2.0, 0.5, 0.2))
    assert mpjpe(predicted, gt) > 0.5
    aligned_error = clip_level_rigid_aligned_mpjpe(predicted, gt)
    assert aligned_error == pytest.approx(0.0, abs=1e-6)


def test_clip_level_rigid_aligned_mpjpe_still_penalizes_per_frame_shape_error() -> None:
    """Unlike per-frame Procrustes, a single fitted transform can't hide genuine
    per-frame pose error -- this is the point of using it over PA-MPJPE for a
    'world' proxy metric."""

    gt = _synthetic_pose_sequence(n_frames=8, n_joints=17, seed=1)
    predicted = gt.copy()
    # Real, growing per-frame drift error that a single rigid fit cannot fully undo.
    for frame_index in range(predicted.shape[0]):
        predicted[frame_index, 5, :] += 0.01 * frame_index

    rigid_error = clip_level_rigid_aligned_mpjpe(predicted, gt)
    pa_error = procrustes_aligned_mpjpe(predicted, gt)
    assert rigid_error > 0.0
    # Per-frame Procrustes can absorb more of the drift than a single clip-level fit.
    assert rigid_error >= pa_error


def test_score_external_gt_clip_reports_all_variants_and_metadata() -> None:
    gt = _synthetic_pose_sequence(n_frames=6, n_joints=12, seed=2)
    predicted = _translate(gt, (0.02, 0.0, 0.0))
    joint_names = [
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]

    result = score_external_gt_clip(
        predicted_joints=predicted,
        gt_joints=gt,
        joint_names=joint_names,
        root_joint_names=("left_hip", "right_hip"),
    )

    assert result["frame_count"] == 6
    assert result["joint_count"] == 12
    assert result["joint_names"] == joint_names
    assert result["variants"]["mpjpe"]["value_m"] == pytest.approx(0.02, abs=1e-9)
    assert result["variants"]["root_relative_mpjpe"]["value_m"] == pytest.approx(0.0, abs=1e-9)
    assert result["variants"]["pa_mpjpe"]["value_m"] == pytest.approx(0.0, abs=1e-6)
    assert result["variants"]["clip_level_rigid_aligned_mpjpe"]["value_m"] == pytest.approx(0.0, abs=1e-6)
    for variant in result["variants"].values():
        assert "description" in variant and variant["description"]
    assert result["gate_variant"] == "clip_level_rigid_aligned_mpjpe"


def test_score_external_gt_clip_requires_matching_frame_and_joint_counts() -> None:
    gt = _synthetic_pose_sequence(n_frames=4, n_joints=3)
    predicted = _synthetic_pose_sequence(n_frames=5, n_joints=3)
    with pytest.raises(ValueError):
        score_external_gt_clip(
            predicted_joints=predicted,
            gt_joints=gt,
            joint_names=["a", "b", "c"],
            root_joint_names=("a",),
        )


def test_per_joint_mpjpe_averages_to_scalar_mpjpe() -> None:
    gt = _synthetic_pose_sequence(n_frames=5, n_joints=4, seed=3)
    predicted = _translate(gt, (0.01, -0.02, 0.03))
    per_joint = per_joint_mpjpe(predicted, gt)
    assert per_joint.shape == (4,)
    assert float(np.mean(per_joint)) == pytest.approx(mpjpe(predicted, gt), abs=1e-9)


def test_per_joint_mpjpe_isolates_a_single_bad_joint() -> None:
    gt = _synthetic_pose_sequence(n_frames=5, n_joints=4, seed=4)
    predicted = gt.copy()
    predicted[:, 2, :] += np.array([1.0, 0.0, 0.0])  # only joint index 2 is offset
    per_joint = per_joint_mpjpe(predicted, gt)
    for index in (0, 1, 3):
        assert per_joint[index] == pytest.approx(0.0, abs=1e-9)
    assert per_joint[2] == pytest.approx(1.0, abs=1e-9)


def test_per_joint_root_relative_mpjpe_averages_to_scalar() -> None:
    gt = _synthetic_pose_sequence(n_frames=5, n_joints=4, seed=5)
    predicted = _translate(gt, (0.05, 0.0, 0.0))
    per_joint = per_joint_root_relative_mpjpe(predicted, gt, root_index=0)
    assert float(np.mean(per_joint)) == pytest.approx(root_relative_mpjpe(predicted, gt, root_index=0), abs=1e-9)
    # the root joint is exactly zero relative to itself every frame.
    assert per_joint[0] == pytest.approx(0.0, abs=1e-9)


def test_per_joint_pa_mpjpe_averages_to_scalar() -> None:
    gt = _synthetic_pose_sequence(n_frames=5, n_joints=6, seed=6)
    predicted = _rotate_z(_translate(gt, (0.1, -0.1, 0.05)), 15.0)
    per_joint = per_joint_pa_mpjpe(predicted, gt)
    assert per_joint.shape == (6,)
    assert float(np.mean(per_joint)) == pytest.approx(procrustes_aligned_mpjpe(predicted, gt), abs=1e-6)


def test_per_joint_clip_level_rigid_aligned_mpjpe_averages_to_scalar() -> None:
    gt = _synthetic_pose_sequence(n_frames=6, n_joints=5, seed=7)
    predicted = _rotate_z(_translate(gt, (0.2, 0.1, -0.1)), 7.0)
    per_joint = per_joint_clip_level_rigid_aligned_mpjpe(predicted, gt)
    assert per_joint.shape == (5,)
    assert float(np.mean(per_joint)) == pytest.approx(clip_level_rigid_aligned_mpjpe(predicted, gt), abs=1e-6)


def test_per_joint_breakdown_matches_scalar_variants_and_root_averaging() -> None:
    gt = _synthetic_pose_sequence(n_frames=6, n_joints=12, seed=8)
    predicted = _rotate_z(_translate(gt, (0.03, -0.01, 0.02)), 5.0)
    joint_names = [
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]

    breakdown = per_joint_breakdown(
        predicted_joints=predicted,
        gt_joints=gt,
        joint_names=joint_names,
        root_joint_names=("left_hip", "right_hip"),
    )

    assert set(breakdown.keys()) == set(joint_names)
    for joint_name in joint_names:
        assert set(breakdown[joint_name].keys()) == {
            "mpjpe", "root_relative_mpjpe", "pa_mpjpe", "clip_level_rigid_aligned_mpjpe",
        }

    # averaging each variant across all 12 joints must match the whole-clip scalar score.
    scored = score_external_gt_clip(
        predicted_joints=predicted,
        gt_joints=gt,
        joint_names=joint_names,
        root_joint_names=("left_hip", "right_hip"),
    )
    for variant in ("mpjpe", "root_relative_mpjpe", "pa_mpjpe", "clip_level_rigid_aligned_mpjpe"):
        joint_mean = float(np.mean([breakdown[name][variant] for name in joint_names]))
        assert joint_mean == pytest.approx(scored["variants"][variant]["value_m"], abs=1e-6)


def test_per_joint_breakdown_requires_known_root_joint() -> None:
    gt = _synthetic_pose_sequence(n_frames=3, n_joints=3)
    predicted = gt.copy()
    with pytest.raises(ValueError):
        per_joint_breakdown(
            predicted_joints=predicted,
            gt_joints=gt,
            joint_names=["a", "b", "c"],
            root_joint_names=("not_present",),
        )
