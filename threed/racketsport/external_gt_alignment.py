"""MPJPE alignment/scoring for the external-ground-truth BODY validation lane.

Owner decision (2026-07-02, binding): with the two-camera owner capture session
cancelled (`runs/body_independent_gt_plan_20260702T031104Z/DECISION_DOC.md` Option A is
dead), the BODY accuracy gate is validated primarily via a **public dataset with real 3D
ground truth**, fed through the gate's already-accepted `external_ground_truth` label
source (`threed.racketsport.body_world_label_finalize.INDEPENDENT_LABEL_SOURCES`). See
`runs/body_external_gt_20260702T*/DATASET_SELECTION.md` for the dataset choice (ASPset-510)
and license, and `runs/body_external_gt_20260702T*/METHODOLOGY.md` for the full write-up
this module implements.

**Why alignment is needed at all.** Our own product's `world_mpjpe` gate
(`threed.racketsport.eval.body_gate_report`) compares predicted and labeled joints with a
plain per-index Euclidean distance (`_joint_errors`), because both sides are already
expressed in the *same* metric court-world frame (grounded via our own known camera
calibration + court plane). An external dataset's ground truth lives in *its own* mocap
world frame, which is not automatically the same frame our model's output is expressed in
unless our pipeline's world-grounding step is fed that dataset's own trusted camera
calibration (see the BODY_INFERENCE_INTEGRATION_TODO note in the methodology doc for the
concrete adapter this requires). Until that adapter is wired end-to-end and exercised,
this module provides three well-defined, standard MPJPE *variants* that make the
comparison meaningful without silently mixing up two different coordinate systems:

1. ``mpjpe`` -- the raw, zero-alignment Euclidean joint error. Only meaningful once both
   sides are genuinely expressed in the same frame (e.g. after the real camera-calibration
   adapter above is wired in). This is what `body_gate_report.py`'s 0.05m threshold is
   calibrated against for pickleball, and is the strictest, least-forgiving variant.
2. ``root_relative_mpjpe`` -- both sequences are shifted so a chosen root joint (or
   root-joint average) sits at the origin every frame, before computing MPJPE. This
   removes absolute translation error entirely and is the classic academic MPJPE
   protocol (Human3.6M-style). It cannot detect any translation/world-grounding error at
   all, only pose *shape* error.
3. ``pa_mpjpe`` -- Procrustes-Aligned MPJPE: a *separate* similarity transform
   (rotation + uniform scale + translation) is solved independently for *every frame* to
   best-fit predicted onto ground-truth joints before measuring error. This is the
   standard "how good is the underlying pose estimate, ignoring camera/scale/orientation
   entirely" metric used throughout the 3D pose literature (e.g. 3DPW PA-MPJPE). Because
   it re-solves per frame, it can silently absorb real temporal drift/inconsistency.
4. ``clip_level_rigid_aligned_mpjpe`` -- a single similarity transform (rotation + scale +
   translation) is solved *once* using every joint across every frame of the clip pooled
   together (Umeyama/Kabsch), then applied uniformly to the whole predicted sequence
   before measuring error. Unlike (3), this cannot hide per-frame drift behind a
   re-fit-every-frame alignment, so it is a stricter proxy for "how good would this look
   if we only had to solve one camera-to-world transform for the whole clip" -- the
   closest available proxy to `world_mpjpe` when the real camera-calibration adapter has
   not (yet) been exercised. **This is the variant `score_external_gt_clip` reports as
   `gate_variant` -- i.e. the number this harness proposes to score against the 0.05m
   gate threshold until the real camera-calibration adapter lands.** It is honestly
   weaker evidence than true zero-fit `mpjpe` after real calibration, and callers/readers
   must not conflate the two -- see the caveats in `METHODOLOGY.md`.

5. ``grounding_consistent_mpjpe`` (BODY-EXT-3, see `runs/manager/heldout_eval_ledger.md`) --
   an ankle/floor-anchored, translation-only variant that isolates our pipeline's
   floor-plus-track grounding *convention* from real pose error, without conflating the two
   the way ``clip_level_rigid_aligned_mpjpe`` did.

   **Why this exists.** BODY-EXT-2's real scoring pass found that our product's own
   world-grounding step (`worldhmr.py:_ground_fast_sam_sample`) re-anchors every predicted
   frame with a per-frame *additive constant*: it snaps the frame's lowest point to an
   assumed floor (product convention: court Z=0) and translates the low-point cluster's
   horizontal position to an externally supplied track position. ASPset-510's ground truth
   was never put through that step -- it lives in the dataset's own real camera-rig
   coordinate frame (verified against the real staged `raw_provenance/cameras/<subject>-
   left.json`: an **identity** extrinsic matrix, i.e. ASPset defines its "world" frame as
   literally the `left` camera's own coordinate frame -- standard computer-vision camera
   convention, X-right/Y-down/Z-forward. Empirically confirmed against real converted
   labels: ankle Y is consistently *larger* than shoulder Y (~0.9-1.0 vs. ~-0.2, i.e. Y
   increases downward) and Z sits at ~15-20m, roughly constant per clip -- camera depth, not
   height). ``clip_level_rigid_aligned_mpjpe`` (the pre-registered BODY-EXT-1 gate variant)
   fits one rotation+scale+translation per clip and therefore *mostly* inherits this
   redefinition as apparent error (BODY-EXT-2: pooled 0.372m, ~7x the 0.05m gate), because
   a single 3D similarity transform cannot simultaneously fix a floor-axis mismatch that
   varies which physical axis it lands on for different clips *and* recover real per-frame
   pose error at the same time.

   **What this variant computes, and why it needs no axis relabeling.** Both the raw
   predicted and raw GT joints for any one ASPset clip are provably expressed in the *same*
   underlying camera-calibrated axes (the adapter built `court_calibration.json` from
   ASPset's own real, verified `<subject>-<camera>.json` extrinsics -- see
   `BODY_INFERENCE_INTEGRATION_TODO.md` and `runs/body_external_gt_20260702T040048Z/
   raw_provenance/cameras/`), so axis *correspondence* (predicted axis i <-> GT axis i)
   already holds; what does **not** hold is a *shared, physically meaningful zero point* per
   frame, because the grounding step's floor-snap value and its externally supplied track
   anchor are per-frame constants specific to the (wrong-axis) product convention and have
   no GT analogue. `grounding_consistent_mpjpe` removes exactly that per-frame constant, by
   re-centering **each sequence independently** on the mean position of its own
   ``floor_joint_names`` (default: ``left_ankle``/``right_ankle`` -- the feet/floor-contact
   proxy both our grounding heuristic is trying to target and what a human actually stands
   on) before measuring error. This is provably robust to whatever arbitrary translation the
   pipeline's grounding step already applied: since that translation is a single per-frame
   constant added uniformly to every joint (including the ankles), re-centering on the
   ankle mean of the *already-grounded* predicted output telescopes to exactly the same
   result as re-centering on the ankle mean of the un-grounded raw output -- the constant
   cancels regardless of its value. No rotation or scale is fit (unlike ``pa_mpjpe``/
   ``clip_level_rigid_aligned_mpjpe``), so real per-ankle-relative shape error is not hidden,
   only the translation-reference mismatch is. It is mechanically the same shape as
   ``root_relative_mpjpe`` (a single per-frame anchor-point subtraction) but deliberately
   anchored on the ankles/feet -- the specific joints our grounding step targets -- rather
   than the hips, so it targets the *grounding*-error class BODY-EXT-2 flagged specifically,
   distinct from the general pose-shape signal ``root_relative_mpjpe`` already gives.

   **Honest limitations, disclosed up front:** (a) frames where a subject is airborne
   (neither foot near the ground, e.g. a jump apex) make the ankle-pair a less physically
   grounded anchor for that frame -- the same caveat class ``root_relative_mpjpe`` already
   has for hip translation during a jump; (b) like every variant in this module except raw
   ``mpjpe``, it cannot validate *absolute* floor-height/world-XY placement accuracy, only
   relative-to-feet geometry; (c) it is restricted to the shared 12-of-17 joint set (limbs
   only, no face), the same scope as every other variant here.

All functions operate on ``(frames, joints, 3)`` numpy arrays in meters.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

ARTIFACT_TYPE = "racketsport_external_gt_alignment_score"
SCHEMA_VERSION = 1


def mpjpe(predicted: np.ndarray, gt: np.ndarray) -> float:
    """Zero-alignment mean per-joint Euclidean error, in the units of the inputs."""

    predicted = np.asarray(predicted, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if predicted.shape != gt.shape:
        raise ValueError(f"shape mismatch: predicted={predicted.shape} gt={gt.shape}")
    return float(np.mean(np.linalg.norm(predicted - gt, axis=-1)))


def root_relative_mpjpe(predicted: np.ndarray, gt: np.ndarray, *, root_index: int) -> float:
    """MPJPE after subtracting a single root joint from every joint, per frame."""

    predicted = np.asarray(predicted, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    predicted_rr = predicted - predicted[:, root_index : root_index + 1, :]
    gt_rr = gt - gt[:, root_index : root_index + 1, :]
    return mpjpe(predicted_rr, gt_rr)


def procrustes_aligned_mpjpe(predicted: np.ndarray, gt: np.ndarray) -> float:
    """PA-MPJPE: fit one similarity transform per frame, then average the error."""

    predicted = np.asarray(predicted, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if predicted.shape != gt.shape:
        raise ValueError(f"shape mismatch: predicted={predicted.shape} gt={gt.shape}")
    errors = []
    for frame_index in range(predicted.shape[0]):
        aligned = _apply_similarity(predicted[frame_index], *_fit_similarity(predicted[frame_index], gt[frame_index]))
        errors.append(np.linalg.norm(aligned - gt[frame_index], axis=-1))
    return float(np.mean(np.concatenate(errors)))


def clip_level_rigid_aligned_mpjpe(predicted: np.ndarray, gt: np.ndarray, *, allow_scale: bool = True) -> float:
    """One similarity transform fit across the whole clip (all frames pooled), then MPJPE."""

    predicted = np.asarray(predicted, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if predicted.shape != gt.shape:
        raise ValueError(f"shape mismatch: predicted={predicted.shape} gt={gt.shape}")
    frames, joints, _ = predicted.shape
    pooled_predicted = predicted.reshape(frames * joints, 3)
    pooled_gt = gt.reshape(frames * joints, 3)
    rotation, translation, scale = _fit_similarity(pooled_predicted, pooled_gt, allow_scale=allow_scale)
    aligned = _apply_similarity(predicted.reshape(-1, 3), rotation, translation, scale)
    return float(np.mean(np.linalg.norm(aligned - pooled_gt, axis=-1)))


def _fit_similarity(
    source: np.ndarray, target: np.ndarray, *, allow_scale: bool = True
) -> tuple[np.ndarray, np.ndarray, float]:
    """Umeyama/Kabsch closed-form similarity transform: target ~= scale * R @ source + t."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean

    covariance = target_centered.T @ source_centered / source.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    d = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        d[-1] = -1.0
    rotation = u @ np.diag(d) @ vt

    if allow_scale:
        source_var = (source_centered**2).sum() / source.shape[0]
        scale = float((singular_values * d).sum() / source_var) if source_var > 1e-12 else 1.0
    else:
        scale = 1.0

    translation = target_mean - scale * rotation @ source_mean
    return rotation, translation, scale


def _apply_similarity(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray, scale: float) -> np.ndarray:
    return (scale * (rotation @ points.T)).T + translation


def _per_joint_norm(diff: np.ndarray) -> np.ndarray:
    """(frames, joints, 3) Euclidean-distance array -> (joints,) mean over frames."""

    return np.mean(np.linalg.norm(diff, axis=-1), axis=0)


def per_joint_mpjpe(predicted: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-joint zero-alignment mean Euclidean error, averaged over frames. Shape (joints,)."""

    predicted = np.asarray(predicted, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if predicted.shape != gt.shape:
        raise ValueError(f"shape mismatch: predicted={predicted.shape} gt={gt.shape}")
    return _per_joint_norm(predicted - gt)


def per_joint_root_relative_mpjpe(predicted: np.ndarray, gt: np.ndarray, *, root_index: int) -> np.ndarray:
    """Per-joint root-relative MPJPE (see `root_relative_mpjpe`). Shape (joints,)."""

    predicted = np.asarray(predicted, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    predicted_rr = predicted - predicted[:, root_index : root_index + 1, :]
    gt_rr = gt - gt[:, root_index : root_index + 1, :]
    return per_joint_mpjpe(predicted_rr, gt_rr)


def per_joint_pa_mpjpe(predicted: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-joint PA-MPJPE (one similarity fit per frame; see `procrustes_aligned_mpjpe`)."""

    predicted = np.asarray(predicted, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if predicted.shape != gt.shape:
        raise ValueError(f"shape mismatch: predicted={predicted.shape} gt={gt.shape}")
    aligned_frames = []
    for frame_index in range(predicted.shape[0]):
        aligned = _apply_similarity(predicted[frame_index], *_fit_similarity(predicted[frame_index], gt[frame_index]))
        aligned_frames.append(aligned)
    return _per_joint_norm(np.stack(aligned_frames, axis=0) - gt)


def per_joint_clip_level_rigid_aligned_mpjpe(predicted: np.ndarray, gt: np.ndarray, *, allow_scale: bool = True) -> np.ndarray:
    """Per-joint clip-level-rigid-aligned MPJPE (see `clip_level_rigid_aligned_mpjpe`)."""

    predicted = np.asarray(predicted, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if predicted.shape != gt.shape:
        raise ValueError(f"shape mismatch: predicted={predicted.shape} gt={gt.shape}")
    frames, joints, _ = predicted.shape
    pooled_predicted = predicted.reshape(frames * joints, 3)
    pooled_gt = gt.reshape(frames * joints, 3)
    rotation, translation, scale = _fit_similarity(pooled_predicted, pooled_gt, allow_scale=allow_scale)
    aligned = _apply_similarity(predicted.reshape(-1, 3), rotation, translation, scale).reshape(frames, joints, 3)
    return _per_joint_norm(aligned - gt)


def per_joint_breakdown(
    *,
    predicted_joints: np.ndarray,
    gt_joints: np.ndarray,
    joint_names: Sequence[str],
    root_joint_names: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Per-joint mean error (meters) for all four named variants, keyed by joint name.

    Mirrors `score_external_gt_clip`'s root-averaging behavior for
    ``root_relative_mpjpe`` (average of every ``root_joint_names`` entry present in
    ``joint_names``, not just the first).
    """

    predicted = np.asarray(predicted_joints, dtype=np.float64)
    gt = np.asarray(gt_joints, dtype=np.float64)
    if predicted.shape != gt.shape:
        raise ValueError(f"shape mismatch: predicted={predicted.shape} gt={gt.shape}")
    if predicted.shape[1] != len(joint_names):
        raise ValueError(f"joint_names length {len(joint_names)} does not match joint axis {predicted.shape[1]}")

    root_indices = [joint_names.index(name) for name in root_joint_names if name in joint_names]
    if not root_indices:
        raise ValueError(f"none of root_joint_names={list(root_joint_names)} found in joint_names")
    if len(root_indices) > 1:
        predicted_root = predicted[:, root_indices, :].mean(axis=1, keepdims=True)
        gt_root = gt[:, root_indices, :].mean(axis=1, keepdims=True)
        rr_per_joint = per_joint_mpjpe(predicted - predicted_root, gt - gt_root)
    else:
        rr_per_joint = per_joint_root_relative_mpjpe(predicted, gt, root_index=root_indices[0])

    per_variant = {
        "mpjpe": per_joint_mpjpe(predicted, gt),
        "root_relative_mpjpe": rr_per_joint,
        "pa_mpjpe": per_joint_pa_mpjpe(predicted, gt),
        "clip_level_rigid_aligned_mpjpe": per_joint_clip_level_rigid_aligned_mpjpe(predicted, gt),
    }
    return {
        name: {variant: float(values[index]) for variant, values in per_variant.items()}
        for index, name in enumerate(joint_names)
    }


VARIANT_DESCRIPTIONS: dict[str, str] = {
    "mpjpe": (
        "Zero-alignment Euclidean joint error. Only valid once predicted and GT are in "
        "the same world frame (requires the real camera-calibration adapter)."
    ),
    "root_relative_mpjpe": (
        "Classic academic MPJPE: both sequences re-centered on a root joint per frame. "
        "Ignores all translation/world-grounding error; measures pose shape only."
    ),
    "pa_mpjpe": (
        "Procrustes-Aligned MPJPE: a separate rotation+scale+translation fit per frame. "
        "Standard literature metric; can absorb real per-frame temporal drift."
    ),
    "clip_level_rigid_aligned_mpjpe": (
        "One rotation+scale+translation fit for the whole clip (Umeyama, pooled across "
        "frames), then Euclidean error. The proxy this harness reports against the 0.05m "
        "gate threshold until the real camera-calibration adapter is wired in."
    ),
}


DEFAULT_FLOOR_JOINT_NAMES: tuple[str, ...] = ("left_ankle", "right_ankle")

GROUNDING_CONSISTENT_DESCRIPTION = (
    "Both sequences independently re-anchored, per frame, on the mean position of their "
    "own floor_joint_names (default: left_ankle/right_ankle), then zero-alignment "
    "(translation-only) Euclidean error. Designed for BODY-EXT-3 (see "
    "runs/manager/heldout_eval_ledger.md) to isolate our pipeline's floor/track grounding "
    "convention from real pose error -- see module docstring section 5."
)


def grounding_consistent_mpjpe(
    predicted: np.ndarray,
    gt: np.ndarray,
    *,
    joint_names: Sequence[str],
    floor_joint_names: Sequence[str] = DEFAULT_FLOOR_JOINT_NAMES,
) -> float:
    """MPJPE after independently re-anchoring each sequence to its own floor-joint mean.

    See module docstring section 5 (``BODY-EXT-3``) for the full rationale. In short: our
    product's own world-grounding (`worldhmr.py:_ground_fast_sam_sample`) re-anchors every
    predicted frame by (a) snapping the frame's lowest point to an assumed floor and
    (b) translating the low-point cluster's horizontal position to an externally supplied
    track position -- both are per-frame *additive constants* applied uniformly to every
    joint. An external dataset's ground truth was never put through that same step, so
    comparing raw ``predicted`` against raw ``gt`` (``mpjpe``) mixes real pose error with
    this grounding-convention gap. This function removes the gap by re-centering **both**
    sequences on the mean position of ``floor_joint_names`` (their own ankles/feet, the
    same joints our pipeline's grounding heuristic is trying to target) independently, per
    frame, before measuring error -- no rotation or scale is fit (unlike
    ``procrustes_aligned_mpjpe``/``clip_level_rigid_aligned_mpjpe``), so real per-joint
    shape error relative to the feet is not hidden, only the shared-but-differently-defined
    translation reference is.
    """

    predicted = np.asarray(predicted, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if predicted.shape != gt.shape:
        raise ValueError(f"shape mismatch: predicted={predicted.shape} gt={gt.shape}")
    floor_indices = [joint_names.index(name) for name in floor_joint_names if name in joint_names]
    if not floor_indices:
        raise ValueError(f"none of floor_joint_names={list(floor_joint_names)} found in joint_names")
    predicted_anchor = predicted[:, floor_indices, :].mean(axis=1, keepdims=True)
    gt_anchor = gt[:, floor_indices, :].mean(axis=1, keepdims=True)
    return mpjpe(predicted - predicted_anchor, gt - gt_anchor)


def per_joint_grounding_consistent_mpjpe(
    predicted: np.ndarray,
    gt: np.ndarray,
    *,
    joint_names: Sequence[str],
    floor_joint_names: Sequence[str] = DEFAULT_FLOOR_JOINT_NAMES,
) -> np.ndarray:
    """Per-joint grounding-consistent MPJPE (see `grounding_consistent_mpjpe`). Shape (joints,)."""

    predicted = np.asarray(predicted, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if predicted.shape != gt.shape:
        raise ValueError(f"shape mismatch: predicted={predicted.shape} gt={gt.shape}")
    floor_indices = [joint_names.index(name) for name in floor_joint_names if name in joint_names]
    if not floor_indices:
        raise ValueError(f"none of floor_joint_names={list(floor_joint_names)} found in joint_names")
    predicted_anchor = predicted[:, floor_indices, :].mean(axis=1, keepdims=True)
    gt_anchor = gt[:, floor_indices, :].mean(axis=1, keepdims=True)
    return per_joint_mpjpe(predicted - predicted_anchor, gt - gt_anchor)


def score_grounding_consistent_variant(
    *,
    predicted_joints: np.ndarray,
    gt_joints: np.ndarray,
    joint_names: Sequence[str],
    floor_joint_names: Sequence[str] = DEFAULT_FLOOR_JOINT_NAMES,
) -> dict[str, Any]:
    """Scalar + per-joint grounding-consistent MPJPE, in the same shape as one entry of
    `score_external_gt_clip`'s ``variants`` dict, for BODY-EXT-3 reporting alongside the
    four pre-existing variants without touching their code paths."""

    predicted = np.asarray(predicted_joints, dtype=np.float64)
    gt = np.asarray(gt_joints, dtype=np.float64)
    value = grounding_consistent_mpjpe(
        predicted, gt, joint_names=joint_names, floor_joint_names=floor_joint_names
    )
    per_joint = per_joint_grounding_consistent_mpjpe(
        predicted, gt, joint_names=joint_names, floor_joint_names=floor_joint_names
    )
    return {
        "value_m": value,
        "description": GROUNDING_CONSISTENT_DESCRIPTION,
        "floor_joint_names": list(floor_joint_names),
        "per_joint_m": {name: float(per_joint[index]) for index, name in enumerate(joint_names)},
    }


def score_external_gt_clip(
    *,
    predicted_joints: np.ndarray,
    gt_joints: np.ndarray,
    joint_names: Sequence[str],
    root_joint_names: Sequence[str],
    clip_id: str = "",
    subject_id: str = "",
    gate_variant: str = "clip_level_rigid_aligned_mpjpe",
) -> dict[str, Any]:
    """Score one external-GT clip across all MPJPE variants.

    ``predicted_joints``/``gt_joints`` must already be reduced to the shared, mapped
    joint set (same joint count/order as ``joint_names``) -- see
    `threed.racketsport.external_gt_aspset510` for the ASPset-510-specific mapping that
    produces this input.
    """

    predicted = np.asarray(predicted_joints, dtype=np.float64)
    gt = np.asarray(gt_joints, dtype=np.float64)
    if predicted.shape != gt.shape:
        raise ValueError(f"shape mismatch: predicted={predicted.shape} gt={gt.shape}")
    if predicted.ndim != 3 or predicted.shape[2] != 3:
        raise ValueError(f"expected (frames, joints, 3) arrays, got shape {predicted.shape}")
    if predicted.shape[1] != len(joint_names):
        raise ValueError(
            f"joint_names length {len(joint_names)} does not match joint axis {predicted.shape[1]}"
        )
    if gate_variant not in VARIANT_DESCRIPTIONS:
        raise ValueError(f"unknown gate_variant: {gate_variant}")

    root_indices = [joint_names.index(name) for name in root_joint_names if name in joint_names]
    if not root_indices:
        raise ValueError(f"none of root_joint_names={list(root_joint_names)} found in joint_names")
    root_index = root_indices[0]
    if len(root_indices) > 1:
        # Average multiple root candidates (e.g. left_hip/right_hip) into one point per frame.
        predicted_root = predicted[:, root_indices, :].mean(axis=1, keepdims=True)
        gt_root = gt[:, root_indices, :].mean(axis=1, keepdims=True)
        rr_value = mpjpe(predicted - predicted_root, gt - gt_root)
    else:
        rr_value = root_relative_mpjpe(predicted, gt, root_index=root_index)

    variants = {
        "mpjpe": {"value_m": mpjpe(predicted, gt), "description": VARIANT_DESCRIPTIONS["mpjpe"]},
        "root_relative_mpjpe": {"value_m": rr_value, "description": VARIANT_DESCRIPTIONS["root_relative_mpjpe"]},
        "pa_mpjpe": {
            "value_m": procrustes_aligned_mpjpe(predicted, gt),
            "description": VARIANT_DESCRIPTIONS["pa_mpjpe"],
        },
        "clip_level_rigid_aligned_mpjpe": {
            "value_m": clip_level_rigid_aligned_mpjpe(predicted, gt),
            "description": VARIANT_DESCRIPTIONS["clip_level_rigid_aligned_mpjpe"],
        },
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "clip_id": clip_id,
        "subject_id": subject_id,
        "frame_count": int(predicted.shape[0]),
        "joint_count": int(predicted.shape[1]),
        "joint_names": list(joint_names),
        "root_joint_names": list(root_joint_names),
        "variants": variants,
        "gate_variant": gate_variant,
        "gate_value_m": variants[gate_variant]["value_m"],
    }


__all__ = [
    "ARTIFACT_TYPE",
    "DEFAULT_FLOOR_JOINT_NAMES",
    "GROUNDING_CONSISTENT_DESCRIPTION",
    "SCHEMA_VERSION",
    "VARIANT_DESCRIPTIONS",
    "clip_level_rigid_aligned_mpjpe",
    "grounding_consistent_mpjpe",
    "mpjpe",
    "per_joint_breakdown",
    "per_joint_clip_level_rigid_aligned_mpjpe",
    "per_joint_grounding_consistent_mpjpe",
    "per_joint_mpjpe",
    "per_joint_pa_mpjpe",
    "per_joint_root_relative_mpjpe",
    "procrustes_aligned_mpjpe",
    "root_relative_mpjpe",
    "score_external_gt_clip",
    "score_grounding_consistent_variant",
]
