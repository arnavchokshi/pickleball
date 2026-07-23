"""Training-only dense targets, dynamic assignment, and auxiliary losses.

This module deliberately has no decoding or metric dependencies.  Assignment is
performed from detached logits and therefore cannot backpropagate through the
discrete Hungarian decision; gradients flow only through the returned targets'
losses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F


@dataclass(frozen=True)
class AssignmentResult:
    """Targets and auditable counters produced by label assignment."""

    dense_targets: torch.Tensor
    assigned_targets: torch.Tensor
    offset_targets: torch.Tensor
    offset_mask: torch.Tensor
    event_count: int
    shifted_event_count: int
    total_abs_shift: int


def _validate_target_inputs(
    hard_targets: torch.Tensor,
    validity_mask: torch.Tensor,
    frame_loss_mask: torch.Tensor,
) -> tuple[int, int, int, torch.Tensor, torch.Tensor]:
    if hard_targets.ndim != 2:
        raise ValueError("hard_targets must be [B,T]")
    if hard_targets.dtype == torch.bool or torch.is_floating_point(hard_targets):
        raise ValueError("hard_targets must contain integer class ids")
    batch_size, time_steps = hard_targets.shape
    if validity_mask.ndim != 2 or validity_mask.shape[0] != batch_size:
        raise ValueError("validity_mask must be [B,C]")
    class_count = validity_mask.shape[1]
    if class_count < 2:
        raise ValueError("at least background and one event class are required")
    if frame_loss_mask.shape != hard_targets.shape:
        raise ValueError("frame_loss_mask must be [B,T]")

    validity = validity_mask.to(device=hard_targets.device, dtype=torch.bool)
    frame_validity = frame_loss_mask.to(device=hard_targets.device, dtype=torch.bool)
    if not bool(validity[:, 0].all()):
        raise ValueError("background must be valid for every sample")

    active_targets = hard_targets[frame_validity]
    if active_targets.numel() and (
        bool((active_targets < 0).any()) or bool((active_targets >= class_count).any())
    ):
        raise ValueError("loss-valid hard_targets contain an out-of-range class id")
    return batch_size, time_steps, class_count, validity, frame_validity


def build_soft_dense_targets(
    hard_targets: torch.Tensor,
    validity_mask: torch.Tensor,
    frame_loss_mask: torch.Tensor,
    *,
    label_dilation_frames: int = 0,
    neighbor_positive_weight: float = 0.5,
) -> torch.Tensor:
    """Convert hard per-frame labels to dense class distributions.

    With one-frame dilation, an otherwise-background neighbor receives total
    positive mass ``neighbor_positive_weight``.  If two event centers collide on
    that neighbor, the positive mass is divided by their contribution counts.
    An event center is always left hard one-hot ("centers win").  UNKNOWN/ignored
    frames and loss-valid frames whose hard class is invalid for that sample have
    zero target mass, allowing the loss mask to exclude them without converting
    them to background.
    """

    (
        batch_size,
        time_steps,
        class_count,
        validity,
        frame_validity,
    ) = _validate_target_inputs(hard_targets, validity_mask, frame_loss_mask)
    if label_dilation_frames not in {0, 1}:
        raise ValueError("label_dilation_frames must be 0 or 1")
    if (
        not np.isfinite(neighbor_positive_weight)
        or not 0.0 < neighbor_positive_weight <= 1.0
    ):
        raise ValueError("neighbor_positive_weight must be finite and in (0,1]")

    dense = torch.zeros(
        (batch_size, time_steps, class_count),
        dtype=torch.float32,
        device=hard_targets.device,
    )

    # Preserve the hard target wherever both frame-level and class-level
    # supervision say it is loss-valid.  Other rows intentionally remain zero.
    for batch_index in range(batch_size):
        for frame_index in range(time_steps):
            if not bool(frame_validity[batch_index, frame_index]):
                continue
            class_id = int(hard_targets[batch_index, frame_index])
            if bool(validity[batch_index, class_id]):
                dense[batch_index, frame_index, class_id] = 1.0

    if label_dilation_frames == 0:
        return dense

    # Count neighbor contributions before writing them.  This makes collision
    # behavior independent of event traversal order and keeps total positive
    # neighbor mass normalized rather than growing with the number of centers.
    neighbor_counts = torch.zeros_like(dense)
    for batch_index in range(batch_size):
        for center_frame in range(time_steps):
            if not bool(frame_validity[batch_index, center_frame]):
                continue
            class_id = int(hard_targets[batch_index, center_frame])
            if class_id == 0 or not bool(validity[batch_index, class_id]):
                continue
            for neighbor_frame in (center_frame - 1, center_frame + 1):
                if not 0 <= neighbor_frame < time_steps:
                    continue
                if not bool(frame_validity[batch_index, neighbor_frame]):
                    continue
                # Dilation applies only to a known background label.  In
                # particular it can never overwrite another event center.
                if int(hard_targets[batch_index, neighbor_frame]) != 0:
                    continue
                neighbor_counts[batch_index, neighbor_frame, class_id] += 1.0

    positive_counts = neighbor_counts[..., 1:].sum(dim=-1)
    collision_frames = positive_counts > 0
    if bool(collision_frames.any()):
        normalized = neighbor_counts[..., 1:] / positive_counts.clamp_min(1)[..., None]
        dense[..., 0][collision_frames] = 1.0 - neighbor_positive_weight
        dense[..., 1:][collision_frames] = (
            normalized[collision_frames] * neighbor_positive_weight
        )
    return dense


def dynamic_label_assignment(
    logits: torch.Tensor,
    hard_targets: torch.Tensor,
    validity_mask: torch.Tensor,
    frame_loss_mask: torch.Tensor,
    event_subframe_offsets: torch.Tensor | None = None,
    *,
    mode: Literal["fixed", "hungarian"] = "fixed",
    max_shift_frames: int = 0,
    class_cost_weight: float = 1.0,
    temporal_cost_weight: float = 1.0,
    label_dilation_frames: int = 0,
    neighbor_positive_weight: float = 0.5,
) -> AssignmentResult:
    """Assign sparse events to unique nearby frames using detached logits.

    Hungarian cost is ``-log(p(class))`` plus absolute temporal displacement,
    normalized by ``max(max_shift_frames, 1)``.  The latter uses the optional
    sub-frame timestamp.  Candidate eligibility itself remains an integer
    ``+-max_shift_frames`` radius around the registered source frame.
    """

    (
        batch_size,
        time_steps,
        class_count,
        validity,
        frame_validity,
    ) = _validate_target_inputs(hard_targets, validity_mask, frame_loss_mask)
    if logits.shape != (batch_size, time_steps, class_count):
        raise ValueError("logits must be [B,T,C] and match target dimensions")
    if class_count != 3:
        raise ValueError("event assignment requires background, HIT, and BOUNCE")
    if mode not in {"fixed", "hungarian"}:
        raise ValueError("mode must be 'fixed' or 'hungarian'")
    if isinstance(max_shift_frames, bool) or not isinstance(max_shift_frames, int):
        raise ValueError("max_shift_frames must be a non-negative integer")
    if max_shift_frames < 0:
        raise ValueError("max_shift_frames must be a non-negative integer")
    for name, value in (
        ("class_cost_weight", class_cost_weight),
        ("temporal_cost_weight", temporal_cost_weight),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")

    if event_subframe_offsets is None:
        subframe_offsets = torch.zeros(
            hard_targets.shape, dtype=logits.dtype, device=hard_targets.device
        )
    else:
        if event_subframe_offsets.shape != hard_targets.shape:
            raise ValueError("event_subframe_offsets must be [B,T]")
        subframe_offsets = event_subframe_offsets.detach().to(
            device=hard_targets.device, dtype=logits.dtype
        )

    assigned_targets = torch.zeros_like(hard_targets)
    offset_targets = torch.zeros(
        (batch_size, time_steps, 2), dtype=logits.dtype, device=logits.device
    )
    offset_mask = torch.zeros(
        (batch_size, time_steps, 2), dtype=torch.bool, device=logits.device
    )
    event_count = 0
    shifted_event_count = 0
    total_abs_shift = 0

    masked_logits = logits.masked_fill(
        ~validity.to(logits.device)[:, None, :], -1.0e4
    )
    if not bool(torch.isfinite(masked_logits).all()):
        raise ValueError("logits must be finite")
    detached_log_probabilities = F.log_softmax(masked_logits, dim=-1).detach().cpu()

    for batch_index in range(batch_size):
        events: list[tuple[int, int, float]] = []
        for original_frame in range(time_steps):
            if not bool(frame_validity[batch_index, original_frame]):
                continue
            class_id = int(hard_targets[batch_index, original_frame])
            if class_id == 0 or not bool(validity[batch_index, class_id]):
                continue
            subframe = float(subframe_offsets[batch_index, original_frame].detach().cpu())
            if not np.isfinite(subframe):
                raise ValueError("event_subframe_offsets must be finite at event frames")
            events.append((original_frame, class_id, subframe))

        event_count += len(events)
        if not events:
            continue

        if mode == "fixed":
            selected_frames = [event[0] for event in events]
        else:
            candidate_frames = sorted({
                candidate_frame
                for original_frame, _, _ in events
                for candidate_frame in range(
                    max(0, original_frame - max_shift_frames),
                    min(time_steps, original_frame + max_shift_frames + 1),
                )
                if bool(frame_validity[batch_index, candidate_frame])
            })
            # Every event's original frame is valid and therefore provides a
            # distinct feasible candidate, so a complete one-to-one assignment
            # must exist.
            costs = np.full((len(events), len(candidate_frames)), np.inf, dtype=np.float64)
            temporal_scale = float(max(max_shift_frames, 1))
            for event_index, (original_frame, class_id, subframe) in enumerate(events):
                for candidate_index, candidate_frame in enumerate(candidate_frames):
                    if abs(candidate_frame - original_frame) > max_shift_frames:
                        continue
                    negative_log_probability = -float(
                        detached_log_probabilities[
                            batch_index, candidate_frame, class_id
                        ]
                    )
                    temporal_offset = abs(
                        original_frame + subframe - candidate_frame
                    ) / temporal_scale
                    # The non-separable row*column term resolves exact equal-cost
                    # pairings while remaining far below meaningful loss deltas.
                    tie_break = 1.0e-12 * (event_index + 1) * (candidate_index + 1)
                    costs[event_index, candidate_index] = (
                        class_cost_weight * negative_log_probability
                        + temporal_cost_weight * temporal_offset
                        + tie_break
                    )
            selected_rows, selected_columns = linear_sum_assignment(costs)
            if len(selected_rows) != len(events):
                raise RuntimeError("Hungarian assignment did not cover every event")
            selected_frames_by_event = {
                int(row): candidate_frames[int(column)]
                for row, column in zip(selected_rows, selected_columns, strict=True)
            }
            selected_frames = [
                selected_frames_by_event[event_index]
                for event_index in range(len(events))
            ]

        if len(set(selected_frames)) != len(selected_frames):
            raise RuntimeError("assignment produced duplicate candidate frames")
        for (original_frame, class_id, subframe), assigned_frame in zip(
            events, selected_frames, strict=True
        ):
            if not bool(frame_validity[batch_index, assigned_frame]):
                raise RuntimeError("assignment selected an UNKNOWN/ignored frame")
            assigned_targets[batch_index, assigned_frame] = class_id
            channel = class_id - 1
            offset_targets[batch_index, assigned_frame, channel] = (
                original_frame + subframe - assigned_frame
            )
            offset_mask[batch_index, assigned_frame, channel] = True
            integer_shift = abs(assigned_frame - original_frame)
            shifted_event_count += int(integer_shift > 0)
            total_abs_shift += integer_shift

    dense_targets = build_soft_dense_targets(
        assigned_targets,
        validity,
        frame_validity,
        label_dilation_frames=label_dilation_frames,
        neighbor_positive_weight=neighbor_positive_weight,
    )
    return AssignmentResult(
        dense_targets=dense_targets,
        assigned_targets=assigned_targets,
        offset_targets=offset_targets,
        offset_mask=offset_mask,
        event_count=event_count,
        shifted_event_count=shifted_event_count,
        total_abs_shift=total_abs_shift,
    )


def dense_cross_entropy(
    logits: torch.Tensor,
    dense_targets: torch.Tensor,
    validity_mask: torch.Tensor,
    frame_loss_mask: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized dense CE and weighted per-sample sufficient statistics."""

    if logits.ndim != 3 or dense_targets.shape != logits.shape:
        raise ValueError("logits and dense_targets must have matching [B,T,C] shapes")
    batch_size, time_steps, class_count = logits.shape
    if validity_mask.shape != (batch_size, class_count):
        raise ValueError("validity_mask must be [B,C]")
    if frame_loss_mask.shape != (batch_size, time_steps):
        raise ValueError("frame_loss_mask must be [B,T]")
    validity = validity_mask.to(device=logits.device, dtype=torch.bool)
    if not bool(validity[:, 0].all()):
        raise ValueError("background must be valid for every sample")

    targets = dense_targets.to(device=logits.device, dtype=logits.dtype)
    if not bool(torch.isfinite(targets).all()) or bool((targets < 0).any()):
        raise ValueError("dense_targets must be finite and non-negative")
    target_mass = targets.sum(dim=-1)
    mass_is_zero_or_one = torch.isclose(
        target_mass, torch.zeros_like(target_mass), atol=1e-6, rtol=0
    ) | torch.isclose(
        target_mass, torch.ones_like(target_mass), atol=1e-6, rtol=0
    )
    if not bool(mass_is_zero_or_one.all()):
        raise ValueError("each dense target row must sum to zero or one")

    if class_weights is None:
        classes = torch.ones(class_count, dtype=logits.dtype, device=logits.device)
    else:
        classes = torch.as_tensor(
            class_weights, dtype=logits.dtype, device=logits.device
        )
        if classes.shape != (class_count,):
            raise ValueError("class_weights must be [C]")
        if not bool(torch.isfinite(classes).all()) or not bool((classes > 0).all()):
            raise ValueError("class_weights must be finite and strictly positive")

    if sample_weights is None:
        samples = torch.ones(batch_size, dtype=logits.dtype, device=logits.device)
    else:
        samples = torch.as_tensor(
            sample_weights, dtype=logits.dtype, device=logits.device
        )
        if samples.shape != (batch_size,):
            raise ValueError("sample_weights must be [B]")
        if not bool(torch.isfinite(samples).all()) or bool((samples < 0).any()):
            raise ValueError("sample_weights must be finite and non-negative")

    invalid_target_mass = (targets * (~validity[:, None, :])).sum(dim=-1)
    target_columns_valid = invalid_target_mass <= 1e-7
    active_frames = (
        frame_loss_mask.to(device=logits.device, dtype=torch.bool)
        & (target_mass > 0)
        & target_columns_valid
    )
    masked_logits = logits.masked_fill(~validity[:, None, :], -1.0e4)
    log_probabilities = F.log_softmax(masked_logits, dim=-1)
    weighted_targets = targets * classes
    per_frame_numerator = -(weighted_targets * log_probabilities).sum(dim=-1)
    per_frame_normalizer = weighted_targets.sum(dim=-1)
    active = active_frames.to(logits.dtype)
    per_sample_numerator = (per_frame_numerator * active).sum(dim=1) * samples
    per_sample_normalizer = (per_frame_normalizer * active).sum(dim=1) * samples
    total_normalizer = per_sample_normalizer.sum()
    if not bool(total_normalizer > 0):
        raise ValueError("batch contains no loss-valid target mass")
    loss = per_sample_numerator.sum() / total_normalizer
    return loss, per_sample_numerator, per_sample_normalizer


def offset_smooth_l1(
    predicted_offsets: torch.Tensor,
    offset_targets: torch.Tensor,
    offset_mask: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return masked Smooth-L1 loss and weighted per-sample statistics."""

    if predicted_offsets.ndim != 3 or offset_targets.shape != predicted_offsets.shape:
        raise ValueError("predicted_offsets and offset_targets must match [B,T,K]")
    if offset_mask.shape != predicted_offsets.shape:
        raise ValueError("offset_mask must match predicted_offsets")
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    batch_size = predicted_offsets.shape[0]
    targets = offset_targets.to(
        device=predicted_offsets.device, dtype=predicted_offsets.dtype
    )
    mask = offset_mask.to(device=predicted_offsets.device, dtype=torch.bool)
    if bool(mask.any()) and not bool(torch.isfinite(targets[mask]).all()):
        raise ValueError("offset_targets must be finite where offset_mask is true")
    if sample_weights is None:
        samples = torch.ones(
            batch_size, dtype=predicted_offsets.dtype, device=predicted_offsets.device
        )
    else:
        samples = torch.as_tensor(
            sample_weights,
            dtype=predicted_offsets.dtype,
            device=predicted_offsets.device,
        )
        if samples.shape != (batch_size,):
            raise ValueError("sample_weights must be [B]")
        if not bool(torch.isfinite(samples).all()) or bool((samples < 0).any()):
            raise ValueError("sample_weights must be finite and non-negative")

    elementwise = F.smooth_l1_loss(
        predicted_offsets, targets, reduction="none", beta=float(beta)
    )
    active = mask.to(predicted_offsets.dtype)
    per_sample_numerator = (
        (elementwise * active).flatten(1).sum(dim=1) * samples
    )
    per_sample_normalizer = active.flatten(1).sum(dim=1) * samples
    total_normalizer = per_sample_normalizer.sum()
    if not bool(total_normalizer > 0):
        zero = predicted_offsets.sum() * 0.0
        return zero, per_sample_numerator, per_sample_normalizer
    return (
        per_sample_numerator.sum() / total_normalizer,
        per_sample_numerator,
        per_sample_normalizer,
    )


__all__ = [
    "AssignmentResult",
    "build_soft_dense_targets",
    "dense_cross_entropy",
    "dynamic_label_assignment",
    "offset_smooth_l1",
]
