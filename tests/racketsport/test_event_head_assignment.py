from __future__ import annotations

import torch
from torch.nn import functional as F

from threed.racketsport.event_head.assignment import (
    build_soft_dense_targets,
    dense_cross_entropy,
    dynamic_label_assignment,
    offset_smooth_l1,
)


def _all_valid(batch_size: int, time_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.ones((batch_size, 3), dtype=torch.bool),
        torch.ones((batch_size, time_steps), dtype=torch.bool),
    )


def test_fixed_assignment_without_dilation_is_the_hard_target_identity() -> None:
    targets = torch.tensor([[0, 1, 0, 2, 0]])
    validity, frame_mask = _all_valid(1, 5)
    logits = torch.randn(1, 5, 3, generator=torch.Generator().manual_seed(7))

    result = dynamic_label_assignment(
        logits,
        targets,
        validity,
        frame_mask,
        mode="fixed",
        max_shift_frames=2,
        label_dilation_frames=0,
    )

    assert torch.equal(result.assigned_targets, targets)
    assert torch.equal(result.dense_targets.argmax(dim=-1), targets)
    assert torch.equal(result.dense_targets.sum(dim=-1), torch.ones_like(targets).float())
    assert result.event_count == 2
    assert result.shifted_event_count == 0
    assert result.total_abs_shift == 0
    assert result.offset_mask.nonzero().tolist() == [[0, 1, 0], [0, 3, 1]]
    assert torch.equal(
        result.offset_targets[result.offset_mask], torch.tensor([0.0, 0.0])
    )


def test_one_frame_dilation_makes_soft_positive_neighbors_and_hard_center() -> None:
    targets = torch.tensor([[0, 0, 1, 0, 0]])
    validity, frame_mask = _all_valid(1, 5)

    dense = build_soft_dense_targets(
        targets,
        validity,
        frame_mask,
        label_dilation_frames=1,
        neighbor_positive_weight=0.25,
    )

    assert torch.equal(dense[0, 2], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.equal(dense[0, 1], torch.tensor([0.75, 0.25, 0.0]))
    assert torch.equal(dense[0, 3], torch.tensor([0.75, 0.25, 0.0]))
    assert torch.equal(dense[0, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(dense.sum(dim=-1), torch.ones((1, 5)))


def test_dilation_boundary_collision_normalization_and_centers_winning() -> None:
    # HIT at the boundary and BOUNCE two frames later both dilate onto frame 1.
    # The adjacent HIT/BOUNCE centers at 2 and 3 must never soften each other.
    targets = torch.tensor([[1, 0, 2, 1]])
    validity, frame_mask = _all_valid(1, 4)

    dense = build_soft_dense_targets(
        targets,
        validity,
        frame_mask,
        label_dilation_frames=1,
        neighbor_positive_weight=0.6,
    )

    assert torch.allclose(dense[0, 1], torch.tensor([0.4, 0.3, 0.3]))
    assert torch.equal(dense[0, 0], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.equal(dense[0, 2], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.equal(dense[0, 3], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.allclose(dense.sum(dim=-1), torch.ones((1, 4)))


def test_hungarian_assignment_shifts_events_and_keeps_candidates_one_to_one() -> None:
    targets = torch.tensor([[0, 1, 0, 2, 0]])
    validity, frame_mask = _all_valid(1, 5)
    # Both classes strongly prefer shared frame 2.  HIT has no good alternative,
    # while BOUNCE also has a strong frame 4, forcing a unique global solution.
    logits = torch.tensor([[[
        6.0, -6.0, -6.0,
    ], [6.0, -6.0, -6.0], [-8.0, 9.0, 8.0], [6.0, -6.0, -6.0], [-8.0, -7.0, 8.0]]])

    result = dynamic_label_assignment(
        logits,
        targets,
        validity,
        frame_mask,
        mode="hungarian",
        max_shift_frames=1,
        class_cost_weight=1.0,
        temporal_cost_weight=0.01,
    )

    assert result.assigned_targets.tolist() == [[0, 0, 1, 0, 2]]
    assert result.offset_mask.nonzero().tolist() == [[0, 2, 0], [0, 4, 1]]
    assert result.offset_targets[result.offset_mask].tolist() == [-1.0, -1.0]
    assert result.event_count == 2
    assert result.shifted_event_count == 2
    assert result.total_abs_shift == 2


def test_hungarian_forbids_unknown_and_out_of_radius_high_score_frames() -> None:
    targets = torch.tensor([[0, 0, 1, 0, 0, 0]])
    validity, frame_mask = _all_valid(1, 6)
    frame_mask[0, 3] = False  # UNKNOWN despite being in the shift radius.
    logits = torch.zeros((1, 6, 3))
    logits[..., 0] = 2.0
    logits[0, 1] = torch.tensor([-5.0, 5.0, -5.0])  # eligible
    logits[0, 3] = torch.tensor([-9.0, 9.0, -9.0])  # UNKNOWN
    logits[0, 5] = torch.tensor([-10.0, 10.0, -10.0])  # outside radius

    result = dynamic_label_assignment(
        logits,
        targets,
        validity,
        frame_mask,
        mode="hungarian",
        max_shift_frames=1,
        temporal_cost_weight=0.0,
    )

    assert result.assigned_targets.tolist() == [[0, 1, 0, 0, 0, 0]]
    assert result.dense_targets[0, 3].sum() == 0
    assert result.total_abs_shift == 1


def test_hungarian_exact_ties_repeat_deterministically() -> None:
    targets = torch.tensor([[0, 1, 0, 2, 0]])
    validity, frame_mask = _all_valid(1, 5)
    logits = torch.zeros((1, 5, 3))

    baseline = dynamic_label_assignment(
        logits,
        targets,
        validity,
        frame_mask,
        mode="hungarian",
        max_shift_frames=2,
        class_cost_weight=0.0,
        temporal_cost_weight=0.0,
    )
    for _ in range(5):
        repeated = dynamic_label_assignment(
            logits,
            targets,
            validity,
            frame_mask,
            mode="hungarian",
            max_shift_frames=2,
            class_cost_weight=0.0,
            temporal_cost_weight=0.0,
        )
        assert torch.equal(repeated.assigned_targets, baseline.assigned_targets)
        assert torch.equal(repeated.offset_targets, baseline.offset_targets)
        assert repeated.total_abs_shift == baseline.total_abs_shift


def test_dense_cross_entropy_applies_soft_class_sample_frame_and_validity_masks() -> None:
    logits = torch.tensor(
        [
            [[0.2, 0.9, -0.4], [1.1, -0.3, 0.1]],
            [[-0.2, 0.3, 1.2], [0.4, 0.6, -0.5]],
        ],
        requires_grad=True,
    )
    dense = torch.tensor(
        [
            [[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]],
            [[0.25, 0.0, 0.75], [0.0, 1.0, 0.0]],
        ]
    )
    validity = torch.tensor([[True, True, False], [True, False, True]])
    frame_mask = torch.tensor([[True, False], [True, True]])
    classes = torch.tensor([1.0, 4.0, 2.0])
    samples = torch.tensor([2.0, 0.5])

    loss, numerators, normalizers = dense_cross_entropy(
        logits,
        dense,
        validity,
        frame_mask,
        class_weights=classes,
        sample_weights=samples,
    )

    masked_logits = logits.masked_fill(~validity[:, None, :], -1.0e4)
    log_probabilities = F.log_softmax(masked_logits, dim=-1)
    weighted_targets = dense * classes
    active = frame_mask & (
        (dense * (~validity[:, None, :])).sum(dim=-1) <= 1e-7
    )
    expected_frame_numerator = -(weighted_targets * log_probabilities).sum(-1)
    expected_frame_normalizer = weighted_targets.sum(-1)
    expected_numerators = (
        (expected_frame_numerator * active).sum(1) * samples
    )
    expected_normalizers = (
        (expected_frame_normalizer * active).sum(1) * samples
    )
    expected_loss = expected_numerators.sum() / expected_normalizers.sum()

    assert torch.allclose(numerators, expected_numerators)
    assert torch.allclose(normalizers, expected_normalizers)
    assert torch.allclose(loss, expected_loss)
    loss.backward()
    assert torch.count_nonzero(logits.grad[0, 1]) == 0
    # Sample 1 frame 1 targets an invalid class and is excluded as a whole.
    assert torch.count_nonzero(logits.grad[1, 1]) == 0


def test_subframe_offset_targets_and_masked_smooth_l1() -> None:
    targets = torch.tensor([[0, 1, 0, 0, 2]])
    validity, frame_mask = _all_valid(1, 5)
    logits = torch.zeros((1, 5, 3))
    logits[0, 2, 1] = 8.0
    logits[0, 3, 2] = 8.0
    subframes = torch.tensor([[0.0, 0.25, 0.0, 0.0, -0.2]])
    assignment = dynamic_label_assignment(
        logits,
        targets,
        validity,
        frame_mask,
        subframes,
        mode="hungarian",
        max_shift_frames=1,
        temporal_cost_weight=0.01,
    )
    assert assignment.offset_targets[0, 2, 0].item() == -0.75
    assert torch.isclose(
        assignment.offset_targets[0, 3, 1], torch.tensor(0.8), atol=1e-6
    )

    predictions = torch.zeros((1, 5, 2), requires_grad=True)
    loss, numerators, normalizers = offset_smooth_l1(
        predictions,
        assignment.offset_targets,
        assignment.offset_mask,
        sample_weights=torch.tensor([0.5]),
        beta=1.0,
    )
    expected_elements = F.smooth_l1_loss(
        predictions,
        assignment.offset_targets,
        reduction="none",
        beta=1.0,
    )
    expected_numerator = expected_elements[assignment.offset_mask].sum() * 0.5
    assert torch.allclose(numerators, expected_numerator[None])
    assert torch.equal(normalizers, torch.tensor([1.0]))
    assert torch.allclose(loss, expected_numerator)
    loss.backward()
    assert torch.count_nonzero(predictions.grad) == 2


def test_offset_smooth_l1_empty_mask_returns_differentiable_zero() -> None:
    predictions = torch.randn((2, 3, 2), requires_grad=True)
    targets = torch.zeros_like(predictions)
    mask = torch.zeros_like(predictions, dtype=torch.bool)

    loss, numerators, normalizers = offset_smooth_l1(predictions, targets, mask)

    assert loss.item() == 0.0
    assert torch.equal(numerators, torch.zeros(2))
    assert torch.equal(normalizers, torch.zeros(2))
    loss.backward()
    assert torch.equal(predictions.grad, torch.zeros_like(predictions))
