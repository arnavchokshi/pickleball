from __future__ import annotations

import pytest

from threed.racketsport.court_keypoint_geometric_loss import (
    COURT_LINE_GROUPS,
    GROUND_PLANE_KEYPOINT_NAMES,
    NET_KEYPOINT_NAMES,
    court_colinearity_loss,
    court_geometric_consistency_loss,
    court_homography_self_consistency_loss,
    court_layout_spread_guard,
    court_line_group_indices,
    derive_court_line_groups,
    soft_argmax_keypoints,
)
from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS, keypoint_labels_from_court_corners

KEYPOINT_NAMES = [point.name for point in PICKLEBALL_KEYPOINTS]


def _perfect_homography_points() -> dict[str, list[float]]:
    """A full 15-point layout that is, by construction, exactly explained by ONE planar
    homography (the same primitive `keypoint_labels_from_court_corners` uses to expand 4
    labeled court corners into the full canonical taxonomy elsewhere in this codebase)."""

    return keypoint_labels_from_court_corners(
        {
            "near_left": [100.0, 900.0],
            "near_right": [900.0, 900.0],
            "far_right": [700.0, 100.0],
            "far_left": [300.0, 100.0],
        }
    )


def _points_tensor(labels: dict[str, list[float]], *, requires_grad: bool = False):
    torch = pytest.importorskip("torch")
    rows = [labels[name] for name in KEYPOINT_NAMES]
    return torch.tensor([rows], dtype=torch.float32, requires_grad=requires_grad)


def test_derive_court_line_groups_matches_canonical_court_structure_with_net_exclusions() -> None:
    groups = derive_court_line_groups()

    assert groups == COURT_LINE_GROUPS
    assert len(groups) == 8

    # 3 world-X groups (sidelines/centerline) at 4 points each -- the net points are EXCLUDED
    # from these (owner labels place them at net TOP, off the ground lines' image projection,
    # per the CAL-METRIC finding); 5 world-Y groups (baselines/NVZ/net) at 3 points each.
    sizes = sorted(len(group) for group in groups)
    assert sizes == [3, 3, 3, 3, 3, 4, 4, 4]

    group_sets = [set(group) for group in groups]
    assert {"near_left_corner", "near_nvz_left", "far_nvz_left", "far_left_corner"} in group_sets
    assert {"near_baseline_center", "near_nvz_center", "far_nvz_center", "far_baseline_center"} in group_sets
    assert {"near_right_corner", "near_nvz_right", "far_nvz_right", "far_right_corner"} in group_sets
    assert {"near_left_corner", "near_baseline_center", "near_right_corner"} in group_sets
    assert {"far_left_corner", "far_baseline_center", "far_right_corner"} in group_sets
    assert {"near_nvz_left", "near_nvz_center", "near_nvz_right"} in group_sets
    assert {"far_nvz_left", "far_nvz_center", "far_nvz_right"} in group_sets
    # The net group itself SURVIVES: the net-top edge is a straight 3D line, so the 3 net
    # points stay collinear in image space at either height convention.
    assert {"net_left_sideline", "net_center", "net_right_sideline"} in group_sets

    # Membership: net points appear in exactly 1 group (their own net line); every other
    # keypoint appears in exactly 2 (one world-X group, one world-Y group).
    membership_count: dict[str, int] = {}
    for group in groups:
        for name in group:
            membership_count[name] = membership_count.get(name, 0) + 1
    assert set(membership_count) == set(KEYPOINT_NAMES)
    for name, count in membership_count.items():
        assert count == (1 if name in NET_KEYPOINT_NAMES else 2), name


def test_ground_plane_keypoint_names_exclude_exactly_the_net_points() -> None:
    assert set(GROUND_PLANE_KEYPOINT_NAMES) == set(KEYPOINT_NAMES) - NET_KEYPOINT_NAMES
    assert len(GROUND_PLANE_KEYPOINT_NAMES) == 12


def test_court_line_group_indices_resolves_positions_for_a_given_channel_order() -> None:
    reversed_names = list(reversed(KEYPOINT_NAMES))

    indices = court_line_group_indices(reversed_names)

    assert len(indices) == 8
    # Every resolved index set, mapped back through reversed_names, must reproduce one of the
    # known COURT_LINE_GROUPS name-sets -- i.e. index resolution is correct regardless of
    # channel ordering.
    resolved_name_sets = [frozenset(reversed_names[idx] for idx in group) for group in indices]
    expected_name_sets = [frozenset(group) for group in COURT_LINE_GROUPS]
    assert sorted(resolved_name_sets, key=lambda s: (len(s), sorted(s))) == sorted(
        expected_name_sets, key=lambda s: (len(s), sorted(s))
    )


def test_soft_argmax_keypoints_recovers_a_strong_known_peak() -> None:
    torch = pytest.importorskip("torch")
    height, width = 12, 16
    logits = torch.full((1, 2, height, width), -20.0)
    logits[0, 0, 5, 9] = 20.0
    logits[0, 1, 2, 3] = 20.0

    decoded = soft_argmax_keypoints(logits)

    assert tuple(decoded.shape) == (1, 2, 2)
    assert decoded[0, 0, 0].item() == pytest.approx(9.0, abs=0.05)
    assert decoded[0, 0, 1].item() == pytest.approx(5.0, abs=0.05)
    assert decoded[0, 1, 0].item() == pytest.approx(3.0, abs=0.05)
    assert decoded[0, 1, 1].item() == pytest.approx(2.0, abs=0.05)


def test_colinearity_loss_is_near_zero_for_a_perfect_homography_layout() -> None:
    pytest.importorskip("torch")
    points = _points_tensor(_perfect_homography_points())

    loss = court_colinearity_loss(points, keypoint_names=KEYPOINT_NAMES)

    assert loss.item() == pytest.approx(0.0, abs=1e-4)


def test_colinearity_loss_penalizes_a_perturbed_point_with_a_localized_gradient() -> None:
    torch = pytest.importorskip("torch")
    labels = _perfect_homography_points()
    labels["near_nvz_center"] = [labels["near_nvz_center"][0] + 25.0, labels["near_nvz_center"][1] - 15.0]
    points = _points_tensor(labels, requires_grad=True)

    loss = court_colinearity_loss(points, keypoint_names=KEYPOINT_NAMES)
    assert loss.item() > 1e-4

    loss.backward()
    grad = points.grad[0]

    # near_nvz_center's two groups: the centerline world-X group (near_baseline_center,
    # near_nvz_center, far_nvz_center, far_baseline_center -- net_center excluded per the
    # CAL-METRIC net-top finding) and the near-NVZ world-Y group (near_nvz_left,
    # near_nvz_center, near_nvz_right).
    affected = {
        "near_nvz_center",
        "near_baseline_center",
        "far_nvz_center",
        "far_baseline_center",
        "near_nvz_left",
        "near_nvz_right",
    }
    unaffected = set(KEYPOINT_NAMES) - affected

    # The perturbed keypoint's own two line groups must show a nonzero gradient response;
    # every keypoint outside those two groups shares no group with near_nvz_center at all, so
    # the colinearity loss must have EXACTLY zero gradient there -- this is the "localized
    # gradient" property the CAL-R2 geometric loss is designed to have.
    assert grad[KEYPOINT_NAMES.index("near_nvz_center")].abs().sum().item() > 0.0
    for name in unaffected:
        assert grad[KEYPOINT_NAMES.index(name)].abs().sum().item() == pytest.approx(0.0, abs=1e-8), name
    assert any(grad[KEYPOINT_NAMES.index(name)].abs().sum().item() > 0.0 for name in affected if name != "near_nvz_center")


def test_net_top_displacement_is_not_penalized_by_colinearity_or_homography_terms() -> None:
    """CAL-METRIC net-line finding: owner labels place the 3 net keypoints at the net TOP
    (~0.9m above the court plane). A correct net-top prediction is displaced off the ground
    sideline/centerline image lines and off the ground-plane homography -- the loss must NOT
    penalize that. Model it here by shifting all 3 net points up by the same image offset
    (which preserves their own collinearity, as the real net-top edge does)."""
    pytest.importorskip("torch")
    labels = _perfect_homography_points()
    for name in ("net_left_sideline", "net_center", "net_right_sideline"):
        labels[name] = [labels[name][0], labels[name][1] - 55.0]
    points = _points_tensor(labels)

    colinearity = court_colinearity_loss(points, keypoint_names=KEYPOINT_NAMES)
    homography = court_homography_self_consistency_loss(
        points, keypoint_names=KEYPOINT_NAMES, image_width=1000.0, image_height=1000.0
    )

    assert colinearity.item() == pytest.approx(0.0, abs=1e-4)
    assert homography.item() == pytest.approx(0.0, abs=1e-6)


def test_homography_self_consistency_loss_is_near_zero_for_a_perfect_layout_and_positive_when_perturbed() -> None:
    torch = pytest.importorskip("torch")
    perfect_points = _points_tensor(_perfect_homography_points())

    perfect_loss = court_homography_self_consistency_loss(
        perfect_points, keypoint_names=KEYPOINT_NAMES, image_width=1000.0, image_height=1000.0
    )
    assert perfect_loss.item() == pytest.approx(0.0, abs=1e-6)

    labels = _perfect_homography_points()
    labels["near_nvz_right"] = [labels["near_nvz_right"][0] - 60.0, labels["near_nvz_right"][1] + 40.0]
    perturbed_points = _points_tensor(labels, requires_grad=True)
    perturbed_loss = court_homography_self_consistency_loss(
        perturbed_points, keypoint_names=KEYPOINT_NAMES, image_width=1000.0, image_height=1000.0
    )
    assert perturbed_loss.item() > 1e-6

    perturbed_loss.backward()
    assert perturbed_points.grad is not None
    assert perturbed_points.grad.abs().sum().item() > 0.0
    # Net points are excluded from the ground-plane homography term entirely, so they must
    # receive exactly zero gradient from it.
    for name in NET_KEYPOINT_NAMES:
        assert perturbed_points.grad[0, KEYPOINT_NAMES.index(name)].abs().sum().item() == pytest.approx(0.0, abs=1e-9)


def test_homography_self_consistency_loss_skips_degenerate_rows_without_raising() -> None:
    torch = pytest.importorskip("torch")
    degenerate = torch.zeros((1, len(KEYPOINT_NAMES), 2), dtype=torch.float32, requires_grad=True)

    loss = court_homography_self_consistency_loss(
        degenerate, keypoint_names=KEYPOINT_NAMES, image_width=200.0, image_height=100.0
    )

    assert loss.item() == pytest.approx(0.0)


def test_spread_guard_penalizes_point_collapse_and_line_collapse_but_not_a_real_layout() -> None:
    """Adversarial diff review finding 5 (review_diff_20260702.md, MEDIUM): collapsed layouts
    trivially satisfy colinearity (0.0) and silently skip the degenerate homography fit (0.0).
    The spread guard must make them strictly positive instead."""
    torch = pytest.importorskip("torch")

    # Full point-collapse: all 15 points on one image point.
    collapsed = torch.full((1, len(KEYPOINT_NAMES), 2), 50.0, requires_grad=True)
    collapsed_guard = court_layout_spread_guard(collapsed, image_width=160.0, image_height=90.0)
    assert collapsed_guard.item() == pytest.approx(1.0, abs=1e-2)
    collapsed_guard.backward()
    assert collapsed.grad is not None

    # Line-collapse: all 15 points on one image line (large extent along it, zero across it).
    line = torch.stack(
        [torch.linspace(0.0, 150.0, len(KEYPOINT_NAMES)), torch.linspace(0.0, 80.0, len(KEYPOINT_NAMES))],
        dim=-1,
    ).unsqueeze(0)
    line = line.clone().requires_grad_(True)
    line_guard = court_layout_spread_guard(line, image_width=160.0, image_height=90.0)
    assert line_guard.item() > 0.9
    line_guard.backward()
    assert line.grad is not None
    assert line.grad.abs().sum().item() > 0.0

    # A legitimate perspective court layout has genuine 2D extent -> exactly zero guard.
    real_layout = _points_tensor(_perfect_homography_points())
    real_guard = court_layout_spread_guard(real_layout, image_width=1000.0, image_height=1000.0)
    assert real_guard.item() == pytest.approx(0.0)


def test_combined_geometric_loss_is_strictly_positive_for_collapsed_predictions() -> None:
    """End-to-end regression test for the collapse degeneracy: the COMBINED loss (as the
    trainer consumes it) must be strictly positive for a collapse that previously scored an
    exact 0.0 across every term (verified empirically before the guard was added: point-collapse
    gave colinearity 0.0 AND homography 0.0 because the degenerate DLT fit was silently
    skipped).

    Gradient note: at EXACTLY symmetric collapse (all points bit-identical) every smooth spread
    measure has a vanishing gradient by symmetry -- a measure-zero unstable critical point that
    real logits never sit on. The load-bearing property is a positive loss AT collapse plus a
    restoring gradient in its NEIGHBORHOOD, so this test uses near-collapse (a tiny asymmetric
    logit perturbation, as any real model state has) and asserts both."""
    torch = pytest.importorskip("torch")
    height, width = 18, 32
    # Near-uniform logits -> soft-argmax puts every keypoint within a fraction of a pixel of
    # the image center -> near-point-collapse (this is the model's state at initialization, so
    # the guard is active, and the geometric loss non-silent, from the first epoch onward).
    base = torch.zeros((1, len(KEYPOINT_NAMES), height, width))
    generator = torch.Generator().manual_seed(13)
    base = base + 0.01 * torch.randn(base.shape, generator=generator)
    logits = base.clone().requires_grad_(True)

    result = court_geometric_consistency_loss(
        logits,
        keypoint_names=KEYPOINT_NAMES,
        image_width=float(width),
        image_height=float(height),
    )

    assert result["colinearity"].item() < 0.05
    assert result["spread_guard"].item() > 0.9
    assert result["loss"].item() > 0.9
    result["loss"].backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum().item() > 0.0

    # And the exact-collapse value itself (no gradient assertion there -- see docstring):
    exact = court_geometric_consistency_loss(
        torch.zeros((1, len(KEYPOINT_NAMES), height, width)),
        keypoint_names=KEYPOINT_NAMES,
        image_width=float(width),
        image_height=float(height),
    )
    assert exact["colinearity"].item() == pytest.approx(0.0, abs=1e-6)
    assert exact["homography"].item() == pytest.approx(0.0, abs=1e-6)
    assert exact["spread_guard"].item() > 0.9
    assert exact["loss"].item() > 0.9


def test_court_geometric_consistency_loss_combines_terms_and_is_differentiable_through_logits() -> None:
    torch = pytest.importorskip("torch")
    height, width = 18, 32
    labels = _perfect_homography_points()
    # Scale corner coordinates (originally in a 1000x1000-ish space) down into the small test
    # heatmap's pixel grid.
    scaled = {name: [xy[0] * width / 1000.0, xy[1] * height / 1000.0] for name, xy in labels.items()}
    logits = torch.zeros((1, len(KEYPOINT_NAMES), height, width), requires_grad=True)
    with torch.no_grad():
        base = logits.clone()
        for idx, name in enumerate(KEYPOINT_NAMES):
            x, y = scaled[name]
            xi, yi = int(round(min(max(x, 0), width - 1))), int(round(min(max(y, 0), height - 1)))
            base[0, idx] = -15.0
            base[0, idx, yi, xi] = 15.0
    logits = base.clone().requires_grad_(True)

    result = court_geometric_consistency_loss(
        logits,
        keypoint_names=KEYPOINT_NAMES,
        image_width=float(width),
        image_height=float(height),
        colinearity_weight=1.0,
        homography_weight=1.0,
    )

    assert set(result) == {"loss", "colinearity", "homography", "spread_guard", "predicted_points"}
    assert result["loss"].item() >= 0.0
    assert result["colinearity"].item() == pytest.approx(0.0, abs=0.05)
    assert result["homography"].item() == pytest.approx(0.0, abs=0.05)
    assert result["spread_guard"].item() == pytest.approx(0.0, abs=0.05)

    result["loss"].backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum().item() > 0.0
