from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from threed.racketsport.court_structured_model import STRUCTURED_FLOOR_KEYPOINT_COUNT  # noqa: E402
from threed.racketsport.court_structured_training import (  # noqa: E402
    project_homography,
    structured_floor_training_loss,
    structured_floor_world_xy,
    weighted_homography_dlt,
)


def test_weighted_dlt_recovers_known_projective_transform() -> None:
    world = structured_floor_world_xy(dtype=torch.float64)
    truth = torch.tensor(
        [[[42.0, 3.0, 320.0], [1.5, -31.0, 250.0], [0.003, -0.006, 1.0]]],
        dtype=torch.float64,
    )
    image = project_homography(truth, world)
    fit = weighted_homography_dlt(world, image, torch.ones((1, world.shape[0]), dtype=torch.float64))
    reconstructed = project_homography(fit, world)
    assert torch.max(torch.abs(reconstructed - image)).item() < 1e-6


def test_low_weight_outlier_does_not_move_weighted_fit_materially() -> None:
    world = structured_floor_world_xy(dtype=torch.float64)
    truth = torch.tensor(
        [[[35.0, 2.0, 300.0], [1.0, -28.0, 220.0], [0.002, -0.004, 1.0]]],
        dtype=torch.float64,
    )
    image = project_homography(truth, world)
    corrupted = image.clone()
    corrupted[0, 0] += torch.tensor([800.0, -500.0], dtype=torch.float64)
    weights = torch.ones((1, world.shape[0]), dtype=torch.float64)
    weights[0, 0] = 1e-8
    fit = weighted_homography_dlt(world, corrupted, weights)
    reconstructed = project_homography(fit, world)
    assert torch.median(torch.linalg.vector_norm(reconstructed - image, dim=-1)).item() < 1e-3


def test_structured_loss_backpropagates_through_heatmaps_and_covariance() -> None:
    batch = 1
    count = STRUCTURED_FLOOR_KEYPOINT_COUNT
    height, width = 24, 40
    heatmaps = torch.randn((batch, count, height, width), requires_grad=True)
    vis_logits = torch.zeros((batch, count), requires_grad=True)
    covariance_params = torch.zeros((batch, count, 3), requires_grad=True)
    sigma_x = torch.exp(covariance_params[..., 0])
    sigma_y = torch.exp(covariance_params[..., 1])
    covariance = torch.diag_embed(torch.stack((sigma_x.square(), sigma_y.square()), dim=-1))
    target = torch.stack(
        (
            torch.linspace(4.0, width - 5.0, count),
            torch.linspace(3.0, height - 4.0, count),
        ),
        dim=-1,
    ).unsqueeze(0)
    result = structured_floor_training_loss(
        {
            "keypoint_heatmaps": heatmaps,
            "keypoint_vis_logits": vis_logits,
            "keypoint_covariance": covariance,
        },
        target_xy_heatmap=target,
        target_mask=torch.ones((batch, count)),
    )
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert heatmaps.grad is not None and torch.isfinite(heatmaps.grad).all()
    assert vis_logits.grad is not None and torch.isfinite(vis_logits.grad).all()
    assert covariance_params.grad is not None and torch.isfinite(covariance_params.grad).all()


def test_rows_with_fewer_than_four_targets_skip_structured_solve() -> None:
    count = STRUCTURED_FLOOR_KEYPOINT_COUNT
    outputs = {
        "keypoint_heatmaps": torch.zeros((1, count, 8, 8), requires_grad=True),
        "keypoint_vis_logits": torch.zeros((1, count), requires_grad=True),
    }
    mask = torch.zeros((1, count))
    mask[:, :3] = 1.0
    result = structured_floor_training_loss(
        outputs,
        target_xy_heatmap=torch.zeros((1, count, 2)),
        target_mask=mask,
    )
    assert result["homography"] is None
    assert result["structured_reprojection_loss"].item() == 0.0
