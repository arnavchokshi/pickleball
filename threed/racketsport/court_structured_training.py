"""Differentiable floor-template fitting losses for ``court_structured_v3``.

The public court still comes from the robust deterministic solver.  These
helpers train the evidence network toward the same regulation-template result
by differentiating through a confidence-weighted normalized DLT solve.
"""

from __future__ import annotations

from typing import Any

from threed.racketsport.court_structured_model import (
    STRUCTURED_DISTANCE_SEGMENTS,
    STRUCTURED_FLOOR_KEYPOINT_NAMES,
    STRUCTURED_FLOOR_KEYPOINTS,
)


def structured_floor_world_xy(*, device: Any = None, dtype: Any = None) -> Any:
    import torch

    return torch.tensor(
        [[point.world_xyz_m[0], point.world_xyz_m[1]] for point in STRUCTURED_FLOOR_KEYPOINTS],
        device=device,
        dtype=dtype or torch.float32,
    )


def softargmax_points(logits: Any) -> Any:
    """Decode ``(B,K,H,W)`` logits to differentiable heatmap-pixel coordinates."""

    import torch

    if logits.ndim != 4:
        raise ValueError("heatmap logits must have shape (B,K,H,W)")
    batch, count, height, width = logits.shape
    probabilities = torch.softmax(logits.reshape(batch, count, -1), dim=-1).reshape_as(logits)
    xs = torch.arange(width, dtype=logits.dtype, device=logits.device)
    ys = torch.arange(height, dtype=logits.dtype, device=logits.device)
    x = (probabilities.sum(dim=-2) * xs).sum(dim=-1)
    y = (probabilities.sum(dim=-1) * ys).sum(dim=-1)
    return torch.stack((x, y), dim=-1)


def _normalization_transform(points: Any, weights: Any) -> tuple[Any, Any]:
    """Return normalized points and a batch of Hartley similarity transforms."""

    import torch

    safe_weights = weights.clamp_min(0.0)
    total = safe_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    centroid = (points * safe_weights[..., None]).sum(dim=-2) / total
    centered = points - centroid[..., None, :]
    distances = torch.sqrt(centered.square().sum(dim=-1) + 1e-12)
    mean_distance = (distances * safe_weights).sum(dim=-1) / total.squeeze(-1)
    scale = (2.0**0.5) / mean_distance.clamp_min(1e-4)

    zeros = torch.zeros_like(scale)
    ones = torch.ones_like(scale)
    transform = torch.stack(
        (
            scale,
            zeros,
            -scale * centroid[..., 0],
            zeros,
            scale,
            -scale * centroid[..., 1],
            zeros,
            zeros,
            ones,
        ),
        dim=-1,
    ).reshape(*scale.shape, 3, 3)
    homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
    normalized_h = torch.einsum("...ij,...nj->...ni", transform, homogeneous)
    return normalized_h[..., :2] / normalized_h[..., 2:].clamp_min(1e-8), transform


def weighted_homography_dlt(world_xy: Any, image_xy: Any, weights: Any) -> Any:
    """Differentiable confidence-weighted normalized DLT.

    ``world_xy`` may be ``(K,2)`` or ``(B,K,2)``; ``image_xy`` is ``(B,K,2)`` and
    ``weights`` is ``(B,K)``.  Callers must supply at least four non-collinear points.  Training
    batches with insufficient supervision should mask this loss rather than fabricate a solve.
    """

    import torch

    if image_xy.ndim != 3 or image_xy.shape[-1] != 2:
        raise ValueError("image_xy must have shape (B,K,2)")
    if weights.shape != image_xy.shape[:2]:
        raise ValueError("weights must have shape (B,K)")
    if world_xy.ndim == 2:
        world_xy = world_xy.unsqueeze(0).expand(image_xy.shape[0], -1, -1)
    if world_xy.shape != image_xy.shape:
        raise ValueError("world_xy and image_xy must resolve to the same shape")
    if image_xy.shape[1] < 4:
        raise ValueError("at least four points are required")

    world_n, world_t = _normalization_transform(world_xy, weights)
    image_n, image_t = _normalization_transform(image_xy, weights)
    x, y = world_n.unbind(dim=-1)
    u, v = image_n.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)
    # Fix normalized h33=1 and solve the remaining eight parameters by
    # regularized weighted least squares. This is the inhomogeneous form of
    # normalized DLT; unlike a homogeneous SVD, its gradient remains defined
    # while newly initialized auxiliary predictions are near-collapsed.
    row_u = torch.stack((x, y, ones, zeros, zeros, zeros, -u * x, -u * y), dim=-1)
    row_v = torch.stack((zeros, zeros, zeros, x, y, ones, -v * x, -v * y), dim=-1)
    target = torch.stack((u, v), dim=-1)
    positive_weight = (weights > 0.0).to(weights.dtype)
    sqrt_weights = (weights.clamp_min(0.0) + 1e-8).sqrt()[..., None]
    sqrt_weights = sqrt_weights * positive_weight[..., None]
    matrix = torch.stack((row_u * sqrt_weights, row_v * sqrt_weights), dim=-2).reshape(
        image_xy.shape[0], image_xy.shape[1] * 2, 8
    )
    right = (target * sqrt_weights)[..., None].reshape(
        image_xy.shape[0], image_xy.shape[1] * 2, 1
    )
    solve_dtype = torch.float64 if matrix.dtype == torch.float64 else torch.float32
    matrix_solve = matrix.to(solve_dtype)
    right_solve = right.to(solve_dtype)
    normal = matrix_solve.transpose(-1, -2) @ matrix_solve
    rhs = matrix_solve.transpose(-1, -2) @ right_solve
    ridge = 1e-12 if solve_dtype == torch.float64 else 1e-6
    identity = torch.eye(8, dtype=solve_dtype, device=matrix.device).expand(
        image_xy.shape[0], 8, 8
    )
    parameters = torch.linalg.solve(
        (normal + ridge * identity).to(solve_dtype),
        rhs.to(solve_dtype),
    ).squeeze(-1)
    final_one = torch.ones((image_xy.shape[0], 1), dtype=solve_dtype, device=matrix.device)
    normalized_h = torch.cat((parameters, final_one), dim=-1).reshape(-1, 3, 3)
    homography = (
        torch.linalg.inv(image_t.to(solve_dtype))
        @ normalized_h
        @ world_t.to(solve_dtype)
    )
    denominator = homography[..., 2:3, 2:3]
    signed_epsilon = torch.where(denominator < 0, -torch.ones_like(denominator), torch.ones_like(denominator))
    return homography / torch.where(denominator.abs() < 1e-8, signed_epsilon * 1e-8, denominator)


def project_homography(homography: Any, world_xy: Any) -> Any:
    import torch

    if world_xy.ndim == 2:
        world_xy = world_xy.unsqueeze(0).expand(homography.shape[0], -1, -1)
    homogeneous = torch.cat((world_xy, torch.ones_like(world_xy[..., :1])), dim=-1)
    projected = torch.einsum("bij,bnj->bni", homography, homogeneous)
    denominator = projected[..., 2:3]
    signed_epsilon = torch.where(denominator < 0, -torch.ones_like(denominator), torch.ones_like(denominator))
    return projected[..., :2] / torch.where(
        denominator.abs() < 1e-8, signed_epsilon * 1e-8, denominator
    )


def weighted_masked_mean(values: Any, mask: Any, sample_weights: Any | None = None) -> Any:
    """Return a finite weighted mean while preserving an all-masked zero.

    ``values`` and ``mask`` share a leading batch dimension.  ``sample_weights`` is a scalar per
    row and is broadcast across every remaining dimension.  This is the common mechanism used
    to keep external teacher rows at a bounded fraction of human/synthetic supervision.
    """

    weights = mask.to(values.dtype)
    if sample_weights is not None:
        if sample_weights.ndim != 1 or sample_weights.shape[0] != values.shape[0]:
            raise ValueError("sample_weights must have shape (B,)")
        weights = weights * sample_weights.to(values.dtype).reshape(
            values.shape[0], *((1,) * (values.ndim - 1))
        )
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def gaussian_point_nll(
    residual: Any,
    covariance: Any,
    mask: Any,
    *,
    sample_weights: Any | None = None,
) -> Any:
    """Masked 2-D Gaussian NLL used to make predicted covariance meaningful."""

    import torch

    covariance = covariance + torch.eye(2, device=covariance.device, dtype=covariance.dtype) * 1e-4
    # Closed-form SPD 2x2 algebra avoids backend-specific batched logdet/inverse
    # resizing bugs and is cheaper than a generic matrix factorization.
    a = covariance[..., 0, 0]
    b = covariance[..., 0, 1]
    c = covariance[..., 1, 0]
    d = covariance[..., 1, 1]
    determinant = (a * d - b * c).clamp_min(1e-8)
    dx = residual[..., 0]
    dy = residual[..., 1]
    mahalanobis = (d * dx.square() - (b + c) * dx * dy + a * dy.square()) / determinant
    logdet = torch.log(determinant)
    values = 0.5 * (mahalanobis + logdet)
    return weighted_masked_mean(values, mask, sample_weights)


def semantic_segment_distance_targets(
    keypoints_xy: Any,
    *,
    height: int,
    width: int,
    max_distance_px: float = 16.0,
) -> Any:
    """Build normalized dense distances for the eight semantic painted floor segments.

    Coordinates are in the same pixel space as the requested output grid.  Values are Euclidean
    point-to-segment distances, truncated at ``max_distance_px`` and normalized to ``[0, 1]``.
    Segment validity is intentionally separate from this pure geometry helper so human adapters
    can mask targets unless a regulation projection was anchored by enough reviewed points.
    """

    import torch

    if keypoints_xy.ndim != 3 or keypoints_xy.shape[-1] != 2:
        raise ValueError("keypoints_xy must have shape (B,K,2)")
    if keypoints_xy.shape[1] != len(STRUCTURED_FLOOR_KEYPOINT_NAMES):
        raise ValueError("keypoints_xy does not match the structured floor taxonomy")
    if isinstance(height, bool) or isinstance(width, bool) or height <= 0 or width <= 0:
        raise ValueError("height and width must be positive integers")
    if max_distance_px <= 0:
        raise ValueError("max_distance_px must be positive")

    name_to_index = {name: index for index, name in enumerate(STRUCTURED_FLOOR_KEYPOINT_NAMES)}
    ys = torch.arange(height, dtype=keypoints_xy.dtype, device=keypoints_xy.device)
    xs = torch.arange(width, dtype=keypoints_xy.dtype, device=keypoints_xy.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1)[None, :, :, :]
    distance_maps = []
    for _, start_name, end_name in STRUCTURED_DISTANCE_SEGMENTS:
        start = keypoints_xy[:, name_to_index[start_name]][:, None, None, :]
        end = keypoints_xy[:, name_to_index[end_name]][:, None, None, :]
        delta = end - start
        denominator = delta.square().sum(dim=-1, keepdim=True).clamp_min(1e-8)
        fraction = ((grid - start) * delta).sum(dim=-1, keepdim=True) / denominator
        closest = start + fraction.clamp(0.0, 1.0) * delta
        distance = torch.linalg.vector_norm(grid - closest, dim=-1)
        distance_maps.append(distance.clamp_max(float(max_distance_px)) / float(max_distance_px))
    return torch.stack(distance_maps, dim=1)


def structured_floor_training_loss(
    outputs: dict[str, Any],
    *,
    target_xy_heatmap: Any,
    target_mask: Any,
    direct_weight: float = 1.0,
    structured_weight: float = 1.0,
    confidence_weight: float = 0.1,
    sample_weights: Any | None = None,
) -> dict[str, Any]:
    """Compute direct evidence, structured reprojection, and covariance losses.

    Targets are expressed in heatmap pixels and ordered exactly as
    ``STRUCTURED_FLOOR_KEYPOINTS``.  Rows with fewer than four supervised floor points receive
    direct supervision but are excluded from the structured DLT term.
    """

    import torch
    import torch.nn.functional as F

    predicted = softargmax_points(outputs["keypoint_heatmaps"])
    if predicted.shape != target_xy_heatmap.shape or target_mask.shape != predicted.shape[:2]:
        raise ValueError("target shapes must match the structured floor taxonomy")
    mask = target_mask > 0
    valid = mask.to(predicted.dtype)
    direct_values = F.smooth_l1_loss(predicted, target_xy_heatmap, reduction="none").sum(dim=-1)
    direct = weighted_masked_mean(direct_values, valid, sample_weights)

    predicted_visibility = torch.sigmoid(outputs["keypoint_vis_logits"])
    solve_weights = predicted_visibility * valid
    enough = mask.sum(dim=-1) >= 4
    structured = predicted.new_zeros(())
    homography = None
    projected = predicted.clone()
    if bool(enough.any()):
        indexes = torch.nonzero(enough, as_tuple=False).flatten()
        world = structured_floor_world_xy(device=predicted.device, dtype=predicted.dtype)
        homography_valid = weighted_homography_dlt(
            world,
            predicted[indexes],
            solve_weights[indexes],
        )
        projected_valid = project_homography(homography_valid, world).to(predicted.dtype)
        projected = projected.clone()
        projected[indexes] = projected_valid
        structured_values = F.smooth_l1_loss(
            projected_valid, target_xy_heatmap[indexes], reduction="none"
        ).sum(dim=-1)
        structured_mask = valid[indexes]
        structured = weighted_masked_mean(
            structured_values,
            structured_mask,
            None if sample_weights is None else sample_weights[indexes],
        )
        homography = homography_valid

    covariance = outputs.get("keypoint_covariance")
    confidence_nll = predicted.new_zeros(())
    if covariance is not None:
        confidence_nll = gaussian_point_nll(
            predicted - target_xy_heatmap,
            covariance,
            mask,
            sample_weights=sample_weights,
        )

    total = direct_weight * direct + structured_weight * structured + confidence_weight * confidence_nll
    return {
        "loss": total,
        "direct_point_loss": direct,
        "structured_reprojection_loss": structured,
        "confidence_nll": confidence_nll,
        "predicted_points": predicted,
        "projected_points": projected,
        "homography": homography,
        "structured_row_mask": enough,
    }


__all__ = [
    "gaussian_point_nll",
    "project_homography",
    "softargmax_points",
    "semantic_segment_distance_targets",
    "structured_floor_training_loss",
    "structured_floor_world_xy",
    "weighted_masked_mean",
    "weighted_homography_dlt",
]
