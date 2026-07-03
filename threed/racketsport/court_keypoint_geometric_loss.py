"""CAL-R2 PnLCalib-style point+line geometric-consistency loss.

Round 1 (`runs/cal_external_retrain_20260702T003120Z/REPORT.md`) trained the court-keypoint
heatmap net with `court_keypoint_heatmap_loss` alone: a per-channel spatial-softmax
cross-entropy, so each of the 15 canonical court keypoints is supervised completely
independently of the other 14. The round-1 owner-clip gate failed decisively (pooled PCK@5px
0.0167) and the diagnostic evidence was architectural, not "needs more data": the model's own
predicted 15-point layout was not even internally consistent as one planar court on unseen
cameras (`_homography_self_consistency_px` in `scripts/racketsport/train_court_keypoint_heatmap.py`
measured a 120-420px median self-fit residual across the 4 owner clips). A point-only loss has
no mechanism to prevent that: nothing tells the net that these 15 points must lie on a single
rigid, known court.

This module adds that missing structure as differentiable regularizers, all computed purely
from the model's own predictions (soft-argmax decoded from the raw heatmap logits, never from
ground truth -- they are unsupervised structural priors on prediction geometry, applicable to
every row regardless of whether that row happens to have full ground truth):

1. `court_colinearity_loss` -- known-collinear keypoint groups derived programmatically from
   `PICKLEBALL_KEYPOINTS`' world coordinates via `derive_court_line_groups` (never hand-listed,
   so this can never drift out of sync with the canonical taxonomy). Collinearity in the world
   plane survives ANY planar homography, so these groups must stay collinear in image space too
   for any geometrically valid camera, independent of viewpoint. Penalizes the fraction of a
   group's positional variance that falls perpendicular to its own best-fit line -- 0.0 for a
   perfect line, growing toward 0.5 for an isotropic (non-line-like) scatter. This fraction is
   scale-invariant (a variance ratio, not raw px^2), so near/far and wide/narrow line groups
   combine into one number without extra per-group normalization.

2. `court_homography_self_consistency_loss` -- for each row, fits ONE homography via a
   stop-gradient DLT solve (`threed.racketsport.court_calibration.homography_from_planar_points`,
   the exact pure-Python DLT the round-1 gate script already used as a read-only diagnostic) from
   that row's own DETACHED predicted GROUND-PLANE points to the known regulation-court world XY,
   then penalizes the (differentiable) squared reprojection residual between the row's LIVE
   predicted points and that fixed homography's projection of the same world points, normalized
   by the image diagonal squared (again for scale-invariance). Gradient flows only through the
   live predicted points; the homography fit itself receives no gradient (stop-gradient DLT, per
   the CAL-R2 task brief) -- this is required because the DLT solve is a non-differentiable
   pure-Python linear solve, and is also what keeps the term well-behaved (no need to
   differentiate through a linear system solve).

3. `court_layout_spread_guard` -- closes a real degeneracy the adversarial diff review flagged
   (`runs/manager/codex_lanes/reports/review_diff_20260702.md` finding 5, MEDIUM; verified
   empirically in this session before fixing): a fully collapsed prediction (all points on one
   image point) scores EXACTLY ZERO on both terms above -- colinearity is trivially satisfied by
   coincident points, and the DLT homography fit degenerates, raises `ValueError`, and was
   silently skipped (contributing nothing). The guard penalizes the predicted layout's smallest
   covariance eigenvalue falling below a minimum fraction of the image diagonal: a real court
   seen by any valid camera occupies genuine 2D extent in the image, so both point-collapse
   (both eigenvalues ~0) and line-collapse (smallest eigenvalue ~0) are penalized, while
   legitimate perspective layouts (even very oblique ones) sit above the threshold and receive
   exactly zero.

NET-LINE CAVEAT (CAL-METRIC lane finding, 2026-07-02, `runs/cal_metric_15pt_20260702T041729Z/`):
the owner CVAT labels place the 3 net keypoints (`net_left_sideline`, `net_center`,
`net_right_sideline`) at the TOP of the physical net (~0.9m above the court plane, 0.914m center
/ 0.991m posts per regulation), NOT at the Z=0 court-plane net line -- confirmed there by
systematic 2-6x leave-one-out residuals. A ground-plane homography cannot explain out-of-plane
points, so this module:

- EXCLUDES the 3 net points from the homography self-consistency term (12-point ground-plane
  homography, `GROUND_PLANE_KEYPOINT_NAMES`);
- EXCLUDES them from the sideline/centerline (world-X) colinearity groups -- a net-top point is
  displaced off the ground sideline's image line by the net's height;
- KEEPS the 3-point net colinearity group itself: the net-top edge is a straight 3D line, so its
  image stays a straight line under any perspective camera, exactly like the ground net line --
  the group is valid regardless of which height convention a given label source uses.

Label-source consistency note (recorded for the CAL-R2 report): the synthetic corpus generator
(`scripts/racketsport/generate_synthetic_court_keypoints.py`) both labels AND renders the net at
the ground plane -- it projects `PICKLEBALL_KEYPOINTS.world_xyz_m` (z=0 for net points) directly
and draws `court_templates.line_segments_m`'s z=0 net line, rendering no vertical net at all --
while the owner gate labels are net-top per the CAL-METRIC finding. That is a label-space
inconsistency on 3/15 keypoints that the loss-side exclusions above tolerate but cannot resolve;
unifying the convention (and rendering an actual vertical net in the synthetic generator) is
data work for the next round, not something a loss term can fix.

`court_geometric_consistency_loss` combines all three terms from raw heatmap logits into one
scalar, weighted by the caller (see `--geometric-loss-weight`, `--geometric-colinearity-weight`,
`--geometric-homography-weight` in `scripts/racketsport/train_court_keypoint_heatmap.py`).
"""

from __future__ import annotations

from typing import Any, Sequence

from threed.racketsport.court_calibration import homography_from_planar_points, project_planar_points
from threed.racketsport.court_keypoint_net import (
    PICKLEBALL_KEYPOINT_BY_NAME,
    PICKLEBALL_KEYPOINTS,
    court_keypoint_probabilities,
)

MIN_COLINEARITY_GROUP_SIZE = 3
MIN_HOMOGRAPHY_POINTS = 4
DEFAULT_MIN_SPREAD_FRACTION = 0.05

# The 3 keypoints the owner labels place at the TOP of the physical net (~0.9m above the court
# plane) rather than on Z=0 -- see the NET-LINE CAVEAT in the module docstring. Excluded from
# the ground-plane homography term and from world-X (sideline/centerline) colinearity groups.
NET_KEYPOINT_NAMES: frozenset[str] = frozenset({"net_left_sideline", "net_center", "net_right_sideline"})
GROUND_PLANE_KEYPOINT_NAMES: tuple[str, ...] = tuple(
    point.name for point in PICKLEBALL_KEYPOINTS if point.name not in NET_KEYPOINT_NAMES
)


def derive_court_line_groups(
    keypoints: Sequence[Any] = PICKLEBALL_KEYPOINTS,
    *,
    min_group_size: int = MIN_COLINEARITY_GROUP_SIZE,
    tol_m: float = 1e-6,
    exclude_from_x_groups: frozenset[str] = NET_KEYPOINT_NAMES,
) -> tuple[tuple[str, ...], ...]:
    """Group canonical court keypoints that share a world X or world Y coordinate.

    Any set of points sharing a world X (a sideline or the centerline) or world Y (a baseline,
    an NVZ line, or the net line) is collinear in the world plane by construction, and
    collinearity survives any planar homography -- so these groups must stay collinear in image
    space too, for any geometrically valid camera. Derived from `keypoints`' world coordinates
    (never hand-listed), so this always matches the canonical 15-point taxonomy, including if it
    is ever extended.

    ``exclude_from_x_groups`` (default: the 3 net keypoints) drops names from the world-X
    (sideline/centerline) groups only: labels that place net points at the net TOP (the owner
    CVAT convention, per the CAL-METRIC finding -- see the module docstring's NET-LINE CAVEAT)
    are displaced off the ground sideline/centerline image lines by the net's height, so
    including them would penalize CORRECT net-top predictions. They stay in their world-Y group
    (the net line itself), which is a straight 3D line at either height convention.

    For the current `PICKLEBALL_KEYPOINTS` taxonomy this yields 8 groups: 3 four-point world-X
    groups (left sideline x=-10ft, centerline x=0ft, right sideline x=10ft, each minus its net
    point) and 5 three-point world-Y groups (near baseline y=-22ft, near NVZ y=-7ft, net y=0ft,
    far NVZ y=7ft, far baseline y=22ft).
    """

    groups: list[tuple[str, ...]] = []
    for axis in (0, 1):
        buckets: dict[int, list[str]] = {}
        order: list[int] = []
        for point in keypoints:
            if axis == 0 and point.name in exclude_from_x_groups:
                continue
            key = round(point.world_xyz_m[axis] / tol_m)
            if key not in buckets:
                buckets[key] = []
                order.append(key)
            buckets[key].append(point.name)
        for key in sorted(order):
            names = buckets[key]
            if len(names) >= min_group_size:
                groups.append(tuple(names))
    return tuple(groups)


COURT_LINE_GROUPS: tuple[tuple[str, ...], ...] = derive_court_line_groups()


def court_line_group_indices(keypoint_names: Sequence[str]) -> tuple[tuple[int, ...], ...]:
    """Resolve `COURT_LINE_GROUPS` (keypoint names) into positional indices for a specific
    `keypoint_names` channel ordering (e.g. the trainer's heatmap channel order). Groups with
    fewer than `MIN_COLINEARITY_GROUP_SIZE` names resolvable in `keypoint_names` are dropped
    (defensive; not expected to trigger for the current canonical 15-point taxonomy, where every
    group name is always present)."""

    index_of = {name: index for index, name in enumerate(keypoint_names)}
    resolved: list[tuple[int, ...]] = []
    for group in COURT_LINE_GROUPS:
        indices = tuple(index_of[name] for name in group if name in index_of)
        if len(indices) >= MIN_COLINEARITY_GROUP_SIZE:
            resolved.append(indices)
    return tuple(resolved)


def soft_argmax_keypoints(logits: Any) -> Any:
    """Differentiable heatmap logits -> (..., K, 2) subpixel (x, y) decode via the spatial-softmax
    expectation over the pixel grid. Uses the same `spatial_softmax` normalization the trainer's
    supervised heatmap loss already applies (`court_keypoint_probabilities`), so the geometric
    loss reasons about the same probability surface the model is trained to produce -- it is not
    a separate, inconsistent decode path from the hard-argmax `decode_subpixel_heatmap` used at
    inference, just a differentiable expectation instead of a non-differentiable argmax."""

    import torch

    probs = court_keypoint_probabilities(logits, activation="spatial_softmax")
    height, width = probs.shape[-2:]
    xs = torch.arange(width, device=probs.device, dtype=probs.dtype)
    ys = torch.arange(height, device=probs.device, dtype=probs.dtype)
    x = (probs.sum(dim=-2) * xs).sum(dim=-1)
    y = (probs.sum(dim=-1) * ys).sum(dim=-1)
    return torch.stack([x, y], dim=-1)


def _covariance_eigenvalues(points: Any) -> tuple[Any, Any]:
    """(..., n, 2) -> ((...,), (...,)) smallest and largest eigenvalues of the 2x2 positional
    covariance matrix (closed-form; no `torch.linalg` needed for 2x2). A small epsilon inside
    the sqrt keeps gradients finite at exact degeneracy."""

    import torch

    centered = points - points.mean(dim=-2, keepdim=True)
    cov_xx = (centered[..., 0] ** 2).mean(dim=-1)
    cov_yy = (centered[..., 1] ** 2).mean(dim=-1)
    cov_xy = (centered[..., 0] * centered[..., 1]).mean(dim=-1)
    trace = cov_xx + cov_yy
    half_diff = (cov_xx - cov_yy) / 2.0
    radius = torch.sqrt(half_diff**2 + cov_xy**2 + 1e-9)
    smallest = ((trace / 2.0) - radius).clamp_min(0.0)
    largest = (trace / 2.0) + radius
    return smallest, largest


def _colinearity_fraction(points: Any) -> Any:
    """(..., n, 2) -> (...,) fraction of positional variance perpendicular to the group's own
    best-fit line: the smallest eigenvalue of the 2x2 covariance matrix, divided by its trace.
    0.0 for perfectly collinear points; up to 0.5 for an isotropic (non-line-like) scatter."""

    smallest, largest = _covariance_eigenvalues(points)
    trace = smallest + largest
    return smallest / trace.clamp_min(1e-6)


def court_colinearity_loss(
    predicted_points: Any,
    *,
    keypoint_names: Sequence[str],
    line_groups: tuple[tuple[int, ...], ...] | None = None,
) -> Any:
    """`predicted_points`: (batch, K, 2) soft-argmax-decoded predictions in model-input pixel
    space. Returns the mean, over the canonical court-line groups (see `derive_court_line_groups`,
    including its net-point exclusions) and over the batch, of each group's colinearity-violation
    fraction (`_colinearity_fraction`). Zero iff every group is perfectly collinear on every row.

    NOTE: trivially zero for fully collapsed (coincident-point) predictions -- that degeneracy is
    intentionally handled by `court_layout_spread_guard`, not here."""

    import torch

    groups = line_groups if line_groups is not None else court_line_group_indices(keypoint_names)
    if not groups:
        return predicted_points.new_zeros(())
    per_group = [_colinearity_fraction(predicted_points[..., list(group), :]) for group in groups]
    return torch.stack(per_group, dim=0).mean()


def court_homography_self_consistency_loss(
    predicted_points: Any,
    *,
    keypoint_names: Sequence[str],
    image_width: float,
    image_height: float,
    min_points: int = MIN_HOMOGRAPHY_POINTS,
    ground_plane_names: Sequence[str] = GROUND_PLANE_KEYPOINT_NAMES,
) -> Any:
    """`predicted_points`: (batch, K, 2). For each row, fits ONE stop-gradient DLT homography
    from that row's own DETACHED predicted ground-plane points (the 12 non-net keypoints by
    default -- see the module docstring's NET-LINE CAVEAT for why the 3 net-top points are
    excluded from a Z=0 planar fit) to the known regulation-court world XY, then returns the
    mean (over rows with a successful fit) of the squared reprojection residual between the
    row's LIVE predicted points and the same (constant) homography's projection of the same
    world points, normalized by the image diagonal squared (px^2 / px^2, so this stays a
    comparable-scale fraction to `court_colinearity_loss` regardless of the model's input
    resolution).

    Gradient flows only through `predicted_points` -- the homography fit itself never receives
    gradient (stop-gradient DLT, per the CAL-R2 task brief); `threed.racketsport.court_calibration
    .homography_from_planar_points` is a pure-Python linear solve with no autograd support
    anyway, so this is also simply how the fit has to be used.

    Rows whose detached predictions are too degenerate for a homography fit (e.g. near-identical
    points early in training) are skipped here (contribute nothing), matching the read-only
    `_homography_self_consistency_px` diagnostic's `except ValueError` behavior in
    `scripts/racketsport/train_court_keypoint_heatmap.py` -- but such rows are NOT free overall:
    `court_layout_spread_guard` exists precisely to penalize them (adversarial diff review
    finding 5).
    """

    import torch

    ground_plane_set = set(ground_plane_names)
    names = [name for name in keypoint_names if name in PICKLEBALL_KEYPOINT_BY_NAME and name in ground_plane_set]
    if len(names) < min_points:
        return predicted_points.new_zeros(())
    world_xy = [
        (PICKLEBALL_KEYPOINT_BY_NAME[name].world_xyz_m[0], PICKLEBALL_KEYPOINT_BY_NAME[name].world_xyz_m[1])
        for name in names
    ]
    keypoint_name_list = list(keypoint_names)
    name_indices = [keypoint_name_list.index(name) for name in names]

    diag2 = float(image_width) ** 2 + float(image_height) ** 2
    if diag2 <= 0:
        raise ValueError("image_width/image_height must be positive")

    detached = predicted_points.detach()
    batch = predicted_points.shape[0]
    residual_terms = []
    for row in range(batch):
        image_xy_detached = [tuple(detached[row, idx].tolist()) for idx in name_indices]
        try:
            homography = homography_from_planar_points(world_xy, image_xy_detached)
            projected = project_planar_points(homography, world_xy)
        except ValueError:
            continue
        projected_tensor = torch.tensor(projected, dtype=predicted_points.dtype, device=predicted_points.device)
        live_points = predicted_points[row, name_indices, :]
        residual_terms.append((((live_points - projected_tensor) ** 2).sum(dim=-1).mean()) / diag2)
    if not residual_terms:
        return predicted_points.new_zeros(())
    return torch.stack(residual_terms).mean()


def court_layout_spread_guard(
    predicted_points: Any,
    *,
    image_width: float,
    image_height: float,
    min_spread_fraction: float = DEFAULT_MIN_SPREAD_FRACTION,
) -> Any:
    """Degenerate-layout guard (adversarial diff review finding 5): penalize predicted layouts
    whose smallest positional-covariance eigenvalue's standard deviation falls below
    ``min_spread_fraction`` of the image diagonal.

    `predicted_points`: (batch, K, 2). Per row, computes ``spread = sqrt(lambda_min(cov))``
    (the layout's extent along its NARROWEST direction, in px) and returns the mean over the
    batch of the linear hinge ``relu(1 - spread / (min_spread_fraction * diag))``:

    - full point-collapse (all K points coincident): spread 0 -> guard 1.0 per row (maximal);
    - line-collapse (all K points on one image line): lambda_min ~ 0 -> guard ~ 1.0;
    - any legitimate court layout: a real court seen by any valid camera has genuine 2D image
      extent well above ``min_spread_fraction`` (default 5%) of the diagonal even at oblique
      angles, so the hinge is exactly 0 there and the guard adds no gradient at all.

    Together with `court_colinearity_loss` (trivially zero at collapse) and
    `court_homography_self_consistency_loss` (skips degenerate rows), this makes the combined
    geometric loss strictly positive for collapsed/degenerate predictions instead of silently
    zero.
    """

    import torch

    if min_spread_fraction <= 0:
        raise ValueError("min_spread_fraction must be positive")
    diag = (float(image_width) ** 2 + float(image_height) ** 2) ** 0.5
    if diag <= 0:
        raise ValueError("image_width/image_height must be positive")
    smallest, _ = _covariance_eigenvalues(predicted_points)
    spread = torch.sqrt(smallest + 1e-9)
    hinge = (1.0 - spread / (min_spread_fraction * diag)).clamp_min(0.0)
    return hinge.mean()


def court_geometric_consistency_loss(
    logits: Any,
    *,
    keypoint_names: Sequence[str],
    image_width: float,
    image_height: float,
    colinearity_weight: float = 1.0,
    homography_weight: float = 1.0,
    spread_guard_weight: float = 1.0,
    min_spread_fraction: float = DEFAULT_MIN_SPREAD_FRACTION,
) -> dict[str, Any]:
    """Full CAL-R2 point+line geometric-consistency loss, computed directly from raw heatmap
    logits (the trainer's model output before any activation). Returns a dict with the combined
    weighted scalar plus each unweighted component (for logging/diagnostics), so callers can
    report the breakdown without recomputing soft-argmax twice.
    """

    predicted_points = soft_argmax_keypoints(logits)
    colinearity = court_colinearity_loss(predicted_points, keypoint_names=keypoint_names)
    homography = court_homography_self_consistency_loss(
        predicted_points,
        keypoint_names=keypoint_names,
        image_width=image_width,
        image_height=image_height,
    )
    spread_guard = court_layout_spread_guard(
        predicted_points,
        image_width=image_width,
        image_height=image_height,
        min_spread_fraction=min_spread_fraction,
    )
    combined = colinearity_weight * colinearity + homography_weight * homography + spread_guard_weight * spread_guard
    return {
        "loss": combined,
        "colinearity": colinearity,
        "homography": homography,
        "spread_guard": spread_guard,
        "predicted_points": predicted_points,
    }
