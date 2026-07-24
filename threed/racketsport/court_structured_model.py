"""Trainable evidence network for confidence-aware structured court inference.

The model intentionally predicts *evidence*, not an unconstrained final court.
``court_structured_solver`` consumes the heatmap observations and produces the
single regulation-template hypothesis exposed to callers.

This module is additive: the selected ``court_unet_v2`` checkpoint and public
pipeline remain unchanged until a v3 checkpoint passes the documented gate.
"""

from __future__ import annotations

from typing import Any, Mapping

from threed.racketsport.court_keypoint_geometric_loss import NET_KEYPOINT_NAMES
from threed.racketsport.court_keypoint_net import (
    ALL_PICKLEBALL_KEYPOINTS,
    COURT_UNET_V2_HEATMAP_STRIDE,
    COURT_UNET_V2_SEG_CLASS_NAMES,
    _load_resnet34_encoder,
)


COURT_STRUCTURED_V3_ARCHITECTURE = "court_structured_v3"
STRUCTURED_FLOOR_KEYPOINTS = tuple(
    point for point in ALL_PICKLEBALL_KEYPOINTS if point.name not in NET_KEYPOINT_NAMES
)
STRUCTURED_FLOOR_KEYPOINT_NAMES = tuple(point.name for point in STRUCTURED_FLOOR_KEYPOINTS)
STRUCTURED_FLOOR_KEYPOINT_COUNT = len(STRUCTURED_FLOOR_KEYPOINT_NAMES)
# Dense distance supervision is semantic rather than a duplicate of the five-class
# segmentation head.  Each channel represents one regulation painted segment.  Keeping the
# service centerlines split is important: each terminates at its own NVZ and neither may cross
# the kitchen.  The net is deliberately absent because it is not a painted floor segment.
STRUCTURED_DISTANCE_SEGMENTS: tuple[tuple[str, str, str], ...] = (
    ("near_baseline", "near_left_corner", "near_right_corner"),
    ("far_baseline", "far_left_corner", "far_right_corner"),
    ("left_sideline", "near_left_corner", "far_left_corner"),
    ("right_sideline", "near_right_corner", "far_right_corner"),
    ("near_nvz", "near_nvz_left", "near_nvz_right"),
    ("far_nvz", "far_nvz_left", "far_nvz_right"),
    ("near_centerline", "near_baseline_center", "near_nvz_center"),
    ("far_centerline", "far_nvz_center", "far_baseline_center"),
)
STRUCTURED_DISTANCE_CLASS_NAMES = tuple(segment[0] for segment in STRUCTURED_DISTANCE_SEGMENTS)


def covariance_matrices_from_params(params: Any) -> Any:
    """Convert ``[..., 3]`` log-sigma/correlation parameters to positive covariance matrices.

    The first two channels are bounded log standard deviations in heatmap pixels.  The third is
    an unconstrained correlation logit.  Bounding the values keeps early training numerically
    stable without pretending the result is calibrated before a held-out calibration pass.
    """

    import torch

    if params.shape[-1] != 3:
        raise ValueError("covariance params must end in three channels")
    sigma_x = torch.exp(params[..., 0].clamp(-3.0, 4.0))
    sigma_y = torch.exp(params[..., 1].clamp(-3.0, 4.0))
    correlation = torch.tanh(params[..., 2]) * 0.95
    cov_xy = correlation * sigma_x * sigma_y
    row0 = torch.stack((sigma_x.square(), cov_xy), dim=-1)
    row1 = torch.stack((cov_xy, sigma_y.square()), dim=-1)
    return torch.stack((row0, row1), dim=-2)


def make_court_structured_v3_model(*, encoder_weights_path: Any = None) -> Any:
    """Build the v3 identity-conditioned evidence model.

    The output taxonomy is 30 floor points: 12 canonical planar points plus the existing 18
    auxiliary line points.  The three physical top-of-net points are deliberately absent from
    the planar head and will be handled by a separate 3-D net stage.
    """

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _UpBlock(nn.Module):
        def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
            super().__init__()
            self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            self.conv = nn.Sequential(
                nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: Any, skip: Any) -> Any:
            up = F.interpolate(self.reduce(x), size=skip.shape[-2:], mode="bilinear", align_corners=False)
            return self.conv(torch.cat((up, skip), dim=1))

    class CourtStructuredV3(nn.Module):
        heatmap_stride = COURT_UNET_V2_HEATMAP_STRIDE
        architecture = COURT_STRUCTURED_V3_ARCHITECTURE
        keypoint_names = STRUCTURED_FLOOR_KEYPOINT_NAMES
        seg_class_names = COURT_UNET_V2_SEG_CLASS_NAMES
        distance_class_names = STRUCTURED_DISTANCE_CLASS_NAMES

        def __init__(self) -> None:
            super().__init__()
            resnet = _load_resnet34_encoder(encoder_weights_path)
            self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4

            trunk_channels = 96
            self.up3 = _UpBlock(512, 256, 256)
            self.up2 = _UpBlock(256, 128, 128)
            self.up1 = _UpBlock(128, 64, trunk_channels)

            # Each semantic point owns a dynamic 1x1-filter query over the shared spatial map.
            self.keypoint_queries = nn.Parameter(torch.empty(STRUCTURED_FLOOR_KEYPOINT_COUNT, trunk_channels))
            self.keypoint_bias = nn.Parameter(torch.zeros(STRUCTURED_FLOOR_KEYPOINT_COUNT))
            nn.init.normal_(self.keypoint_queries, mean=0.0, std=0.02)

            self.seg_head = nn.Conv2d(trunk_channels, len(COURT_UNET_V2_SEG_CLASS_NAMES), kernel_size=1)
            self.distance_head = nn.Conv2d(trunk_channels, len(STRUCTURED_DISTANCE_CLASS_NAMES), kernel_size=1)
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.vis_head = nn.Linear(trunk_channels, STRUCTURED_FLOOR_KEYPOINT_COUNT)
            self.covariance_head = nn.Linear(trunk_channels, STRUCTURED_FLOOR_KEYPOINT_COUNT * 3)
            self.supported_view_head = nn.Linear(trunk_channels, 1)
            # These heads have no v2 counterpart. Neutral initialization keeps
            # warm-started feature magnitudes from becoming fake certainty on
            # the first optimization step: new auxiliary visibility starts low,
            # covariance at identity in heatmap pixels, and view support at 0.5.
            nn.init.zeros_(self.vis_head.weight)
            nn.init.constant_(self.vis_head.bias, -2.0)
            nn.init.zeros_(self.covariance_head.weight)
            nn.init.zeros_(self.covariance_head.bias)
            nn.init.zeros_(self.supported_view_head.weight)
            nn.init.zeros_(self.supported_view_head.bias)

        def forward(self, x: Any) -> dict[str, Any]:
            c1 = self.layer1(self.stem(x))
            c2 = self.layer2(c1)
            c3 = self.layer3(c2)
            c4 = self.layer4(c3)
            trunk = self.up1(self.up2(self.up3(c4, c3), c2), c1)

            # Preserve the exact v2 1x1-convolution math for warm-started
            # canonical queries. Cosine normalization would constrain logits
            # to [-1,1] and erase the selected v2 head's learned magnitude.
            heatmaps = torch.einsum("bchw,kc->bkhw", trunk, self.keypoint_queries)
            heatmaps = heatmaps + self.keypoint_bias[None, :, None, None]
            pooled = self.global_pool(trunk).flatten(1)
            covariance_params = self.covariance_head(pooled).reshape(
                x.shape[0], STRUCTURED_FLOOR_KEYPOINT_COUNT, 3
            )
            return {
                "keypoint_heatmaps": heatmaps,
                "keypoint_vis_logits": self.vis_head(pooled),
                "keypoint_covariance_params": covariance_params,
                "keypoint_covariance": covariance_matrices_from_params(covariance_params),
                "line_family_logits": self.seg_head(trunk),
                "line_distance_maps": F.softplus(self.distance_head(trunk)),
                "supported_view_logit": self.supported_view_head(pooled).squeeze(-1),
            }

    return CourtStructuredV3()


def initialize_structured_v3_from_v2(model: Any, v2_state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Warm-start shared v2 weights and canonical point queries without hiding partial loads.

    Returns explicit loaded/skipped key lists for checkpoint provenance.  Auxiliary queries,
    covariance, distance, and supported-view heads remain freshly initialized.
    """

    import torch

    target = model.state_dict()
    compatible: dict[str, Any] = {}
    skipped: list[str] = []
    for name, value in v2_state_dict.items():
        if name in target and tuple(value.shape) == tuple(target[name].shape):
            compatible[name] = value
        else:
            skipped.append(str(name))

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    initialized_queries: list[str] = []
    v2_weight = v2_state_dict.get("keypoint_head.weight")
    v2_bias = v2_state_dict.get("keypoint_head.bias")
    v2_vis_weight = v2_state_dict.get("vis_head.2.weight")
    v2_vis_bias = v2_state_dict.get("vis_head.2.bias")
    if isinstance(v2_weight, torch.Tensor) and v2_weight.ndim == 4 and v2_weight.shape[-2:] == (1, 1):
        canonical_index = {
            point.name: index
            for index, point in enumerate(ALL_PICKLEBALL_KEYPOINTS[:15])
            if point.name not in NET_KEYPOINT_NAMES
        }
        with torch.no_grad():
            for floor_index, name in enumerate(STRUCTURED_FLOOR_KEYPOINT_NAMES):
                old_index = canonical_index.get(name)
                if old_index is None or old_index >= v2_weight.shape[0]:
                    continue
                model.keypoint_queries[floor_index].copy_(v2_weight[old_index, :, 0, 0])
                if isinstance(v2_bias, torch.Tensor) and old_index < v2_bias.shape[0]:
                    model.keypoint_bias[floor_index].copy_(v2_bias[old_index])
                if (
                    isinstance(v2_vis_weight, torch.Tensor)
                    and v2_vis_weight.ndim == 2
                    and old_index < v2_vis_weight.shape[0]
                    and v2_vis_weight.shape[1] == model.vis_head.weight.shape[1]
                ):
                    model.vis_head.weight[floor_index].copy_(v2_vis_weight[old_index])
                if isinstance(v2_vis_bias, torch.Tensor) and old_index < v2_vis_bias.shape[0]:
                    model.vis_head.bias[floor_index].copy_(v2_vis_bias[old_index])
                initialized_queries.append(name)

    return {
        "loaded_shared_keys": sorted(compatible),
        "skipped_v2_keys": sorted(skipped),
        "missing_v3_keys": sorted(str(name) for name in missing),
        "unexpected_keys": sorted(str(name) for name in unexpected),
        "initialized_canonical_queries": initialized_queries,
    }


__all__ = [
    "COURT_STRUCTURED_V3_ARCHITECTURE",
    "STRUCTURED_DISTANCE_CLASS_NAMES",
    "STRUCTURED_DISTANCE_SEGMENTS",
    "STRUCTURED_FLOOR_KEYPOINT_COUNT",
    "STRUCTURED_FLOOR_KEYPOINT_NAMES",
    "STRUCTURED_FLOOR_KEYPOINTS",
    "covariance_matrices_from_params",
    "initialize_structured_v3_from_v2",
    "make_court_structured_v3_model",
]
