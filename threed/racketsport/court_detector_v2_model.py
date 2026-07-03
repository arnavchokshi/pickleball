"""Lightweight multi-task model scaffold for court detector v2."""

from __future__ import annotations

from typing import Any


def make_court_detector_v2_model(*, keypoint_count: int, line_count: int, net_count: int) -> Any:
    if keypoint_count <= 0 or line_count <= 0 or net_count <= 0:
        raise ValueError("head counts must be positive")

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class CourtDetectorV2Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 96, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(96, 128, 3, padding=1),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(128, 96, 3, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(96, 64, 3, padding=1),
                nn.ReLU(),
            )
            self.keypoint_head = nn.Conv2d(64, keypoint_count, 1)
            self.line_head = nn.Conv2d(64, line_count, 1)
            self.net_head = nn.Conv2d(64, net_count, 1)
            self.visibility_head = nn.Linear(128, keypoint_count)

        def forward(self, x: Any) -> dict[str, Any]:
            features = self.encoder(x)
            decoded = self.decoder(features)
            if decoded.shape[-2:] != x.shape[-2:]:
                decoded = F.interpolate(decoded, size=x.shape[-2:], mode="bilinear", align_corners=False)
            pooled = torch.mean(features, dim=(-2, -1))
            return {
                "keypoint_heatmaps": self.keypoint_head(decoded),
                "line_masks": self.line_head(decoded),
                "net_masks": self.net_head(decoded),
                "visibility_logits": self.visibility_head(pooled),
            }

    return CourtDetectorV2Model()


def make_resnet50_court_keypoint_regressor(*, keypoint_count: int = 15, weights: Any = None) -> Any:
    """Build a ResNet50 keypoint-regression baseline.

    Output layout is `[x, y, visibility]` repeated per keypoint. Coordinates are
    intentionally raw regression outputs; training code is responsible for target
    normalization and loss scaling.
    """

    if keypoint_count <= 0:
        raise ValueError("keypoint_count must be positive")

    import torch.nn as nn
    from torchvision.models import resnet50

    model = resnet50(weights=weights)
    in_features = int(model.fc.in_features)
    model.fc = nn.Linear(in_features, keypoint_count * 3)
    model.court_keypoint_count = int(keypoint_count)
    model.court_keypoint_output_layout = "x_y_visibility_per_keypoint"
    return model


def make_mobilenet_v3_court_keypoint_regressor(*, keypoint_count: int = 15, weights: Any = None) -> Any:
    """Build a lightweight MobileNetV3-small keypoint-regression baseline."""

    if keypoint_count <= 0:
        raise ValueError("keypoint_count must be positive")

    import torch.nn as nn
    from torchvision.models import mobilenet_v3_small

    model = mobilenet_v3_small(weights=weights)
    final = model.classifier[-1]
    if not hasattr(final, "in_features"):
        raise ValueError("unexpected MobileNetV3 classifier layout")
    model.classifier[-1] = nn.Linear(int(final.in_features), keypoint_count * 3)
    model.court_keypoint_count = int(keypoint_count)
    model.court_keypoint_output_layout = "x_y_visibility_per_keypoint"
    return model
