"""Compact E2E-Spot-family temporal event head.

Architectural reference: ``third_party/spot@edec4201471beed631bed374bd0b95fcdc8a2f4f``
(per-frame visual features followed by temporal spotting).  Production code
does not import the vendored package because its dependencies are dataset-
specific.  This scaffold is unpromoted and ``VERIFIED=0`` remains binding.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class EventHead(nn.Module):
    """Truncated MobileNetV3-small frame encoder + bidirectional GRU head."""

    def __init__(
        self, *, weights: Literal["none", "imagenet"] = "none", feature_dim: int = 32,
        hidden_dim: int = 32, num_classes: int = 3,
        offset_regression_head: bool = False,
    ) -> None:
        super().__init__()
        if weights not in {"none", "imagenet"}:
            raise ValueError(f"unsupported weights: {weights}")
        selected = MobileNet_V3_Small_Weights.DEFAULT if weights == "imagenet" else None
        try:
            backbone = mobilenet_v3_small(weights=selected)
        except Exception as exc:  # download/cache failures must be explicit
            raise RuntimeError(
                "ImageNet weights were requested but could not be loaded; "
                "pre-stage torchvision weights or use --weights none"
            ) from exc
        # A deliberately small torchvision backbone: first four inverted blocks.
        self.frame_backbone = nn.Sequential(*list(backbone.features.children())[:4])
        self.pool = nn.AdaptiveAvgPool2d(1)
        backbone_channels = 24
        self.frame_projection = nn.Sequential(
            nn.Linear(backbone_channels, feature_dim), nn.ReLU(inplace=True)
        )
        self.temporal = nn.GRU(
            feature_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        # The auxiliary dimensions are ordered HIT, BOUNCE. Keeping the module
        # absent (rather than allocating an unused layer) preserves the exact
        # legacy state-dict surface when the feature is disabled.
        self.offset_regressor = (
            nn.Linear(hidden_dim * 2, 2) if offset_regression_head else None
        )
        self.config = {
            "weights": weights,
            "feature_dim": feature_dim,
            "hidden_dim": hidden_dim,
            "num_classes": num_classes,
            "offset_regression_head": offset_regression_head,
            "backbone": "torchvision_mobilenet_v3_small_truncated_4_blocks",
        }

    def _temporal_features(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5:
            raise ValueError(f"expected [B,T,C,H,W], got {tuple(frames.shape)}")
        batch, time, channels, height, width = frames.shape
        encoded = self.frame_backbone(frames.reshape(batch * time, channels, height, width))
        encoded = self.pool(encoded).flatten(1)
        encoded = self.frame_projection(encoded).reshape(batch, time, -1)
        temporal, _ = self.temporal(encoded)
        return temporal

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Return legacy per-frame classification logits only."""

        return self.classifier(self._temporal_features(frames))

    def forward_with_aux(
        self, frames: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logits and raw sub-frame offsets ordered HIT, BOUNCE."""

        if self.offset_regressor is None:
            raise RuntimeError(
                "offset regression head is disabled; construct EventHead with "
                "offset_regression_head=True"
            )
        temporal = self._temporal_features(frames)
        return self.classifier(temporal), self.offset_regressor(temporal)


def upgrade_event_head_with_offset(model: EventHead) -> EventHead:
    """Clone a legacy head and randomly initialize only its offset regressor.

    This is the model-only initialization seam for a classification checkpoint:
    all legacy parameters and buffers must load, while the two new offset-head
    tensors retain the caller-seeded initialization from the new model.
    """

    if model.offset_regressor is not None:
        raise ValueError("event head already has offset regression enabled")

    config = model.config
    upgraded = EventHead(
        weights="none",
        feature_dim=int(config["feature_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        num_classes=int(config["num_classes"]),
        offset_regression_head=True,
    )
    reference = next(model.parameters())
    upgraded.to(device=reference.device, dtype=reference.dtype)
    incompatible = upgraded.load_state_dict(model.state_dict(), strict=False)
    expected_missing = {"offset_regressor.weight", "offset_regressor.bias"}
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "legacy event-head state did not match the offset-enabled model: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    upgraded.train(model.training)
    return upgraded


def masked_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, validity_mask: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross entropy that removes invalid source/class columns from the loss.

    ``validity_mask`` is [B,C]. Background must remain valid. A BOUNCE-only
    source uses [1,0,1], while a HIT-only source uses [1,1,0].

    ``class_weights`` is optional and ordered background, HIT, BOUNCE. When
    omitted, the calculation remains the original unweighted implementation.
    """

    if logits.ndim != 3 or targets.shape != logits.shape[:2]:
        raise ValueError("logits/targets shape mismatch")
    if validity_mask.shape != (logits.shape[0], logits.shape[2]):
        raise ValueError("validity_mask must be [B,C]")
    if not bool(validity_mask[:, 0].all()):
        raise ValueError("background must be valid for every sample")
    valid_target = validity_mask.gather(1, targets).bool()
    masked_logits = logits.masked_fill(~validity_mask[:, None, :], -1e4)
    if not bool(valid_target.any()):
        raise ValueError("batch contains no loss-valid targets")
    if class_weights is None:
        losses = nn.functional.cross_entropy(
            masked_logits.flatten(0, 1), targets.flatten(), reduction="none"
        ).reshape_as(targets)
        return losses[valid_target].mean()

    weights = torch.as_tensor(class_weights, dtype=logits.dtype, device=logits.device)
    if weights.shape != (logits.shape[2],):
        raise ValueError(f"class_weights must contain {logits.shape[2]} entries")
    if not bool(torch.isfinite(weights).all()) or not bool((weights > 0).all()):
        raise ValueError("class_weights must be finite and strictly positive")
    flat_valid = valid_target.flatten()
    return nn.functional.cross_entropy(
        masked_logits.flatten(0, 1)[flat_valid],
        targets.flatten()[flat_valid],
        weight=weights,
    )


def checkpoint_payload(model: EventHead, **metadata: object) -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_type": "event_head_scaffold",
        "verified": False,
        "model_config": model.config,
        "state_dict": model.state_dict(),
        **metadata,
    }


def load_checkpoint(path: str | bytes | "os.PathLike[str]", *, device: str = "cpu") -> tuple[EventHead, dict[str, object]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("model_type") != "event_head_scaffold":
        raise ValueError(f"not an event-head checkpoint: {path}")
    config = dict(payload["model_config"])
    model = EventHead(
        weights="none", feature_dim=int(config["feature_dim"]),
        hidden_dim=int(config["hidden_dim"]), num_classes=int(config["num_classes"]),
        offset_regression_head=bool(config.get("offset_regression_head", False)),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    return model, payload
