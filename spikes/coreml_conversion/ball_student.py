"""Small CoreML-deployable ball heatmap student.

The model consumes three RGB frames stacked on the channel axis:
``(batch, 9, 288, 512)`` and emits a stride-4 single-ball heatmap.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class SeparableConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class SeparableResidual(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = SeparableConvBNAct(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class InvertedResidual(nn.Module):
    def __init__(self, channels: int, hidden_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class DecodeBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.refine = nn.Sequential(
            SeparableConvBNAct(in_channels + skip_channels, out_channels),
            SeparableConvBNAct(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(torch.cat([x, skip], dim=1))


class WASBLiteBallStudent(nn.Module):
    """WASB-lite style heatmap student for the live CoreML rung-2 spike."""

    input_shape = (1, 9, 288, 512)
    output_shape = (1, 1, 72, 128)

    def __init__(self):
        super().__init__()
        self.stem = ConvBNAct(9, 24)
        self.enc1 = SeparableConvBNAct(24, 40, stride=2)
        self.enc2 = SeparableConvBNAct(40, 64, stride=2)
        self.enc3 = SeparableConvBNAct(64, 128, stride=2)
        self.enc4 = SeparableConvBNAct(128, 256, stride=2)
        self.enc5 = SeparableConvBNAct(256, 384, stride=2)
        self.bottleneck = nn.Sequential(
            InvertedResidual(384, 576),
            InvertedResidual(384, 576),
        )
        self.dec4 = DecodeBlock(384, 256, 192)
        self.dec3 = DecodeBlock(192, 128, 128)
        self.dec2 = DecodeBlock(128, 64, 80)
        self.head = nn.Sequential(
            SeparableConvBNAct(80, 64),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        x = self.enc5(s4)
        x = self.bottleneck(x)
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
