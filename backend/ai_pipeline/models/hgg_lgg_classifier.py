"""Separate four-channel BraTS HGG/LGG cohort classifier.

Distinct from ARMT-GAN (segmentation, [B,4,H,W] -> [B,1,H,W] mask) and from
the Kaggle classifier (single-channel JPG, 4-class tumour type).

Input:  [B, 4, H, W]  channel order T1, T1ce, T2, FLAIR
Output: [B, 2] logits  class order 0 = HGG, 1 = LGG
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Basic convolutional feature-extraction block.

    Two convolution layers keep the baseline lightweight and auditable.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HggLggClassifier(nn.Module):
    """Lightweight 2D CNN for BraTS HGG/LGG cohort classification.

    Input:
        [batch, 4, 224, 224]
    Output:
        [batch, 2] logits

    Class order:
        0 = HGG
        1 = LGG
    """

    def __init__(self, in_channels: int = 4, num_classes: int = 2) -> None:
        super().__init__()

        if in_channels != 4:
            raise ValueError(
                "HggLggClassifier expects four input channels "
                "(T1, T1ce, T2, FLAIR)."
            )
        if num_classes != 2:
            raise ValueError(
                "HggLggClassifier baseline expects exactly two classes "
                "(HGG, LGG)."
            )

        self.features = nn.Sequential(
            ConvBlock(4, 32),
            nn.MaxPool2d(kernel_size=2),
            ConvBlock(32, 64),
            nn.MaxPool2d(kernel_size=2),
            ConvBlock(64, 128),
            nn.MaxPool2d(kernel_size=2),
            ConvBlock(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D input [B,C,H,W], got shape {tuple(x.shape)}."
            )
        if x.shape[1] != 4:
            raise ValueError(
                f"Expected four input channels, got {x.shape[1]}."
            )
        features = self.features(x)
        features = torch.flatten(features, start_dim=1)
        return self.classifier(features)


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
