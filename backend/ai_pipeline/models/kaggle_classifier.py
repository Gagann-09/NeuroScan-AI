from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """
    Basic convolutional feature-extraction block.

    Two convolution layers are used so that the baseline has enough
    capacity to learn spatial MRI features while remaining lightweight
    and easy to audit.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
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


class KaggleBrainTumorClassifier(nn.Module):
    """
    Lightweight 2D CNN baseline for four-class brain MRI classification.

    Input:
        [batch, 1, 224, 224]

    Output:
        [batch, 4]

    Class order:
        0 = glioma
        1 = meningioma
        2 = notumor
        3 = pituitary
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
    ) -> None:
        super().__init__()

        if in_channels != 1:
            raise ValueError(
                "Kaggle classifier expects one grayscale input channel."
            )

        if num_classes != 4:
            raise ValueError(
                "Kaggle classifier baseline expects exactly four classes."
            )

        self.features = nn.Sequential(
            ConvBlock(1, 32),
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

        if x.shape[1] != 1:
            raise ValueError(
                f"Expected one input channel, got {x.shape[1]}."
            )

        features = self.features(x)
        features = torch.flatten(features, start_dim=1)

        return self.classifier(features)


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Return the number of trainable parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )