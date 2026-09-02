import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Standard convolutional block used by the prototype ARMT-GAN generator.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:

        super().__init__()

        self.conv = nn.Sequential(
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

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.conv(x)


class ARMTGenerator2D(nn.Module):
    """
    Lightweight ARMT-GAN 2D segmentation generator.

    Prototype input:
        [B, 4, H, W]

    Channels:
        0 -> T1
        1 -> T1-contrast / T1ce
        2 -> T2
        3 -> FLAIR

    Output:
        [B, 1, H, W]

    The architecture remains intentionally lightweight during the
    prototype-validation phase.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        features: list[int] | None = None,
    ) -> None:

        super().__init__()

        if features is None:
            features = [64, 128, 256]

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        # ------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------

        current_channels = in_channels

        for feature in features:

            self.downs.append(
                DoubleConv(
                    current_channels,
                    feature,
                )
            )

            current_channels = feature

        # ------------------------------------------------------------------
        # Bottleneck
        # ------------------------------------------------------------------

        self.bottleneck = DoubleConv(
            features[-1],
            features[-1] * 2,
        )

        # ------------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------------

        current_channels = features[-1] * 2

        for feature in reversed(features):

            self.ups.append(
                nn.ConvTranspose2d(
                    current_channels,
                    feature,
                    kernel_size=2,
                    stride=2,
                )
            )

            self.ups.append(
                DoubleConv(
                    feature * 2,
                    feature,
                )
            )

            current_channels = feature

        # ------------------------------------------------------------------
        # Segmentation head
        # ------------------------------------------------------------------

        self.final_conv = nn.Conv2d(
            features[0],
            out_channels,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        skip_connections: list[torch.Tensor] = []

        # Encoder
        for down in self.downs:

            x = down(x)

            skip_connections.append(x)

            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Reverse skips for decoder
        skip_connections = skip_connections[::-1]

        # Decoder
        for index in range(
            0,
            len(self.ups),
            2,
        ):

            x = self.ups[index](x)

            skip_connection = skip_connections[index // 2]

            if x.shape[2:] != skip_connection.shape[2:]:

                x = F.interpolate(
                    x,
                    size=skip_connection.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )

            x = torch.cat(
                (
                    skip_connection,
                    x,
                ),
                dim=1,
            )

            x = self.ups[index + 1](x)

        # Probability mask.
        return torch.sigmoid(
            self.final_conv(x)
        )


class ARMTDiscriminator2D(nn.Module):
    """
    Lightweight conditional PatchGAN discriminator.

    Prototype input:
        MRI = 4 channels
        mask = 1 channel

    Total discriminator channels:
        5
    """

    def __init__(
        self,
        in_channels: int = 5,
    ) -> None:

        super().__init__()

        self.model = nn.Sequential(

            nn.Conv2d(
                in_channels,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True,
            ),

            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(128),

            nn.LeakyReLU(
                0.2,
                inplace=True,
            ),

            nn.Conv2d(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(256),

            nn.LeakyReLU(
                0.2,
                inplace=True,
            ),

            nn.Conv2d(
                256,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
            ),

            nn.Sigmoid(),
        )

    def forward(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:

        x = torch.cat(
            [
                image,
                mask,
            ],
            dim=1,
        )

        return self.model(x)