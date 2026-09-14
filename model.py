import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden_channels = max(channels // reduction, 8)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = self.excitation(scale)
        return x * scale


class ResidualSEBlock(nn.Module):
    """Two 3x3 convolutions + SE attention + residual shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out = self.dropout(out)

        out = out + identity
        out = self.act(out)
        return out


class MatteaNet(nn.Module):
    """A custom residual CNN with SE attention for CIFAR-10."""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Stem: keep the original 32x32 resolution.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
        )

        # 32x32, 64 channels.
        self.stage1 = self._make_stage(
            in_channels=64,
            out_channels=64,
            blocks=2,
            first_stride=1,
            dropout=0.05,
        )

        # 32x32 -> 16x16, 64 -> 128 channels.
        self.stage2 = self._make_stage(
            in_channels=64,
            out_channels=128,
            blocks=2,
            first_stride=2,
            dropout=0.08,
        )

        # 16x16 -> 8x8, 128 -> 256 channels.
        self.stage3 = self._make_stage(
            in_channels=128,
            out_channels=256,
            blocks=3,
            first_stride=2,
            dropout=0.10,
        )

        # 8x8 -> 4x4, 256 -> 384 channels.
        self.stage4 = self._make_stage(
            in_channels=256,
            out_channels=384,
            blocks=2,
            first_stride=2,
            dropout=0.12,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.35),
            nn.Linear(384, num_classes),
        )

        self._initialize_weights()

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        blocks: int,
        first_stride: int,
        dropout: float,
    ) -> nn.Sequential:
        layers = [
            ResidualSEBlock(
                in_channels,
                out_channels,
                stride=first_stride,
                dropout=dropout,
            )
        ]

        for _ in range(1, blocks):
            layers.append(
                ResidualSEBlock(
                    out_channels,
                    out_channels,
                    stride=1,
                    dropout=dropout,
                )
            )

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = MatteaNet()
    dummy = torch.randn(4, 3, 32, 32)
    output = model(dummy)

    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(model)
    print("Input shape       :", tuple(dummy.shape))
    print("Output shape      :", tuple(output.shape))
    print(f"Parameters        : {params:,}")
    print(f"Trainable params  : {trainable_params:,}")
