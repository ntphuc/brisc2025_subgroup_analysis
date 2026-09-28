from __future__ import annotations

import warnings
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    resnet50,
)


class ConvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class EfficientNetB0Encoder(nn.Module):
    """Feature pyramid from torchvision EfficientNet-B0."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        try:
            backbone = efficientnet_b0(weights=weights)
        except Exception as exc:
            warnings.warn(
                f"Could not load pretrained EfficientNet-B0 weights ({exc}); "
                "falling back to random initialization."
            )
            backbone = efficientnet_b0(weights=None)

        f = backbone.features
        self.stage1 = nn.Sequential(f[0], f[1])      # 16  @ 1/2
        self.stage2 = f[2]                           # 24  @ 1/4
        self.stage3 = f[3]                           # 40  @ 1/8
        self.stage4 = nn.Sequential(f[4], f[5])      # 112 @ 1/16
        self.stage5 = nn.Sequential(f[6], f[7])      # 320 @ 1/32

    def forward(self, x: torch.Tensor):
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)
        return s1, s2, s3, s4, s5


class UNetEfficientNetB0(nn.Module):
    """Baseline U-Net decoder with EfficientNet-B0 encoder.

    ``forward_with_features`` is retained for PTIR-v6 checkpoint compatibility.
    """

    def __init__(self, pretrained_encoder: bool = True):
        super().__init__()
        self.encoder = EfficientNetB0Encoder(pretrained=pretrained_encoder)
        self.bottleneck = ConvBlock(320, 256)
        self.dec4 = DecoderBlock(256, 112, 128)
        self.dec3 = DecoderBlock(128, 40, 64)
        self.dec2 = DecoderBlock(64, 24, 40)
        self.dec1 = DecoderBlock(40, 16, 24)
        self.head = nn.Sequential(ConvBlock(24, 24), nn.Conv2d(24, 1, 1))

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        out_size = x.shape[-2:]
        s1, s2, s3, s4, s5 = self.encoder(x)
        bottleneck = self.bottleneck(s5)
        d4 = self.dec4(bottleneck, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        pre_head = F.interpolate(d1, size=out_size, mode="bilinear", align_corners=False)
        logits = self.head(pre_head)
        return logits, {
            "bottleneck": bottleneck,
            "dec4": d4,
            "dec3": d3,
            "dec2": d2,
            "dec1": d1,
            "pre_head": pre_head,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_features(x)[0]


class DenseDecoderNode(nn.Module):
    """Compact U-Net++ dense decoder node."""

    def __init__(self, same_level_channels: Sequence[int], up_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(int(sum(same_level_channels)) + int(up_ch), out_ch)

    def forward(self, same_level: Sequence[torch.Tensor], deeper: torch.Tensor) -> torch.Tensor:
        ref = same_level[0]
        up = F.interpolate(deeper, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([*same_level, up], dim=1))


class UNetPlusPlusEfficientNetB0(nn.Module):
    """U-Net++ style dense decoder on the same EfficientNet-B0 encoder.

    This architecture is intentionally implemented without third-party segmentation
    packages so the paper suite remains reproducible on the existing UIT venv.
    """

    def __init__(self, pretrained_encoder: bool = True):
        super().__init__()
        self.encoder = EfficientNetB0Encoder(pretrained=pretrained_encoder)
        self.bottleneck = ConvBlock(320, 256)

        self.x3_1 = DenseDecoderNode([112], 256, 128)
        self.x2_1 = DenseDecoderNode([40], 112, 64)
        self.x1_1 = DenseDecoderNode([24], 40, 40)
        self.x0_1 = DenseDecoderNode([16], 24, 24)

        self.x2_2 = DenseDecoderNode([40, 64], 128, 64)
        self.x1_2 = DenseDecoderNode([24, 40], 64, 40)
        self.x0_2 = DenseDecoderNode([16, 24], 40, 24)

        self.x1_3 = DenseDecoderNode([24, 40, 40], 64, 40)
        self.x0_3 = DenseDecoderNode([16, 24, 24], 40, 24)
        self.x0_4 = DenseDecoderNode([16, 24, 24, 24], 40, 24)
        self.head = nn.Sequential(ConvBlock(24, 24), nn.Conv2d(24, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_size = x.shape[-2:]
        s1, s2, s3, s4, s5 = self.encoder(x)
        b = self.bottleneck(s5)
        x31 = self.x3_1([s4], b)
        x21 = self.x2_1([s3], s4)
        x11 = self.x1_1([s2], s3)
        x01 = self.x0_1([s1], s2)

        x22 = self.x2_2([s3, x21], x31)
        x12 = self.x1_2([s2, x11], x21)
        x02 = self.x0_2([s1, x01], x11)

        x13 = self.x1_3([s2, x11, x12], x22)
        x03 = self.x0_3([s1, x01, x02], x12)
        x04 = self.x0_4([s1, x01, x02, x03], x13)

        x04 = F.interpolate(x04, size=out_size, mode="bilinear", align_corners=False)
        return self.head(x04)


class ASPPConv(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, dilation: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        y = self.proj(self.pool(x))
        return F.interpolate(y, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 256, rates: Sequence[int] = (6, 12, 18)):
        super().__init__()
        branches: list[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        ]
        branches += [ASPPConv(in_ch, out_ch, int(r)) for r in rates]
        branches += [ASPPPooling(in_ch, out_ch)]
        self.branches = nn.ModuleList(branches)
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * len(branches), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([b(x) for b in self.branches], dim=1))


class DeepLabV3PlusResNet50(nn.Module):
    """DeepLabV3+ style segmentation model with a ResNet-50 encoder."""

    def __init__(self, pretrained_encoder: bool = True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained_encoder else None
        try:
            net = resnet50(weights=weights, replace_stride_with_dilation=[False, True, True])
        except Exception as exc:
            warnings.warn(
                f"Could not load pretrained ResNet-50 weights ({exc}); falling back to random initialization."
            )
            net = resnet50(weights=None, replace_stride_with_dilation=[False, True, True])
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.maxpool = net.maxpool
        self.layer1 = net.layer1  # 256 @ 1/4
        self.layer2 = net.layer2  # 512 @ 1/8
        self.layer3 = net.layer3  # 1024 @ 1/8 dilated
        self.layer4 = net.layer4  # 2048 @ 1/8 dilated
        self.aspp = ASPP(2048, 256, rates=(6, 12, 18))
        self.low_proj = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            ConvBlock(256 + 48, 256),
            nn.Conv2d(256, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_size = x.shape[-2:]
        x = self.stem(x)
        x = self.maxpool(x)
        low = self.layer1(x)
        x = self.layer2(low)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.aspp(x)
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, self.low_proj(low)], dim=1)
        x = self.decoder(x)
        return F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)


ARCHITECTURES = {
    "unet_effb0": UNetEfficientNetB0,
    "unetpp_effb0": UNetPlusPlusEfficientNetB0,
    "deeplabv3p_resnet50": DeepLabV3PlusResNet50,
}


def build_model(architecture: str, pretrained_encoder: bool = True) -> nn.Module:
    key = str(architecture).strip().lower()
    if key not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture={architecture!r}; choose from {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[key](pretrained_encoder=pretrained_encoder)
