"""
3D ResUNet — Downstream Segmentation Network for GTGM+

A 3D U-Net with ResNet50 encoder for medical image segmentation.
Supports loading pretrained weights from GTGM+ pretraining for
transfer learning.

Architecture:
    Encoder: ResNet50 (3D) — 5 stages producing features at
             [64, 256, 512, 1024, 2048] channels
    Decoder: Transpose convolution upsampling with skip connections
             and center-crop-padding for spatial alignment

Usage:
    # With pretrained GTGM+ encoder
    model = ResUNet(args, input_size=(24, 224, 224), out_channels=14)

    # Without pretraining
    args.pretrained = False
    model = ResUNet(args, input_size=(24, 224, 224), out_channels=14)

Reference:
    "GTGM+: Knowledge-Enhanced 3D Vision-Language Pretraining for
     Multi-Modal Medical Image Analysis" (CVIU 2026)

Author: Yinda Chen (yindachen@mail.ustc.edu.cn)
"""

import os
import sys
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_resnet3d import resnet50


class ResUNet(nn.Module):
    """
    3D ResUNet for Medical Image Segmentation.

    Combines a ResNet50 encoder with a U-Net style decoder for
    3D volumetric segmentation. Supports loading GTGM+ pretrained
    weights for transfer learning.

    Encoder stages:
        Stage 1: Conv(1→64) + BN + ReLU + MaxPool → 64ch
        Stage 2: ResNet Layer1 → 256ch
        Stage 3: ResNet Layer2 → 512ch
        Stage 4: ResNet Layer3 → 1024ch
        Stage 5: ResNet Layer4 → 2048ch

    Decoder stages:
        Each stage: TransposeConv(up) + Skip Connection + 2×Conv3d + ReLU

    Args:
        cfg: Configuration namespace with:
            - pretrained (bool): Whether to load pretrained weights.
            - pretrained_path (str): Path to pretrained weight file.
        input_size: Spatial dimensions of input volume (D, H, W).
        input_channel: Number of input channels (modalities).
        out_channels: Number of output segmentation classes.
    """

    def __init__(
        self,
        cfg,
        input_size: Tuple[int, int, int] = (64, 64, 64),
        input_channel: int = 1,
        out_channels: int = 14,
    ):
        super(ResUNet, self).__init__()
        self.input_channel = input_channel
        self.output_channel = out_channels
        self.cfg = cfg

        # ── Build ResNet50 Encoder ──────────────────────────────────
        resnet = resnet50()

        # Adjust first conv layer for multi-channel input
        if self.input_channel != 1:
            resnet.conv1 = nn.Conv3d(
                self.input_channel, 64,
                kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3),
                bias=False,
            )
        resnet.fc = nn.Identity()  # Remove classification head

        # ── Load Pretrained Weights ─────────────────────────────────
        if cfg.pretrained:
            self._load_pretrained_weights(resnet, cfg.pretrained_path)

        # ── Encoder Stages ──────────────────────────────────────────
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.encoder2 = resnet.layer1  # 64  → 256 channels
        self.encoder3 = resnet.layer2  # 256 → 512 channels
        self.encoder4 = resnet.layer3  # 512 → 1024 channels
        self.encoder5 = resnet.layer4  # 1024 → 2048 channels

        # ── Decoder Stages ──────────────────────────────────────────
        self.upconv4 = nn.ConvTranspose3d(2048, 1024, kernel_size=2, stride=2)
        self.decoder4 = self._decoder_block(1024)

        self.upconv3 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.decoder3 = self._decoder_block(512)

        self.upconv2 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.decoder2 = self._decoder_block(256)

        self.upconv1 = nn.ConvTranspose3d(
            256, 64, kernel_size=(3, 7, 7), stride=(1, 1, 1), padding=(1, 3, 3)
        )
        self.adapool = nn.AdaptiveAvgPool3d(input_size)
        self.decoder1 = self._decoder_block(64)

        # ── Output Layer ────────────────────────────────────────────
        self.output = nn.Conv3d(64, self.output_channel, kernel_size=1)

    @staticmethod
    def _decoder_block(channels: int) -> nn.Sequential:
        """Create a decoder block with two Conv3d + ReLU layers."""
        return nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def _load_pretrained_weights(self, resnet: nn.Module, pretrained_path: str):
        """
        Load pretrained weights from GTGM+ pretraining or other SSL methods.

        Handles different checkpoint formats:
            - Standard state_dict: Direct parameter mapping
            - BarlowTwins format: Extracts 'module.backbone.*' keys

        Args:
            resnet: ResNet50 module to load weights into.
            pretrained_path: Path to the pretrained weight file.
        """
        state_dict = torch.load(pretrained_path, map_location='cpu')

        if 'barlow' not in pretrained_path:
            # Standard format: direct key matching
            loaded_count = 0
            for name, param in state_dict.items():
                if name in resnet.state_dict() and param.size() == resnet.state_dict()[name].size():
                    resnet.state_dict()[name].copy_(param)
                    loaded_count += 1
                else:
                    print(f'  Skip: {name} (expected {resnet.state_dict().get(name, "N/A")}, '
                          f'got {param.size()})')
        else:
            # BarlowTwins format: extract backbone weights
            state_dict = state_dict['model']
            new_state_dict = {
                name.replace('module.backbone.', ''): param
                for name, param in state_dict.items()
                if 'module.backbone.' in name
            }
            loaded_count = 0
            for name, param in new_state_dict.items():
                if name in resnet.state_dict() and param.size() == resnet.state_dict()[name].size():
                    resnet.state_dict()[name].copy_(param)
                    loaded_count += 1
                else:
                    print(f'  Skip: {name} (shape mismatch: {param.size()})')

        print(f'✓ Loaded pretrained weights from {pretrained_path}')

    def center_crop_padding(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Spatially align two tensors via center cropping or padding.

        When the encoder and decoder produce tensors of different spatial
        dimensions (due to strided operations), this function aligns them
        by either cropping the larger tensor or padding the smaller one.

        Args:
            x: Decoder feature tensor, shape (B, C, D, H, W).
            y: Encoder skip connection tensor, shape (B, C, D', H', W').

        Returns:
            Sum of aligned tensors, shape (B, C, D', H', W').
        """
        _, _, d, h, w = x.shape
        _, _, td, th, tw = y.shape

        if d == td and h == th and w == tw:
            return x + y
        elif d < td or h < th or w < tw:
            # Pad x to match y's spatial dimensions
            d1 = (td - d) // 2
            d2 = td - d - d1
            h1 = (th - h) // 2
            h2 = th - h - h1
            w1 = (tw - w) // 2
            w2 = tw - w - w1
            x = F.pad(x, [w1, w2, h1, h2, d1, d2])
            return x + y
        else:
            # Crop x to match y's spatial dimensions
            d1 = (d - td) // 2
            d2 = d - td - d1
            h1 = (h - th) // 2
            h2 = h - th - h1
            w1 = (w - tw) // 2
            w2 = w - tw - w1
            x = x[:, :, d1:d - d2, h1:h - h2, w1:w - w2]
            return x + y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for 3D segmentation.

        Args:
            x: Input volume, shape (B, C, D, H, W).

        Returns:
            Segmentation probabilities, shape (B, num_classes, D, H, W).
        """
        # ── Encoder ─────────────────────────────────────────────────
        x1 = self.encoder1(x)    # (B, 64, D/2, H/4, W/4)
        x2 = self.encoder2(x1)   # (B, 256, D/2, H/4, W/4)
        x3 = self.encoder3(x2)   # (B, 512, D/4, H/8, W/8)
        x4 = self.encoder4(x3)   # (B, 1024, D/8, H/16, W/16)
        x5 = self.encoder5(x4)   # (B, 2048, D/16, H/32, W/32)

        # ── Decoder with Skip Connections ───────────────────────────
        x = self.center_crop_padding(self.upconv4(x5), x4)
        x = self.decoder4(x)

        x = self.center_crop_padding(self.upconv3(x), x3)
        x = self.decoder3(x)

        x = self.center_crop_padding(self.upconv2(x), x2)
        x = self.decoder2(x)

        x = self.center_crop_padding(self.upconv1(x), x1)
        x = self.adapool(x)
        x = self.decoder1(x)

        # ── Output ──────────────────────────────────────────────────
        x = self.output(x)
        return torch.sigmoid(x)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test ResUNet model')
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--pretrained_path', type=str, default='')
    cfg = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ResUNet(cfg, input_size=(192, 160, 80), input_channel=1, out_channels=1).to(device)
    x = torch.randn(1, 1, 192, 160, 80).to(device)
    y = model(x)
    print(f'Input shape:  {x.shape}')
    print(f'Output shape: {y.shape}')

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')
    torch.cuda.empty_cache()
