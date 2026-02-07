"""
3D ResUNet with Self-Attention — Enhanced Segmentation Network

Extends the standard ResUNet with self-attention modules in skip connections
for capturing long-range spatial dependencies in 3D medical volumes.

The self-attention mechanism computes global context at each decoder level,
enabling the model to leverage non-local spatial relationships for improved
segmentation of complex anatomical structures.

Architecture:
    Encoder: ResNet50 (3D) — same as standard ResUNet
    Decoder: Transpose Conv + Self-Attention(skip) + Conv blocks

Reference:
    "GTGM+: Knowledge-Enhanced 3D Vision-Language Pretraining for
     Multi-Modal Medical Image Analysis" (CVIU 2026)

Author: Yinda Chen (yindachen@mail.ustc.edu.cn)
"""

from typing import Tuple

import torch
import torch.nn as nn

from utils_resnet3d import resnet50


class SelfAttention(nn.Module):
    """
    3D Self-Attention Module for volumetric feature maps.

    Computes global attention across all spatial positions using
    query-key-value decomposition. The channel dimension is reduced
    by 8× for queries and keys to keep computation tractable.

    Attention(Q, K, V) = softmax(Q^T · K) · V + X  (residual connection)

    Args:
        in_channels: Number of input/output channels.
    """

    def __init__(self, in_channels: int):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature map, shape (B, C, D, H, W).

        Returns:
            Attention-enhanced feature map, shape (B, C, D, H, W).
        """
        batch_size, _, depth, height, width = x.size()
        spatial_size = depth * height * width

        # Query: (B, C//8, D*H*W) -> (B, D*H*W, C//8)
        query = self.query(x).view(batch_size, -1, spatial_size).permute(0, 2, 1)
        # Key: (B, C//8, D*H*W)
        key = self.key(x).view(batch_size, -1, spatial_size)
        # Attention: (B, D*H*W, D*H*W)
        attention = self.softmax(torch.bmm(query, key))
        # Value: (B, C, D*H*W)
        value = self.value(x).view(batch_size, -1, spatial_size)
        # Output: weighted sum + residual
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, -1, depth, height, width)

        return out + x  # Residual connection


class ResUNetWithAttention(nn.Module):
    """
    3D ResUNet with Self-Attention in Skip Connections.

    Augments the standard ResUNet decoder with self-attention modules
    applied to encoder skip connections before adding to decoder features.
    This enables the model to capture global spatial context at each
    resolution level.

    Architecture:
        Encoder: ResNet50 (3D) — 5 stages [64, 256, 512, 1024, 2048]
        Skip Attention: SelfAttention at stages 1-4
        Decoder: TransposeConv + Attn(skip) + 2×Conv3d + ReLU

    Args:
        cfg: Configuration namespace with pretrained settings.
        input_size: Output spatial dimensions (D, H, W).
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
        super(ResUNetWithAttention, self).__init__()

        self.input_channel = input_channel
        self.cfg = cfg

        # ── Build ResNet50 Encoder ──────────────────────────────────
        resnet = resnet50()
        if input_channel != 1:
            resnet.conv1 = nn.Conv3d(
                input_channel, 64,
                kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3),
                bias=False,
            )
        resnet.fc = nn.Identity()

        # Load pretrained weights
        if cfg.pretrained:
            state_dict = torch.load(cfg.pretrained_path, map_location='cpu')
            loaded = 0
            for name, param in state_dict.items():
                if name in resnet.state_dict() and param.size() == resnet.state_dict()[name].size():
                    resnet.state_dict()[name].copy_(param)
                    loaded += 1
                else:
                    print(f'Skip: {name} (expected {resnet.state_dict().get(name, "N/A")}, '
                          f'got {param.size()})')
            print(f'✓ Loaded {loaded} pretrained parameters from {cfg.pretrained_path}')

        # ── Encoder Stages ──────────────────────────────────────────
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.encoder2 = resnet.layer1  # → 256ch
        self.encoder3 = resnet.layer2  # → 512ch
        self.encoder4 = resnet.layer3  # → 1024ch
        self.encoder5 = resnet.layer4  # → 2048ch

        # ── Self-Attention Modules ──────────────────────────────────
        # Applied to skip connections for global context
        self.attention4 = SelfAttention(1024)
        self.attention3 = SelfAttention(512)
        self.attention2 = SelfAttention(256)
        self.attention1 = SelfAttention(64)

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
        self.output = nn.Conv3d(64, out_channels, kernel_size=1)

    @staticmethod
    def _decoder_block(channels: int) -> nn.Sequential:
        """Create a decoder block with two Conv3d + ReLU layers."""
        return nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with self-attention enhanced skip connections.

        Args:
            x: Input volume, shape (B, C, D, H, W).

        Returns:
            Segmentation logits, shape (B, num_classes, D, H, W).
        """
        # ── Encoder ─────────────────────────────────────────────────
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)

        # ── Decoder with Attention-Enhanced Skip Connections ────────
        x = self.upconv4(x5) + self.attention4(x4)
        x = self.decoder4(x)

        x = self.upconv3(x) + self.attention3(x3)
        x = self.decoder3(x)

        x = self.upconv2(x) + self.attention2(x2)
        x = self.decoder2(x)

        x = self.upconv1(x) + self.attention1(x1)
        x = self.adapool(x)
        x = self.decoder1(x)

        # ── Output ──────────────────────────────────────────────────
        return self.output(x)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test ResUNet with Attention')
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--pretrained_path', type=str, default='')
    cfg = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ResUNetWithAttention(cfg, input_size=(24, 224, 224)).to(device)
    x = torch.randn(1, 1, 24, 224, 224).to(device)
    y = model(x)
    print(f'Input shape:  {x.shape}')
    print(f'Output shape: {y.shape}')

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')
    torch.cuda.empty_cache()
