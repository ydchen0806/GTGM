"""
GTGM+ Model Builder — Knowledge-Enhanced 3D Vision-Language Pretraining

This module implements the core pretraining model for the GTGM+ framework
as described in:
    "GTGM+: Knowledge-Enhanced 3D Vision-Language Pretraining for
     Multi-Modal Medical Image Analysis" (CVIU 2026)

Architecture Overview:
    - Visual Encoder: SwinUNETR with hierarchical feature extraction
    - Text Encoder: BioBERT (frozen) for medical text embedding
    - Cross-Modal Projection: MLP heads mapping visual/text features to
      shared 512-d embedding space for contrastive learning

Author: Yinda Chen (yindachen@mail.ustc.edu.cn)
"""

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
from transformers import AutoModel, AutoTokenizer

# Local imports
from utils_resnet3d import resnet50
from swin_unetr import SwinUNETR


class ResNet_CXRBert(nn.Module):
    """
    GTGM+ Pretraining Model: Vision-Language Alignment Network.

    Combines a 3D SwinUNETR visual encoder with a frozen BioBERT text encoder
    for cross-modal contrastive pretraining on 3D medical volumes.

    The model processes two augmented views of the same volume (img1, img2)
    and a text description, projecting all into a shared 512-d space for:
      - Image-Text contrastive loss (CLIP-style)
      - Image-Image covariance loss (BarlowTwins-style)

    Architecture:
        Visual:  SwinUNETR(img) -> 2048-d -> MLP -> 512-d (proj_v)
        Text:    BioBERT(text)  ->  768-d -> MLP -> 512-d (proj_t)

    Args:
        img_size: Input volume spatial size. Default: (32, 160, 160).
        visual_feat_dim: Output dim of visual encoder. Default: 2048.
        text_feat_dim: Output dim of text encoder. Default: 768.
        proj_dim: Shared projection dimension. Default: 512.
        text_encoder_name: HuggingFace model name for text encoder.
            Default: 'dmis-lab/biobert-v1.1'.
        max_text_len: Maximum tokenized text length. Default: 128.
    """

    def __init__(
        self,
        img_size: Tuple[int, int, int] = (32, 160, 160),
        visual_feat_dim: int = 2048,
        text_feat_dim: int = 768,
        proj_dim: int = 512,
        text_encoder_name: str = 'dmis-lab/biobert-v1.1',
        max_text_len: int = 128,
    ):
        super(ResNet_CXRBert, self).__init__()

        self.max_text_len = max_text_len

        # ── Visual Encoder ──────────────────────────────────────────────
        # SwinUNETR: hierarchical 3D vision transformer with shifted windows
        # Produces a global feature vector of dimension `visual_feat_dim`
        self.encoder = SwinUNETR(
            img_size=img_size,
            in_channels=1,
            out_channels=1,
            depths=(2, 4, 2, 2),
            for_pretrain=True,
            feature_size=48,
        )

        # ── Visual Projection Head ─────────────────────────────────────
        # Maps visual features to shared embedding space
        # 2048 -> 2048 -> BN -> ReLU -> 512 -> BN
        self.proj_v = nn.Sequential(
            nn.Linear(visual_feat_dim, visual_feat_dim),
            nn.BatchNorm1d(visual_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(visual_feat_dim, proj_dim),
            nn.BatchNorm1d(proj_dim, affine=False),
        )

        # ── Text Projection Head ───────────────────────────────────────
        # Maps text [CLS] token embedding to shared embedding space
        # 768 -> 2048 -> BN -> ReLU -> 512 -> BN
        self.proj_t = nn.Sequential(
            nn.Linear(text_feat_dim, visual_feat_dim),
            nn.BatchNorm1d(visual_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(visual_feat_dim, proj_dim),
            nn.BatchNorm1d(proj_dim, affine=False),
        )

        # ── Text Encoder (Frozen) ──────────────────────────────────────
        # BioBERT: biomedical domain language model (frozen during pretraining)
        self.lm_model = AutoModel.from_pretrained(
            text_encoder_name, trust_remote_code=True, revision='main'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            text_encoder_name, trust_remote_code=True, revision='main'
        )

    def _tokenize(self, text: list) -> dict:
        """
        Tokenize a batch of text strings using the BioBERT tokenizer.

        Args:
            text: List of text strings to tokenize.

        Returns:
            Dictionary with 'input_ids' and 'attention_mask' tensors.
        """
        tokenizer_output = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_text_len,
            padding='max_length',
            return_tensors='pt',
        )
        return tokenizer_output

    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for vision-language contrastive pretraining.

        Args:
            img1: First augmented view, shape (B, 1, D, H, W).
            img2: Second augmented view, shape (B, 1, D, H, W).
            input_ids: Tokenized text input IDs, shape (B, L).
            attention_mask: Text attention mask, shape (B, L).

        Returns:
            Dictionary containing:
                - img_emb1: Raw visual embedding of view 1, shape (B, 2048)
                - img_emb2: Raw visual embedding of view 2, shape (B, 2048)
                - proj_img_emb1: Projected visual embedding of view 1, shape (B, 512)
                - proj_img_emb2: Projected visual embedding of view 2, shape (B, 512)
                - proj_text_emb: Projected text embedding, shape (B, 512)
        """
        # Encode visual views
        img_emb1 = self.encoder(img1)
        img_emb1 = img_emb1.view(img_emb1.shape[0], -1)  # (B, 2048)

        img_emb2 = self.encoder(img2)
        img_emb2 = img_emb2.view(img_emb2.shape[0], -1)  # (B, 2048)

        # Encode text (frozen BioBERT)
        text_emb = self.lm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state  # (B, L, 768)

        # Project to shared embedding space
        proj_img_emb1 = self.proj_v(img_emb1)         # (B, 512)
        proj_img_emb2 = self.proj_v(img_emb2)         # (B, 512)
        proj_text_emb = self.proj_t(text_emb[:, 0])    # (B, 512) — [CLS] token

        return {
            'img_emb1': img_emb1,
            'img_emb2': img_emb2,
            'proj_img_emb1': proj_img_emb1,
            'proj_img_emb2': proj_img_emb2,
            'proj_text_emb': proj_text_emb,
        }


class MLPHead(nn.Module):
    """
    Simple MLP Projection Head for self-supervised learning.

    Maps features through a two-layer MLP: Linear -> BN -> ReLU -> Linear.

    Args:
        in_channels: Input feature dimension.
        mlp_hidden_size: Hidden layer dimension.
        projection_size: Output projection dimension.
    """

    def __init__(self, in_channels: int, mlp_hidden_size: int, projection_size: int):
        super(MLPHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    model = ResNet_CXRBert()
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    torch.cuda.empty_cache()
