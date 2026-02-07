"""
GTGM+ Pretraining Trainer — Vision-Language Contrastive Learning

This module implements the training loop for GTGM+ pretraining, combining:
  - CLIP-style image-text contrastive loss (bidirectional)
  - BarlowTwins-style covariance loss for visual self-supervision
  - Mixed-precision training with GradScaler

The trainer supports:
  - Distributed training via DDP (DistributedDataParallel)
  - Automatic checkpoint saving and resuming
  - Cosine annealing learning rate schedule with warm restarts

Reference:
    "GTGM+: Knowledge-Enhanced 3D Vision-Language Pretraining for
     Multi-Modal Medical Image Analysis" (CVIU 2026)

Author: Yinda Chen (yindachen@mail.ustc.edu.cn)
"""

import os
import re
import sys
import math
import time
import random
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════

def text_filter(text: str) -> str:
    """
    Remove figure/caption prefixes from generated text.

    Filters patterns like "figure 1 : ...", "fig. 2. ..." etc.
    that may appear in auto-generated captions.

    Args:
        text: Raw generated text string.

    Returns:
        Cleaned text string.
    """
    pattern = re.compile(
        r'^figure\s\d+\s:\s|^fig\.\s\d+\.\s|^figure\s\d+\.\s|^fig\.\s\d+\s|figure\s\d+\s'
    )
    return re.sub(pattern, '', text)


def generate_caption(
    tempimg: torch.Tensor,
    base_name: Union[str, list],
    processor: AutoProcessor,
    text_generator: BlipForConditionalGeneration,
) -> Union[str, List[str]]:
    """
    Generate text captions from 3D medical volume slices using BLIP.

    Randomly selects a 2D slice from the 3D volume and generates
    a caption using the BLIP model. This serves as the baseline
    text generation before knowledge graph refinement.

    Args:
        tempimg: 3D volume tensor, shape (D, H, W) or (B, D, H, W).
        base_name: Volume filename(s) used as caption prefix.
        processor: BLIP image processor.
        text_generator: BLIP caption generation model.

    Returns:
        Generated caption string(s).
    """
    tempimg = tempimg.detach().cpu().numpy()

    if len(tempimg.shape) == 3:
        # Single volume: (D, H, W)
        x, _, _ = tempimg.shape
        z_select = random.randint(0, x - 1)
        imgs1_png = Image.fromarray(tempimg[z_select, :, :] * 255)
        imgs1_inputs = processor(imgs1_png, return_tensors="pt")
        pixel_values = imgs1_inputs['pixel_values'].cuda()
        generated_ids = text_generator.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        generated_caption = text_filter(generated_caption)
        name = base_name[0].split('.')[0] if isinstance(base_name, list) else base_name.split('.')[0]
        return f'{name}, {generated_caption}'

    elif len(tempimg.shape) == 4:
        # Batch of volumes: (B, D, H, W)
        b, x, _, _ = tempimg.shape
        text_list = []
        for i in range(b):
            z_select = random.randint(0, x - 1)
            imgs1_png = Image.fromarray(tempimg[i, z_select, :, :] * 255)
            imgs1_inputs = processor(imgs1_png, return_tensors="pt")
            pixel_values = imgs1_inputs['pixel_values'].cuda()
            generated_ids = text_generator.generate(pixel_values=pixel_values, max_length=50)
            generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            generated_caption = text_filter(generated_caption)
            name = base_name[i].split('.')[0] if isinstance(base_name, list) else base_name.split('.')[0]
            text_list.append(f'{name}, {generated_caption}')
        return text_list
    else:
        raise ValueError(f"Unexpected image shape: {tempimg.shape}")


# ═══════════════════════════════════════════════════════════════════════
# Data Provider (Iterator Wrapper)
# ═══════════════════════════════════════════════════════════════════════

class Provider:
    """
    Infinite data iterator wrapper with distributed sampling support.

    Wraps a PyTorch Dataset into an auto-restarting iterator that
    supports distributed training via DistributedSampler.

    Args:
        dataset: PyTorch Dataset to iterate over.
        batch_size: Number of samples per batch.
        num_workers: Number of data loading workers.
    """

    def __init__(self, dataset, batch_size: int, num_workers: int):
        self.data = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_iter = None
        self.iteration = 0
        self.epoch = 1

    def __len__(self):
        return len(self.data)

    def build(self):
        """Build a new DataLoader with DistributedSampler."""
        self.data_iter = iter(
            DataLoader(
                dataset=self.data,
                batch_size=self.batch_size,
                num_workers=0,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
                sampler=DistributedSampler(self.data),
            )
        )

    def next(self):
        """
        Get the next batch, automatically restarting the iterator
        when the dataset is exhausted.
        """
        if self.data_iter is None:
            self.build()
        try:
            batch = next(self.data_iter)
            self.iteration += 1
            return batch
        except StopIteration:
            self.epoch += 1
            self.build()
            self.iteration += 1
            batch = next(self.data_iter)
            return batch


# ═══════════════════════════════════════════════════════════════════════
# GTGM+ Trainer
# ═══════════════════════════════════════════════════════════════════════

class trainer_wBert:
    """
    GTGM+ Pretraining Trainer with BioBERT Text Encoder.

    Implements the multi-objective training loop combining:
        1. CLIP Loss: Bidirectional image-text contrastive alignment
           L_clip = CE(sim(v,t), labels) + CE(sim(t,v), labels)
        2. Covariance Loss: BarlowTwins-style cross-view regularization
           L_cov = ||C - I||² (on-diagonal + off-diagonal)

    Total Loss: L = L_clip(view1, text) + L_clip(view2, text) + 0.01 * L_cov(view1, view2)

    Training Features:
        - Mixed-precision training (AMP) for memory efficiency
        - Cosine annealing with warm restarts learning rate schedule
        - Periodic checkpoint saving of encoder weights
        - Top-1 and Top-5 retrieval accuracy tracking

    Args:
        model: GTGM+ VLP model (ResNet_CXRBert).
        optimizer: PyTorch optimizer (AdamW recommended).
        device: Training device (cuda).
        model_name: Experiment name for checkpoint saving.
        **args: Training hyperparameters from config.
    """

    def __init__(self, model, optimizer, device, model_name: str, **args):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model_name = model_name

        # Training hyperparameters
        self.loss_type = args['loss']
        self.train_batch_size = args['batch_size']
        self.test_batch_size = args['test_batch_size']
        self.max_epochs = args['max_epochs']
        self.lr_max = args['lr']
        self.max_iterations = args['max_iterations']
        self.num_workers = args['num_workers']
        self.checkpoint_interval = args['checkpoint_interval']
        self.smooth = args['smooth']
        self.prior_ratio = args['ratio']

        # Caption generation models (for online text generation mode)
        # NOTE: Update these paths to your local model locations
        self.processor = AutoProcessor.from_pretrained(
            args.get('processor_path', 'Salesforce/blip-image-captioning-base')
        )
        self.text_generator = BlipForConditionalGeneration.from_pretrained(
            args.get('text_generator_path', 'Salesforce/blip-image-captioning-base')
        ).cuda()

    # ── Loss Functions ──────────────────────────────────────────────────

    def covar_loss(self, img_embed: torch.Tensor, text_embed: torch.Tensor) -> torch.Tensor:
        """
        BarlowTwins-style covariance loss for cross-view regularization.

        Encourages the cross-correlation matrix between two views
        to be close to the identity matrix:
            L = sum((diag(C) - 1)²) + lambda * sum(off_diag(C)²)

        This reduces redundancy in learned representations while
        maintaining informative features.

        Args:
            img_embed: First view embedding, shape (B, D).
            text_embed: Second view embedding, shape (B, D).

        Returns:
            Scalar covariance loss.
        """
        def off_diagonal(x):
            """Extract off-diagonal elements of a square matrix."""
            n, m = x.shape
            assert n == m
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

        # Cross-correlation matrix C = (1/B) * Z_a^T * Z_b
        logits = torch.mm(img_embed.T, text_embed).to(self.device)
        logits.div_(self.train_batch_size)

        # On-diagonal: encourage C_ii = 1
        on_diag = torch.diagonal(logits).add_(-1).pow_(2).sum()
        # Off-diagonal: encourage C_ij = 0 (redundancy reduction)
        off_diag = off_diagonal(logits).pow_(2).sum()

        loss = on_diag + 0.0051 * off_diag
        return loss / 2

    def reg_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Regression loss based on cosine similarity.

        L = 2 - 2 * cos_sim(x, y) (bidirectional)

        Args:
            x, y: Feature tensors to align, shape (B, D).

        Returns:
            Scalar regression loss.
        """
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        loss = 2 - 2 * (x * y).sum(dim=-1)
        loss += 2 - 2 * (y * x).sum(dim=-1)
        return loss.mean()

    def entropy_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        KL divergence loss between prediction and target distributions.

        Args:
            x: Log-softmax predictions, shape (B, D).
            y: Softmax targets, shape (B, D).

        Returns:
            Scalar KL divergence loss.
        """
        x = F.log_softmax(x, dim=-1)
        y = F.softmax(y, dim=-1)
        metric = nn.KLDivLoss(reduction="batchmean")
        return metric(x, y).mean()

    def clip_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        temperature: float = 0.07,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        CLIP-style bidirectional contrastive loss.

        Computes symmetric cross-entropy between image and text similarities:
            L = CE(sim(I, T) / τ, labels) + CE(sim(T, I) / τ, labels)

        where sim(I, T) = normalized_dot_product(I, T) and labels
        are the identity (each image matches its own text).

        Also computes Top-1 and Top-5 retrieval accuracy for monitoring.

        Args:
            x: Image embeddings, shape (B, D).
            y: Text embeddings, shape (B, D).
            temperature: Softmax temperature τ. Default: 0.07.

        Returns:
            Tuple of (loss, top1_accuracy, top5_accuracy).
        """
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)

        # Compute similarity matrix: (B, B)
        sim = torch.einsum('i d, j d -> i j', x, y) / temperature
        labels = torch.arange(x.shape[0]).long().to(self.device)

        # Bidirectional cross-entropy loss
        loss_t = F.cross_entropy(sim, labels)      # image -> text
        loss_i = F.cross_entropy(sim.T, labels)     # text -> image

        # Retrieval accuracy
        i2t_acc1, i2t_acc5 = self.precision_at_k(sim, labels, top_k=(1,))
        t2i_acc1, t2i_acc5 = self.precision_at_k(sim.T, labels, top_k=(1,))
        acc1 = (i2t_acc1 + t2i_acc1) / 2.0
        acc5 = (i2t_acc5 + t2i_acc5) / 2.0

        return (loss_t + loss_i), acc1, acc5

    @staticmethod
    def precision_at_k(
        output: torch.Tensor,
        target: torch.Tensor,
        top_k: Tuple[int, ...] = (1,),
    ) -> list:
        """
        Compute precision at k for retrieval evaluation.

        Args:
            output: Similarity scores, shape (B, B).
            target: Ground-truth labels, shape (B,).
            top_k: Tuple of k values to compute precision for.

        Returns:
            List of precision values for each k.
        """
        with torch.no_grad():
            maxk = max(top_k)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in top_k:
                correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    # ── Training Loop ───────────────────────────────────────────────────

    def train_w_TextEmb(self, train_dataset, checkpoint_dir: Optional[str] = None):
        """
        Main pretraining loop with text embeddings.

        Training Procedure:
            1. Build distributed data provider
            2. Load checkpoint if available
            3. For each iteration:
               a. Sample batch (img1, img2, caption)
               b. Tokenize caption with BioBERT
               c. Forward pass through model
               d. Compute CLIP loss + covariance loss
               e. Backward pass with mixed precision
               f. Update learning rate (cosine annealing)
            4. Save encoder checkpoints periodically

        Args:
            train_dataset: Pretraining dataset instance.
            checkpoint_dir: Directory for saving checkpoints.
                Defaults to a standard path if not specified.
        """
        mp.set_start_method('spawn', force=True)
        print('Start training...')

        train_provider = Provider(train_dataset, self.train_batch_size, self.num_workers)

        # Setup checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '..', '..', 'checkpoints', self.model_name
            )
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f'Checkpoint directory: {checkpoint_dir}')

        # ── Resume from checkpoint ──────────────────────────────────
        print('Checking for existing checkpoint...')
        ckpt_path = os.path.join(checkpoint_dir, f'{self.model_name}_checkpoint.pth')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            start_iter = ckpt.get('iteration', 0)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print(f'Resumed from iteration {start_iter}')
        else:
            start_iter = 0
            print('Starting training from scratch')

        # ── Learning rate scheduler ─────────────────────────────────
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=int(self.max_iterations // 4 // self.train_batch_size * 0.4),
            T_mult=1,
            eta_min=1e-8,
        )

        scaler = GradScaler()
        niter = start_iter

        # ── Training iterations ─────────────────────────────────────
        while niter < self.max_iterations:
            epoch_loss = 0
            epoch_loss_clip = 0
            epoch_loss_cov = 0

            img1, img2, caption = train_provider.next()

            # Move images to device
            img1 = img1.to(torch.float32).to(self.device).contiguous()
            img2 = img2.to(torch.float32).to(self.device).contiguous()

            self.optimizer.zero_grad()

            # Mixed-precision forward pass
            with autocast():
                # Tokenize text
                imp_tokenize_output = self.model.module._tokenize(caption)
                input_ids = imp_tokenize_output.input_ids.to(self.device).contiguous()
                attention_mask = imp_tokenize_output.attention_mask.to(self.device).contiguous()

                # Forward pass
                output_dict = self.model(img1, img2, input_ids, attention_mask)
                img_emb1 = output_dict['img_emb1']
                img_emb2 = output_dict['img_emb2']
                proj_img_emb1 = output_dict['proj_img_emb1']
                proj_img_emb2 = output_dict['proj_img_emb2']
                proj_text_emb = output_dict['proj_text_emb']

                if self.loss_type == 'only_clip':
                    # CLIP contrastive loss (both views against text)
                    loss_clip1, acc1_1, _ = self.clip_loss(x=proj_img_emb1, y=proj_text_emb)
                    loss_clip2, acc1_2, _ = self.clip_loss(x=proj_img_emb2, y=proj_text_emb)
                    acc1 = (acc1_1 + acc1_2) / 2

                    # BarlowTwins covariance loss (between two views)
                    cov_loss = self.covar_loss(img_emb1, img_emb2) * 0.01

                    loss = loss_clip1 + loss_clip2 + cov_loss

                    # Logging
                    epoch_loss += loss.item()
                    epoch_loss_clip += (loss_clip1.item() + loss_clip2.item())
                    epoch_loss_cov += cov_loss.item()

                    print(
                        f'[Iter {niter}] loss={loss.item():.4f} | '
                        f'clip={loss_clip1.item() + loss_clip2.item():.4f} | '
                        f'cov={cov_loss.item():.4f} | '
                        f'acc@1={acc1.item():.2f}%'
                    )

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                scheduler.step()

            niter += 1

            # ── Periodic checkpoint saving ──────────────────────────
            if niter % 500 == 0:
                torch.save(
                    self.model.module.encoder.state_dict(),
                    os.path.join(checkpoint_dir, f'{self.model_name}_{niter}_iter_encoder.pth'),
                )
                print(f'  ✓ Saved encoder checkpoint at iteration {niter}')

        # ── Final model saving ──────────────────────────────────────
        torch.save(
            self.model.module.encoder.state_dict(),
            os.path.join(checkpoint_dir, f'{self.model_name}_final_encoder.pth'),
        )
        torch.save(
            self.model.module.state_dict(),
            os.path.join(checkpoint_dir, f'{self.model_name}_final_full.pth'),
        )
        print(f'Training complete! Final model saved to {checkpoint_dir}')

    def save_checkpoints(self, epoch: int, path: str):
        """
        Save training checkpoint for resuming.

        Args:
            epoch: Current epoch/iteration number.
            path: File path to save checkpoint.
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
