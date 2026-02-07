"""
GTGM+ Pretraining Entry Point

Launch script for distributed vision-language pretraining of the GTGM+ framework.

Usage:
    # Single-node multi-GPU training (e.g., 8 GPUs)
    torchrun --nnodes=1 --nproc_per_node=8 pretrain/main.py \\
        --config config/pretraining_all.yaml

    # Single GPU training
    torchrun --nnodes=1 --nproc_per_node=1 pretrain/main.py \\
        --config config/pretraining_all.yaml

Reference:
    "GTGM+: Knowledge-Enhanced 3D Vision-Language Pretraining for
     Multi-Modal Medical Image Analysis" (CVIU 2026)

Author: Yinda Chen (yindachen@mail.ustc.edu.cn)
"""

import os
import sys
import random
import argparse

import yaml
import torch
import numpy as np
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from attrdict import AttrDict

from dataloader.data_provider_pretraining import Train
from utils.utils_builder import ResNet_CXRBert
from utils.utils_trainer import trainer_wBert


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='GTGM+ Vision-Language Pretraining',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/pretraining_all.yaml',
        help='Path to YAML configuration file (default: config/pretraining_all.yaml)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)',
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default=None,
        help='Directory for saving checkpoints (overrides config)',
    )
    return parser.parse_args()


def init_dist(backend: str = 'nccl', **kwargs):
    """
    Initialize distributed training environment.

    Sets up the process group for multi-GPU training using torchrun.
    Each process is assigned to a specific GPU based on its rank.

    Args:
        backend: Distributed backend ('nccl' for GPU, 'gloo' for CPU).
    """
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ.get("RANK", 0))
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, rank=rank, world_size=num_gpus, **kwargs)
    print(f'[Rank {rank}] Distributed training initialized with {num_gpus} GPUs')


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ddp_main():
    """
    Main entry point for distributed pretraining.

    Workflow:
        1. Parse arguments and load configuration
        2. Initialize distributed training
        3. Build dataset, model, optimizer, and trainer
        4. Launch pretraining loop
    """
    args = parse_args()
    device = torch.device('cuda')

    # ── Initialize distributed training ─────────────────────────────
    init_dist()
    torch.cuda.empty_cache()

    # ── Load configuration ──────────────────────────────────────────
    config_path = args.config
    if not os.path.isabs(config_path):
        # Resolve relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, config_path)

    with open(config_path, "r") as f:
        cfg = AttrDict(yaml.safe_load(f))

    rank = dist.get_rank()
    if rank == 0:
        print(f'Configuration loaded from: {config_path}')
        print(f'Data type: {cfg.DATA.type}')
        print(f'Model type: {cfg.MODEL.model_type}')

    # ── Set random seeds ────────────────────────────────────────────
    set_seed(args.seed)

    # ── Build dataset ───────────────────────────────────────────────
    train_dataset = Train(cfg)
    if rank == 0:
        print(f'Training dataset: {len(train_dataset)} samples')

    # ── Build model ─────────────────────────────────────────────────
    model = ResNet_CXRBert()

    # Optionally freeze early layers of the text encoder
    if cfg.get('TRAIN', {}).get('free_layers') is not None:
        num_frozen = int(cfg['network']['free_layers'])
        for layer_idx in range(num_frozen):
            for param in model.lm_model.encoder.layer[layer_idx].parameters():
                param.requires_grad = False
        if rank == 0:
            print(f'Frozen first {num_frozen} layers of text encoder')

    # Convert BatchNorm to SyncBatchNorm for distributed training
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    model = DDP(model, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Total parameters: {total_params:,}')
        print(f'Trainable parameters: {trainable_params:,}')

    # ── Build optimizer ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        **cfg['optimizer']['params'],
        betas=(0.9, 0.999),
    )

    # ── Build trainer and start training ────────────────────────────
    trainer = trainer_wBert(
        model=model,
        optimizer=optimizer,
        device=device,
        model_name=cfg['wandb_name'],
        **cfg['trainer'],
    )

    trainer.train_w_TextEmb(train_dataset, checkpoint_dir=args.checkpoint_dir)

    # Cleanup
    dist.destroy_process_group()


if __name__ == '__main__':
    ddp_main()
