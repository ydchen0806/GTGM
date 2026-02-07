"""
GTGM+ Downstream Fine-tuning — Medical Image Segmentation

Fine-tuning script for evaluating GTGM+ pretrained encoders on
Medical Segmentation Decathlon (MSD) tasks.

Supports multiple downstream tasks:
    - CT: Liver, Pancreas, Lung, HepaticVessel, Colon, Spleen
    - MRI: BrainTumour, Heart, Hippocampus, Prostate

Models:
    - ResUNet: Standard 3D ResUNet (with pretrained encoder)
    - ResUNetWithAttention: ResUNet + self-attention in skip connections

Usage:
    # Fine-tune on Heart task with GTGM+ pretrained weights
    python finetune/train.py \\
        --task_name Task02_Heart \\
        --model_name ResUNet \\
        --pretrained \\
        --pretrained_path checkpoints/gtgm_plus_encoder.pth \\
        --data_dir /path/to/MSD/

    # Fine-tune without pretraining (supervised baseline)
    python finetune/train.py \\
        --task_name Task03_Liver \\
        --model_name ResUNet \\
        --no-pretrained \\
        --data_dir /path/to/MSD/

Reference:
    "GTGM+: Knowledge-Enhanced 3D Vision-Language Pretraining for
     Multi-Modal Medical Image Analysis" (CVIU 2026)

Author: Yinda Chen (yindachen@mail.ustc.edu.cn)
"""

import os
import sys
import json
import logging
import argparse

import torch
from tqdm import tqdm
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    RandSpatialCropd,
)
from monai.metrics import DiceMetric
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)

from resunet import ResUNet
from attresUnet import ResUNetWithAttention


# ═══════════════════════════════════════════════════════════════════════
# Argument Parser
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    """Parse command-line arguments for downstream fine-tuning."""
    parser = argparse.ArgumentParser(
        description='GTGM+ Downstream Fine-tuning for Medical Image Segmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='/path/to/MSD/',
                        help='Root directory of MSD dataset')
    parser.add_argument('--task_name', type=str, default='Task02_Heart',
                        choices=[
                            'Task01_BrainTumour', 'Task02_Heart',
                            'Task03_Liver', 'Task04_Hippocampus',
                            'Task05_Prostate', 'Task06_Lung',
                            'Task07_Pancreas', 'Task08_HepaticVessel',
                            'Task09_Spleen', 'Task10_Colon',
                        ],
                        help='MSD task name for fine-tuning')
    parser.add_argument('--label_ratio', type=float, default=1.0,
                        choices=[0.01, 0.1, 1.0],
                        help='Fraction of labeled training data (0.01/0.1/1.0)')

    # Model arguments
    parser.add_argument('--model_name', type=str, default='ResUNet',
                        choices=['ResUNet', 'ResUNetWithAttention'],
                        help='Segmentation model architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Load pretrained GTGM+ encoder weights')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false')
    parser.add_argument('--pretrained_path', type=str, default='',
                        help='Path to pretrained encoder weight file')
    parser.add_argument('--input_size', type=int, nargs=3, default=[24, 224, 224],
                        help='Input volume size (D H W)')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--max_iterations', type=int, default=25000,
                        help='Maximum training iterations')
    parser.add_argument('--eval_interval', type=int, default=500,
                        help='Validation interval (iterations)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for AdamW')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Training device')

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# Data Transforms
# ═══════════════════════════════════════════════════════════════════════

def get_train_transforms():
    """Build training data augmentation pipeline using MONAI transforms."""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175, a_max=250,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
        RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
    ])


def get_val_transforms():
    """Build validation data preprocessing pipeline."""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175, a_max=250,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ])


# ═══════════════════════════════════════════════════════════════════════
# Training & Validation Functions
# ═══════════════════════════════════════════════════════════════════════

def validation(model, val_loader, input_size, post_label, post_pred, dice_metric, device, logger):
    """
    Run validation and compute mean Dice score.

    Uses sliding window inference for memory-efficient evaluation
    on full-resolution volumes.

    Args:
        model: Segmentation model in eval mode.
        val_loader: Validation DataLoader.
        input_size: Sliding window size (D, H, W).
        post_label: Post-processing transform for labels (one-hot encoding).
        post_pred: Post-processing transform for predictions (argmax + one-hot).
        dice_metric: MONAI DiceMetric accumulator.
        device: Computation device.
        logger: Logger instance.

    Returns:
        Mean Dice score across all validation samples.
    """
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            val_inputs = batch["image"].to(device)
            val_labels = batch["label"].to(device)

            val_outputs = sliding_window_inference(val_inputs, input_size, 4, model)

            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(t) for t in val_labels_list]

            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(t) for t in val_outputs_list]

            dice_metric(y_pred=val_output_convert, y=val_labels_convert)

        mean_dice = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice


def train_loop(model, train_loader, val_loader, optimizer, loss_function,
               max_iterations, eval_interval, input_size, post_label, post_pred,
               dice_metric, device, save_dir, logger):
    """
    Main fine-tuning training loop.

    Args:
        model: Segmentation model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        loss_function: DiceCE loss function.
        max_iterations: Maximum training iterations.
        eval_interval: Steps between validations.
        input_size: Input volume size for sliding window inference.
        post_label, post_pred: Post-processing transforms.
        dice_metric: Dice metric calculator.
        device: Training device.
        save_dir: Directory for saving best model.
        logger: Logger instance.

    Returns:
        Best Dice score achieved during training.
    """
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []

    while global_step < max_iterations:
        model.train()
        epoch_loss = 0
        step = 0

        epoch_iterator = tqdm(
            train_loader,
            desc=f"Training ({global_step}/{max_iterations})",
            dynamic_ncols=True,
        )

        for batch in epoch_iterator:
            step += 1
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            logit_map = model(x)
            loss = loss_function(logit_map, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            logger.info(f'step={global_step}, loss={loss.item():.6f}')

            epoch_iterator.set_description(
                f"Training ({global_step}/{max_iterations}) loss={loss.item():.5f}"
            )

            # ── Periodic Validation ─────────────────────────────────
            if (global_step % eval_interval == 0 and global_step != 0) or global_step == max_iterations:
                dice_val = validation(
                    model, val_loader, input_size,
                    post_label, post_pred, dice_metric,
                    device, logger,
                )

                avg_epoch_loss = epoch_loss / step
                epoch_loss_values.append(avg_epoch_loss)
                metric_values.append(dice_val)

                if dice_val > dice_val_best:
                    dice_val_best = dice_val
                    global_step_best = global_step
                    torch.save(model.state_dict(), os.path.join(save_dir, "best_metric_model.pth"))
                    logger.info(
                        f'✓ New best model saved! Dice={dice_val:.4f} (step={global_step})'
                    )
                else:
                    logger.info(
                        f'  Dice={dice_val:.4f} (best={dice_val_best:.4f} at step={global_step_best})'
                    )

            global_step += 1

    logger.info(f'Training complete. Best Dice={dice_val_best:.4f} at step={global_step_best}')
    return dice_val_best


# ═══════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Setup data paths ────────────────────────────────────────────
    data_dir = os.path.join(args.data_dir, args.task_name)
    split_json = "dataset.json"
    save_dir = data_dir.replace("DATASET", "MODEL")
    os.makedirs(save_dir, exist_ok=True)

    # ── Load dataset metadata ───────────────────────────────────────
    datasets_path = os.path.join(data_dir, split_json)
    with open(datasets_path) as f:
        task_info = json.load(f)
        labels = task_info["labels"]
        modality = task_info["modality"]

    num_classes = len(labels)
    num_channels = len(modality)
    input_size = tuple(args.input_size)

    print(f'Task: {args.task_name}')
    print(f'  Classes: {num_classes} | Input channels: {num_channels}')
    print(f'  Input size: {input_size}')
    print(f'  Label ratio: {args.label_ratio}')

    # ── Prepare data splits ─────────────────────────────────────────
    datalist = load_decathlon_datalist(datasets_path, True, "training")
    total_len = len(datalist)

    # Apply label ratio for data-efficiency experiments
    train_len = int(total_len * 0.9 * args.label_ratio)
    trainlist = datalist[:train_len]
    val_files = datalist[int(total_len * 0.9):]

    print(f'  Train samples: {len(trainlist)} | Val samples: {len(val_files)}')

    # ── Build data loaders ──────────────────────────────────────────
    train_ds = CacheDataset(
        data=trainlist,
        transform=get_train_transforms(),
        cache_num=min(24, len(trainlist)),
        cache_rate=1.0,
        num_workers=args.num_workers,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True,
    )

    val_ds = CacheDataset(
        data=val_files,
        transform=get_val_transforms(),
        cache_num=min(6, len(val_files)),
        cache_rate=1.0,
        num_workers=4,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # ── Build model ─────────────────────────────────────────────────
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.model_name == 'ResUNet':
        model = ResUNet(args, input_size=input_size, input_channel=num_channels, out_channels=num_classes)
    elif args.model_name == 'ResUNetWithAttention':
        model = ResUNetWithAttention(args, input_size=input_size, input_channel=num_channels, out_channels=num_classes)
    else:
        raise ValueError(f'Unknown model: {args.model_name}')

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f'  Using {torch.cuda.device_count()} GPUs')

    total_params = sum(p.numel() for p in model.parameters())
    print(f'  Model parameters: {total_params:,}')

    # ── Setup training ──────────────────────────────────────────────
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Post-processing
    post_label = AsDiscrete(to_onehot=num_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    # ── Setup logging ───────────────────────────────────────────────
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(f"{args.model_name}_{args.task_name}")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(log_dir, f"{args.model_name}.log"))
    logger.addHandler(fh)

    # ── Train ───────────────────────────────────────────────────────
    best_dice = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_function=loss_function,
        max_iterations=args.max_iterations,
        eval_interval=args.eval_interval,
        input_size=input_size,
        post_label=post_label,
        post_pred=post_pred,
        dice_metric=dice_metric,
        device=device,
        save_dir=save_dir,
        logger=logger,
    )

    print(f'\n{"="*60}')
    print(f'Training complete!')
    print(f'Best Dice score: {best_dice:.4f}')
    print(f'Model saved to: {save_dir}/best_metric_model.pth')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
