#!/usr/bin/env python3
"""
SAM2-MoE Training Pipeline

Complete training pipeline for SAM2 with Mixture of Prompt Experts (MoPE).
MoE-LoRA adapters are applied to Memory Attention's projection layers.

Architecture:
    - Base: SAM2.1 Hiera Base+
    - MoE-LoRA on Memory Attention (q_proj, k_proj, v_proj, out_proj)
    - 10 experts, top-k=2 selection, LoRA rank=4

Usage:
    python train_sam2_moe.py --data_dir data/OCID-dataset/ARID20 --epochs 50
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import math
from tqdm import tqdm

from ocid_dataset import get_dataloaders
from sam2.build_sam import build_sam2_video_predictor_moe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class DiceLoss(nn.Module):
    """Dice Loss for segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1.0 - dice


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_sigmoid = torch.sigmoid(pred)
        pred_sigmoid = torch.clamp(pred_sigmoid, 1e-6, 1 - 1e-6)

        bce = F.binary_cross_entropy(pred_sigmoid, target, reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.01,
):
    """
    Create a schedule with linear warmup and cosine decay.

    Args:
        optimizer: The optimizer.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.
        num_cycles: Number of cosine cycles (0.5 = half cycle, decays to min).
        min_lr_ratio: Minimum learning rate as ratio of initial lr.
    """
    def lr_lambda(current_step: int) -> float:
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        # Scale between min_lr_ratio and 1.0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


class SAM2MoETrainer:
    """
    Complete training pipeline for SAM2-MoE.

    This trainer properly integrates with SAM2's forward pass:
    1. Image Encoder: Extracts visual features
    2. Memory Attention (with MoE-LoRA): Conditions features on memory
    3. Mask Decoder: Predicts segmentation masks

    Only MoE-LoRA parameters are trained; base model is frozen.
    """

    def __init__(self, config: argparse.Namespace):
        self.config = config
        self.device = torch.device(config.device)

        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(config.output_dir) / f"run_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(vars(config), f, indent=2)

        # Setup TensorBoard
        self.writer = SummaryWriter(self.output_dir / "tensorboard")

        # Build model
        logger.info("Building SAM2-MoE model...")
        self.model = build_sam2_video_predictor_moe(
            config_file=config.model_config,
            ckpt_path=config.checkpoint,
            device=self.device,
            mode="train",
        )

        # Print parameter statistics
        self._log_parameters()

        # Setup optimizer (only MoE parameters)
        moe_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            moe_params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
        )

        # Setup loss functions
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()

        # Setup data loaders
        logger.info("Loading dataset...")
        self.train_loader, self.val_loader = get_dataloaders(
            root_dir=config.data_dir,
            batch_size=config.batch_size,
            num_frames=config.num_frames,
            image_size=config.image_size,
            num_workers=config.num_workers,
            train_ratio=config.train_ratio,
        )
        logger.info(f"Train: {len(self.train_loader)} batches, Val: {len(self.val_loader)} batches")

        # Setup scheduler with warmup (needs train_loader for step count)
        num_training_steps = config.epochs * len(self.train_loader)
        num_warmup_steps = int(config.warmup_ratio * num_training_steps)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_ratio=config.min_lr_ratio,
        )
        logger.info(f"Scheduler: Cosine with warmup ({num_warmup_steps} warmup steps / {num_training_steps} total steps)")

        # Training state
        self.global_step = 0
        self.best_val_iou = 0.0

        # Early stopping state
        self.early_stopping_patience = config.early_stopping_patience
        self.early_stopping_min_delta = config.early_stopping_min_delta
        self.early_stopping_counter = 0
        self.early_stopping_best_iou = 0.0

    def _log_parameters(self):
        """Log model parameter statistics."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen = total - trainable

        logger.info("=" * 60)
        logger.info("Model Parameters:")
        logger.info(f"  Total:     {total:>12,}")
        logger.info(f"  Trainable: {trainable:>12,} ({100*trainable/total:.2f}%)")
        logger.info(f"  Frozen:    {frozen:>12,} ({100*frozen/total:.2f}%)")
        logger.info("=" * 60)

        # Log MoE-specific parameters
        moe_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                moe_params.append((name, param.numel()))

        logger.info("Trainable MoE Parameters (first 10):")
        for name, count in moe_params[:10]:
            logger.info(f"  {name}: {count:,}")
        if len(moe_params) > 10:
            logger.info(f"  ... and {len(moe_params) - 10} more")

        self.writer.add_scalar("Model/total_params", total, 0)
        self.writer.add_scalar("Model/trainable_params", trainable, 0)

    def _prepare_prompt(self, points: torch.Tensor, device: torch.device) -> Dict:
        """Prepare point prompts for SAM2."""
        # points: (N, 2) in (x, y) format
        point_coords = points.unsqueeze(0).to(device)  # (1, N, 2)
        point_labels = torch.ones(1, points.shape[0], dtype=torch.int32, device=device)
        return {"point_coords": point_coords, "point_labels": point_labels}

    def forward_frame(
        self,
        frame: torch.Tensor,
        points: torch.Tensor,
        output_dict: Dict,
        frame_idx: int,
        is_cond_frame: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass for a single frame through SAM2.

        This is the core training loop that ensures MoE-LoRA in Memory Attention
        is properly invoked and gradients flow through.

        Args:
            frame: Input frame (C, H, W)
            points: Prompt points (N, 2)
            output_dict: Memory dictionary for tracking
            frame_idx: Current frame index
            is_cond_frame: Whether this is the conditioning (first) frame

        Returns:
            pred_mask: Predicted high-res mask logits (H, W)
            obj_ptr: Object pointer for memory
            output_dict: Updated memory dictionary
        """
        device = frame.device

        # Step 1: Image encoding
        img_batch = frame.unsqueeze(0)  # (1, C, H, W)
        backbone_out = self.model.forward_image(img_batch)

        # Step 2: Prepare backbone features
        _, vision_feats, vision_pos_embeds, feat_sizes = \
            self.model._prepare_backbone_features(backbone_out)

        # Step 3: Prepare point prompts
        point_inputs = self._prepare_prompt(points, device)

        # Step 4: Prepare high-resolution features for SAM head
        if len(vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None

        # Step 5: Memory-conditioned features (THIS CALLS MEMORY ATTENTION WITH MOE!)
        pix_feat = self.model._prepare_memory_conditioned_features(
            frame_idx=frame_idx,
            is_init_cond_frame=is_cond_frame,
            current_vision_feats=vision_feats[-1:],
            current_vision_pos_embeds=vision_pos_embeds[-1:],
            feat_sizes=feat_sizes[-1:],
            output_dict=output_dict,
            num_frames=frame_idx + 1,
            track_in_reverse=False,
        )

        # Step 6: SAM mask decoder
        multimask = self.model._use_multimask(is_cond_frame, point_inputs)
        sam_outputs = self.model._forward_sam_heads(
            backbone_features=pix_feat,
            point_inputs=point_inputs,
            mask_inputs=None,
            high_res_features=high_res_features,
            multimask_output=multimask,
        )

        # Unpack outputs
        _, _, _, low_res_masks, high_res_masks, obj_ptr, obj_scores = sam_outputs

        # Step 7: Encode memory for next frames
        current_out = {
            "point_inputs": point_inputs,
            "mask_inputs": None,
            "pred_masks": low_res_masks,
            "obj_ptr": obj_ptr,
        }

        if self.model.num_maskmem > 0:
            maskmem_features, maskmem_pos_enc = self.model._encode_new_memory(
                current_vision_feats=vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks,
                object_score_logits=obj_scores,
                is_mask_from_pts=True,
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

        # Store in output_dict
        if is_cond_frame:
            output_dict["cond_frame_outputs"][frame_idx] = current_out
        else:
            output_dict["non_cond_frame_outputs"][frame_idx] = current_out

        # Return high-res mask (squeeze batch and object dims)
        pred_mask = high_res_masks.squeeze(0).squeeze(0)  # (H, W)

        return pred_mask, obj_ptr, output_dict

    def compute_loss(
        self,
        pred_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss."""
        # Resize target to match prediction if needed
        if pred_mask.shape != target_mask.shape:
            target_mask = F.interpolate(
                target_mask.unsqueeze(0).unsqueeze(0).float(),
                size=pred_mask.shape,
                mode="nearest",
            ).squeeze(0).squeeze(0)

        dice = self.dice_loss(pred_mask, target_mask)
        focal = self.focal_loss(pred_mask, target_mask)
        total = dice + focal

        return total, {"dice": dice.item(), "focal": focal.item(), "total": total.item()}

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        epoch_losses = {"dice": 0.0, "focal": 0.0, "total": 0.0}
        num_samples = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")

        for batch in pbar:
            frames = batch["frames"].to(self.device)  # (B, T, C, H, W)
            masks = batch["masks"].to(self.device)    # (B, T, H, W)
            points_list = batch["points"]             # List of (N, 2)

            B, T, C, H, W = frames.shape
            losses = []
            batch_metrics = {"dice": 0.0, "focal": 0.0, "total": 0.0}
            valid_frames = 0

            for b in range(B):
                # Initialize memory for this sequence
                output_dict = {
                    "cond_frame_outputs": {},
                    "non_cond_frame_outputs": {},
                }

                for t in range(T):
                    frame = frames[b, t]
                    target = masks[b, t]
                    points = points_list[b].to(self.device)

                    is_cond = (t == 0)

                    try:
                        # Forward pass
                        pred_mask, _, output_dict = self.forward_frame(
                            frame=frame,
                            points=points,
                            output_dict=output_dict,
                            frame_idx=t,
                            is_cond_frame=is_cond,
                        )

                        # Compute loss
                        loss, metrics = self.compute_loss(pred_mask, target)

                        # Skip first frame (conditioning frame) for loss - it doesn't go through MoE
                        # Memory attention (with MoE) is only used for non-conditioning frames
                        if is_cond:
                            continue

                        losses.append(loss)
                        for k in batch_metrics:
                            batch_metrics[k] += metrics[k]
                        valid_frames += 1

                    except Exception as e:
                        logger.warning(f"Error in batch {b}, frame {t}: {e}")
                        continue

            if valid_frames == 0 or len(losses) == 0:
                continue

            # Stack and average losses to maintain gradient chain
            batch_loss = torch.stack(losses).mean()

            # Backward pass
            self.optimizer.zero_grad()
            batch_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.config.max_grad_norm,
            )

            self.optimizer.step()
            self.scheduler.step()

            # Update metrics
            for k in epoch_losses:
                epoch_losses[k] += batch_metrics[k]
            num_samples += valid_frames

            # Log to TensorBoard
            self.global_step += 1
            self.writer.add_scalar("Train/loss", batch_loss.item(), self.global_step)
            self.writer.add_scalar("Train/dice_loss", batch_metrics["dice"] / valid_frames, self.global_step)
            self.writer.add_scalar("Train/focal_loss", batch_metrics["focal"] / valid_frames, self.global_step)
            self.writer.add_scalar("Train/lr", self.scheduler.get_last_lr()[0], self.global_step)

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{batch_loss.item():.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.6f}",
            })

        # Average epoch losses
        if num_samples > 0:
            for k in epoch_losses:
                epoch_losses[k] /= num_samples

        return epoch_losses

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()

        val_metrics = {"dice": 0.0, "focal": 0.0, "total": 0.0, "iou": 0.0}
        num_samples = 0

        pbar = tqdm(self.val_loader, desc="Validation")

        for batch in pbar:
            frames = batch["frames"].to(self.device)
            masks = batch["masks"].to(self.device)
            points_list = batch["points"]

            B, T, C, H, W = frames.shape

            for b in range(B):
                output_dict = {
                    "cond_frame_outputs": {},
                    "non_cond_frame_outputs": {},
                }

                for t in range(T):
                    frame = frames[b, t]
                    target = masks[b, t]
                    points = points_list[b].to(self.device)

                    is_cond = (t == 0)

                    try:
                        pred_mask, _, output_dict = self.forward_frame(
                            frame=frame,
                            points=points,
                            output_dict=output_dict,
                            frame_idx=t,
                            is_cond_frame=is_cond,
                        )

                        _, metrics = self.compute_loss(pred_mask, target)

                        # Compute IoU
                        pred_binary = (torch.sigmoid(pred_mask) > 0.5).float()
                        if pred_binary.shape != target.shape:
                            target = F.interpolate(
                                target.unsqueeze(0).unsqueeze(0).float(),
                                size=pred_binary.shape,
                                mode="nearest",
                            ).squeeze(0).squeeze(0)

                        intersection = (pred_binary * target).sum()
                        union = pred_binary.sum() + target.sum() - intersection
                        iou = (intersection / (union + 1e-6)).item()

                        for k in ["dice", "focal", "total"]:
                            val_metrics[k] += metrics[k]
                        val_metrics["iou"] += iou
                        num_samples += 1

                    except Exception as e:
                        continue

        if num_samples > 0:
            for k in val_metrics:
                val_metrics[k] /= num_samples

        # Log to TensorBoard
        self.writer.add_scalar("Val/loss", val_metrics["total"], epoch)
        self.writer.add_scalar("Val/dice_loss", val_metrics["dice"], epoch)
        self.writer.add_scalar("Val/focal_loss", val_metrics["focal"], epoch)
        self.writer.add_scalar("Val/iou", val_metrics["iou"], epoch)

        return val_metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint (MoE parameters only)."""
        # Extract only MoE parameters
        moe_state_dict = {
            k: v for k, v in self.model.state_dict().items()
            if any(kw in k for kw in ["lora", "gate", "experts"])
        }

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": moe_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_val_iou": self.best_val_iou,
            "config": vars(self.config),
        }

        # Save latest
        torch.save(checkpoint, self.output_dir / "checkpoint_latest.pt")

        # Save epoch checkpoint
        if (epoch + 1) % self.config.save_every == 0:
            torch.save(checkpoint, self.output_dir / f"checkpoint_epoch_{epoch+1}.pt")

        # Save best
        if is_best:
            torch.save(checkpoint, self.output_dir / "checkpoint_best.pt")
            logger.info(f"  ✓ New best model (IoU: {self.best_val_iou:.4f})")

    def check_early_stopping(self, val_iou: float) -> bool:
        """
        Check if training should stop early.

        Args:
            val_iou: Current validation IoU

        Returns:
            True if training should stop, False otherwise
        """
        if val_iou > self.early_stopping_best_iou + self.early_stopping_min_delta:
            # Improvement found
            self.early_stopping_best_iou = val_iou
            self.early_stopping_counter = 0
            return False
        else:
            # No improvement
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.early_stopping_patience:
                logger.info(f"  ⚠ Early stopping triggered! No improvement for {self.early_stopping_patience} epochs.")
                return True
            return False

    def train(self):
        """Main training loop."""
        logger.info("=" * 60)
        logger.info("Starting SAM2-MoE Training")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Epochs: {self.config.epochs}")
        logger.info(f"Learning rate: {self.config.lr}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Early stopping patience: {self.early_stopping_patience}")
        logger.info("=" * 60)

        for epoch in range(self.config.epochs):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Check if best
            is_best = val_metrics["iou"] > self.best_val_iou
            if is_best:
                self.best_val_iou = val_metrics["iou"]

            # Save checkpoint
            self.save_checkpoint(epoch, is_best)

            # Log summary
            logger.info("")
            logger.info(f"Epoch {epoch+1}/{self.config.epochs}")
            logger.info(f"  Train Loss: {train_metrics['total']:.4f} (Dice: {train_metrics['dice']:.4f}, Focal: {train_metrics['focal']:.4f})")
            logger.info(f"  Val Loss:   {val_metrics['total']:.4f} (Dice: {val_metrics['dice']:.4f}, Focal: {val_metrics['focal']:.4f})")
            logger.info(f"  Val IoU:    {val_metrics['iou']:.4f}")
            logger.info(f"  LR:         {self.scheduler.get_last_lr()[0]:.6f}")
            logger.info(f"  Early Stop: {self.early_stopping_counter}/{self.early_stopping_patience}")

            # Check early stopping
            if self.check_early_stopping(val_metrics["iou"]):
                logger.info("")
                logger.info("=" * 60)
                logger.info(f"Training stopped early at epoch {epoch+1}")
                logger.info(f"Best Val IoU: {self.best_val_iou:.4f}")
                logger.info(f"Checkpoints saved to: {self.output_dir}")
                logger.info("=" * 60)
                self.writer.close()
                return

        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info(f"Best Val IoU: {self.best_val_iou:.4f}")
        logger.info(f"Checkpoints saved to: {self.output_dir}")
        logger.info("=" * 60)

        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train SAM2-MoE")

    # Model
    parser.add_argument("--model_config", type=str,
                        default="configs/sam2.1/sam2.1_hiera_b+_moe.yaml")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/sam2.1_hiera_base_plus.pt")
    parser.add_argument("--device", type=str, default="cuda")

    # Data
    parser.add_argument("--data_dir", type=str, default="data/OCID-dataset/ARID20")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_ratio", type=float, default=0.8)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Ratio of total steps for warmup (default: 0.1 = 10%%)")
    parser.add_argument("--min_lr_ratio", type=float, default=0.01,
                        help="Minimum LR as ratio of initial LR (default: 0.01)")

    # Early stopping
    parser.add_argument("--early_stopping_patience", type=int, default=15,
                        help="Stop training if no improvement for N epochs")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.001,
                        help="Minimum change to qualify as an improvement")

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/sam2_moe")

    args = parser.parse_args()

    trainer = SAM2MoETrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
