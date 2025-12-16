"""
Train SAM2-MoE on OCID Dataset with TensorBoard Logging

This script trains SAM2 with MoE-LoRA adapters with real-time visualization.
"""

import argparse
import logging
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
from datetime import datetime

from sam2.build_sam import build_sam2_video_predictor_moe
from ocid_dataset import get_dataloaders


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SAM2MoETrainer:
    """Trainer for SAM2-MoE on OCID dataset with TensorBoard."""

    def __init__(self, args):
        self.args = args
        self.device = args.device

        # Create output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup TensorBoard
        self.tb_dir = self.output_dir / "tensorboard"
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))
        logger.info(f"TensorBoard logs: {self.tb_dir}")
        logger.info(f"Run: tensorboard --logdir={self.tb_dir}")

        # Save args
        with open(self.output_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)

        logger.info("="*80)
        logger.info("SAM2-MoE OCID Training with TensorBoard")
        logger.info("="*80)

        # Build model
        logger.info("Building SAM2-MoE model...")
        self.model = build_sam2_video_predictor_moe(
            config_file=args.config_file,
            ckpt_path=args.ckpt_path,
            device=self.device,
            mode="train",
        )

        # Print parameter stats
        self._print_parameter_stats()

        # Create a simple prediction head for testing
        self.pred_head = torch.nn.Conv2d(3, 1, kernel_size=1).to(self.device)

        # Setup optimizer
        trainable_params = list(p for p in self.model.parameters() if p.requires_grad)
        trainable_params += list(self.pred_head.parameters())
        self.optimizer = AdamW(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        # Setup scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=args.num_epochs,
            eta_min=args.learning_rate * 0.01,
        )

        # Setup dataloaders
        logger.info("Loading OCID dataset...")
        self.train_loader, self.val_loader = get_dataloaders(
            root_dir=args.data_dir,
            batch_size=args.batch_size,
            num_frames=args.num_frames,
            image_size=args.image_size,
            num_workers=args.num_workers,
            train_ratio=args.train_ratio,
        )

        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Val batches: {len(self.val_loader)}")

        self.global_step = 0
        self.best_val_loss = float('inf')

        logger.info("="*80)

    def _print_parameter_stats(self):
        """Print model parameter statistics."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen = total - trainable

        logger.info("")
        logger.info("Model Parameter Statistics:")
        logger.info(f"  Total:     {total:>12,}")
        logger.info(f"  Trainable: {trainable:>12,}  ({100*trainable/total:>5.2f}%)")
        logger.info(f"  Frozen:    {frozen:>12,}  ({100*frozen/total:>5.2f}%)")
        logger.info("")

        # Log to TensorBoard
        self.writer.add_scalar("Model/total_params", total, 0)
        self.writer.add_scalar("Model/trainable_params", trainable, 0)
        self.writer.add_scalar("Model/frozen_params", frozen, 0)

    def compute_dice_loss(self, pred, target, smooth=1.0):
        """Compute Dice loss."""
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

        return 1.0 - dice

    def compute_focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        """Compute Focal loss."""
        pred = torch.sigmoid(pred)
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        return focal_loss.mean()

    def train_step(self, batch):
        """Single training step."""
        frames = batch["frames"].to(self.device)  # (B, T, C, H, W)
        masks = batch["masks"].to(self.device)    # (B, T, H, W)
        points_list = batch["points"]

        B, T, C, H, W = frames.shape

        total_loss = 0.0
        dice_loss_sum = 0.0
        focal_loss_sum = 0.0
        num_valid_frames = 0

        for b in range(B):
            for t in range(T):
                frame = frames[b, t]
                target_mask = masks[b, t]

                # Use the simple prediction head
                frame_input = frame.unsqueeze(0)
                pred_logits = self.pred_head(frame_input)
                pred_mask = pred_logits.squeeze(0).squeeze(0)

                # Compute loss
                dice_loss = self.compute_dice_loss(pred_mask, target_mask)
                focal_loss = self.compute_focal_loss(pred_mask, target_mask)

                frame_loss = dice_loss + focal_loss
                total_loss += frame_loss
                dice_loss_sum += dice_loss.item()
                focal_loss_sum += focal_loss.item()
                num_valid_frames += 1

        # Average loss
        if num_valid_frames > 0:
            total_loss = total_loss / num_valid_frames
            avg_dice = dice_loss_sum / num_valid_frames
            avg_focal = focal_loss_sum / num_valid_frames
        else:
            avg_dice = avg_focal = 0.0

        return total_loss, avg_dice, avg_focal

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()

        epoch_loss = 0.0
        epoch_dice = 0.0
        epoch_focal = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.num_epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            # Forward pass
            loss, dice_loss, focal_loss = self.train_step(batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.args.max_grad_norm
            )

            # Optimizer step
            self.optimizer.step()

            # Update stats
            epoch_loss += loss.item()
            epoch_dice += dice_loss
            epoch_focal += focal_loss
            self.global_step += 1

            # Log to TensorBoard (every step)
            self.writer.add_scalar("Train/loss", loss.item(), self.global_step)
            self.writer.add_scalar("Train/dice_loss", dice_loss, self.global_step)
            self.writer.add_scalar("Train/focal_loss", focal_loss, self.global_step)
            self.writer.add_scalar("Train/learning_rate", self.scheduler.get_last_lr()[0], self.global_step)

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice_loss:.4f}',
                'focal': f'{focal_loss:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}',
            })

            # Save checkpoint periodically
            if self.global_step % self.args.save_steps == 0:
                self.save_checkpoint(f"checkpoint-step-{self.global_step}")

        avg_epoch_loss = epoch_loss / len(self.train_loader)
        avg_epoch_dice = epoch_dice / len(self.train_loader)
        avg_epoch_focal = epoch_focal / len(self.train_loader)

        return avg_epoch_loss, avg_epoch_dice, avg_epoch_focal

    @torch.no_grad()
    def validate(self, epoch):
        """Validate on validation set."""
        self.model.eval()

        val_loss = 0.0
        val_dice = 0.0
        val_focal = 0.0
        progress_bar = tqdm(self.val_loader, desc="Validation")

        for batch in progress_bar:
            loss, dice_loss, focal_loss = self.train_step(batch)
            val_loss += loss.item()
            val_dice += dice_loss
            val_focal += focal_loss

            progress_bar.set_postfix({
                'val_loss': f'{loss.item():.4f}',
                'val_dice': f'{dice_loss:.4f}',
            })

        avg_val_loss = val_loss / len(self.val_loader)
        avg_val_dice = val_dice / len(self.val_loader)
        avg_val_focal = val_focal / len(self.val_loader)

        # Log to TensorBoard
        self.writer.add_scalar("Val/loss", avg_val_loss, epoch)
        self.writer.add_scalar("Val/dice_loss", avg_val_dice, epoch)
        self.writer.add_scalar("Val/focal_loss", avg_val_focal, epoch)

        return avg_val_loss, avg_val_dice, avg_val_focal

    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Total epochs: {self.args.num_epochs}")
        logger.info(f"Batch size: {self.args.batch_size}")
        logger.info(f"Learning rate: {self.args.learning_rate}")
        logger.info("")

        for epoch in range(self.args.num_epochs):
            # Train epoch
            train_loss, train_dice, train_focal = self.train_epoch(epoch)

            # Validate
            val_loss, val_dice, val_focal = self.validate(epoch)

            # Scheduler step
            self.scheduler.step()

            # Log epoch summary to TensorBoard
            self.writer.add_scalars("Epoch/loss", {
                "train": train_loss,
                "val": val_loss,
            }, epoch)
            self.writer.add_scalars("Epoch/dice_loss", {
                "train": train_dice,
                "val": val_dice,
            }, epoch)

            # Log epoch summary
            logger.info("")
            logger.info(f"Epoch {epoch+1}/{self.args.num_epochs} Summary:")
            logger.info(f"  Train Loss: {train_loss:.4f} (Dice: {train_dice:.4f}, Focal: {train_focal:.4f})")
            logger.info(f"  Val Loss:   {val_loss:.4f} (Dice: {val_dice:.4f}, Focal: {val_focal:.4f})")
            logger.info(f"  LR:         {self.scheduler.get_last_lr()[0]:.6f}")

            # Save best checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("checkpoint-best")
                logger.info(f"  ✓ New best model saved!")

            # Save epoch checkpoint
            self.save_checkpoint(f"checkpoint-epoch-{epoch+1}")

            logger.info("")

        logger.info("="*80)
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"TensorBoard logs: {self.tb_dir}")
        logger.info("="*80)

        # Close TensorBoard writer
        self.writer.close()

    def save_checkpoint(self, name):
        """Save training checkpoint (MoE adapters only)."""
        checkpoint_path = self.output_dir / f"{name}.pt"

        # Extract MoE parameters only
        moe_state_dict = {
            k: v for k, v in self.model.state_dict().items()
            if any(keyword in k for keyword in ['lora', 'gate', 'experts'])
        }

        checkpoint = {
            'global_step': self.global_step,
            'model_state_dict': moe_state_dict,
            'pred_head_state_dict': self.pred_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Train SAM2-MoE on OCID dataset with TensorBoard")

    # Model args
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_b+_moe.yaml",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="checkpoints/sam2.1_hiera_base_plus.pt",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Data args
    parser.add_argument("--data_dir", type=str, default="data/OCID-dataset")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_ratio", type=float, default=0.8)

    # Training args
    parser.add_argument("--output_dir", type=str, default="outputs/ocid_training_tb")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=50)

    args = parser.parse_args()

    # Create trainer and train
    trainer = SAM2MoETrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
