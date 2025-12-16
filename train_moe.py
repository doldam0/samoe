"""
SAM2-MoE Fine-tuning Script

This script fine-tunes SAM2 with MoE-LoRA adapters on video object segmentation tasks.
Only the LoRA adapters and gating networks are trainable, while the base model remains frozen.

Usage:
    uv run python train_moe.py --config configs/training/train_moe_config.yaml
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from sam2.build_sam import build_sam2_video_predictor_moe


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SAM2MoETrainer:
    """
    Trainer for SAM2 with MoE-LoRA adapters.

    Features:
    - Loads pre-trained SAM2 base weights
    - Freezes base model, trains only LoRA adapters
    - Supports gradient accumulation
    - Expert usage monitoring
    - Checkpoint saving/loading
    """

    def __init__(
        self,
        config_file: str = "configs/sam2.1/sam2.1_hiera_b+_moe.yaml",
        ckpt_path: str = "checkpoints/sam2.1_hiera_base_plus.pt",
        device: str = "cuda",
        output_dir: str = "outputs/moe_training",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 10,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 500,
        save_steps: int = 1000,
        logging_steps: int = 100,
        max_grad_norm: float = 1.0,
    ):
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.max_grad_norm = max_grad_norm

        logger.info("=" * 80)
        logger.info("Building SAM2-MoE Model")
        logger.info("=" * 80)

        # Build model
        self.model = build_sam2_video_predictor_moe(
            config_file=config_file,
            ckpt_path=ckpt_path,
            device=device,
            mode="train",  # Training mode
        )

        # Print parameter statistics
        self._print_parameter_stats()

        # Setup optimizer (only for trainable parameters)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=learning_rate * 0.01,
        )

        self.global_step = 0
        self.current_epoch = 0

        logger.info(f"Optimizer: AdamW (lr={learning_rate}, wd={weight_decay})")
        logger.info(f"Scheduler: CosineAnnealingLR")
        logger.info(f"Output directory: {self.output_dir}")

    def _print_parameter_stats(self):
        """Print model parameter statistics."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        logger.info("")
        logger.info("=" * 80)
        logger.info("Model Parameter Statistics")
        logger.info("=" * 80)
        logger.info(f"Total parameters:      {total_params:,}")
        logger.info(f"Trainable (MoE):       {trainable_params:,}")
        logger.info(f"Frozen (base):         {frozen_params:,}")
        logger.info(f"Trainable ratio:       {100 * trainable_params / total_params:.2f}%")
        logger.info("=" * 80)
        logger.info("")

        # Count MoE modules
        moe_modules = 0
        for name, module in self.model.named_modules():
            if "MoERoPEAttention" in str(type(module).__name__):
                moe_modules += 1
        logger.info(f"MoE attention modules: {moe_modules}")
        logger.info("")

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        pred_iou: Optional[torch.Tensor] = None,
        target_iou: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.

        Args:
            predictions: Predicted masks (B, N, H, W)
            targets: Target masks (B, N, H, W)
            pred_iou: Predicted IoU scores (optional)
            target_iou: Target IoU scores (optional)

        Returns:
            Dictionary containing loss components
        """
        # Dice + Focal loss for mask prediction
        mask_loss = self._compute_mask_loss(predictions, targets)

        total_loss = mask_loss
        losses = {"mask_loss": mask_loss}

        # IoU prediction loss (if available)
        if pred_iou is not None and target_iou is not None:
            iou_loss = F.mse_loss(pred_iou, target_iou)
            total_loss = total_loss + 0.1 * iou_loss
            losses["iou_loss"] = iou_loss

        losses["total_loss"] = total_loss
        return losses

    def _compute_mask_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Dice + Focal loss for mask prediction."""
        # Dice loss
        dice_loss = self._dice_loss(predictions, targets)

        # Focal loss
        focal_loss = self._focal_loss(predictions, targets)

        # Combine
        return dice_loss + focal_loss

    def _dice_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        smooth: float = 1.0,
    ) -> torch.Tensor:
        """Compute Dice loss."""
        predictions = torch.sigmoid(predictions)

        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()
        dice = (2.0 * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)

        return 1.0 - dice

    def _focal_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> torch.Tensor:
        """Compute Focal loss."""
        predictions = torch.sigmoid(predictions)

        bce_loss = F.binary_cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss

        return focal_loss.mean()

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Training batch containing:
                - frames: Video frames (B, T, C, H, W)
                - masks: Target masks (B, T, H, W)
                - points: Point prompts (optional)

        Returns:
            Dictionary of loss values
        """
        # This is a simplified example - actual implementation depends on your data format
        # You'll need to adapt this based on your specific dataset and task

        frames = batch["frames"].to(self.device)
        masks = batch["masks"].to(self.device)

        # Forward pass through SAM2-MoE
        # NOTE: This is a placeholder - you need to implement actual inference logic
        # based on your specific use case (video tracking, segmentation, etc.)

        # Example: Initialize predictor state, add points, propagate
        # See SAM2 documentation for actual usage

        # For now, compute a dummy loss
        # In practice, you would:
        # 1. Initialize inference state
        # 2. Add prompts (points/boxes/masks)
        # 3. Propagate through video
        # 4. Compute loss against ground truth

        # Placeholder loss computation
        predictions = torch.randn_like(masks)  # Replace with actual predictions
        losses = self.compute_loss(predictions, masks)

        return {k: v.item() for k, v in losses.items()}

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_losses = {}
        num_batches = len(dataloader)

        progress_bar = tqdm(dataloader, desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            # Training step
            losses = self.train_step(batch)

            # Backward pass
            loss = losses["total_loss"]
            loss_tensor = torch.tensor(loss).to(self.device)
            loss_tensor = loss_tensor / self.gradient_accumulation_steps
            loss_tensor.backward()

            # Accumulate losses
            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0.0) + v

            # Optimizer step (with gradient accumulation)
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                # Logging
                if self.global_step % self.logging_steps == 0:
                    avg_losses = {k: v / self.logging_steps for k, v in total_losses.items()}
                    progress_bar.set_postfix(avg_losses)
                    total_losses = {}

                # Save checkpoint
                if self.global_step % self.save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{self.global_step}")

        # Average losses over epoch
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        return avg_losses

    def train(self, dataloader: DataLoader):
        """Main training loop."""
        logger.info("=" * 80)
        logger.info("Starting Training")
        logger.info("=" * 80)
        logger.info(f"Epochs: {self.num_epochs}")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        logger.info(f"Total batches per epoch: {len(dataloader)}")
        logger.info("=" * 80)
        logger.info("")

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            # Train epoch
            epoch_losses = self.train_epoch(dataloader)

            # Learning rate step
            self.scheduler.step()

            # Log epoch summary
            logger.info("")
            logger.info(f"Epoch {epoch + 1} Summary:")
            for k, v in epoch_losses.items():
                logger.info(f"  {k}: {v:.4f}")
            logger.info(f"  Learning rate: {self.scheduler.get_last_lr()[0]:.6f}")
            logger.info("")

            # Save epoch checkpoint
            self.save_checkpoint(f"checkpoint-epoch-{epoch + 1}")

        logger.info("=" * 80)
        logger.info("Training Completed!")
        logger.info("=" * 80)

    def save_checkpoint(self, checkpoint_name: str):
        """Save training checkpoint."""
        checkpoint_path = self.output_dir / f"{checkpoint_name}.pt"

        # Only save trainable parameters (MoE adapters)
        moe_state_dict = {
            name: param
            for name, param in self.model.state_dict().items()
            if any(keyword in name for keyword in ['lora', 'gate', 'experts'])
        }

        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': moe_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load MoE parameters
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']

        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")


def create_dummy_dataloader(batch_size: int = 2, num_batches: int = 100):
    """
    Create dummy dataloader for testing.

    In practice, replace this with your actual video dataset.
    """
    class DummyDataset:
        def __init__(self, num_samples):
            self.num_samples = num_samples

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Dummy data: batch of video frames and masks
            return {
                "frames": torch.randn(4, 3, 1024, 1024),  # 4 frames, RGB, 1024x1024
                "masks": torch.randint(0, 2, (4, 1024, 1024)).float(),  # Binary masks
            }

    dataset = DummyDataset(num_batches * batch_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    return dataloader


def main():
    parser = argparse.ArgumentParser(description="Train SAM2 with MoE-LoRA adapters")

    # Model args
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_b+_moe.yaml",
        help="Path to MoE config file",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="checkpoints/sam2.1_hiera_base_plus.pt",
        help="Path to pre-trained SAM2 checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )

    # Training args
    parser.add_argument("--output_dir", type=str, default="outputs/moe_training")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Data args
    parser.add_argument("--use_dummy_data", action="store_true", help="Use dummy data for testing")

    args = parser.parse_args()

    # Create trainer
    trainer = SAM2MoETrainer(
        config_file=args.config_file,
        ckpt_path=args.ckpt_path,
        device=args.device,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        max_grad_norm=args.max_grad_norm,
    )

    # Create dataloader
    if args.use_dummy_data:
        logger.info("Using dummy data for testing")
        dataloader = create_dummy_dataloader(batch_size=args.batch_size)
    else:
        # TODO: Implement your actual dataloader here
        raise NotImplementedError(
            "Please implement your own dataloader or use --use_dummy_data flag"
        )

    # Train
    trainer.train(dataloader)


if __name__ == "__main__":
    main()
