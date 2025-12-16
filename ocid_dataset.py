"""
OCID Dataset Loader for SAM2-MoE Training

Loads OCID (Object Clutter Indoor Dataset) for video object segmentation.
Each sequence contains RGB images and segmentation labels.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import random


class OCIDSequenceDataset(Dataset):
    """
    OCID Dataset for video object segmentation.

    Dataset structure:
        OCID-dataset/
        ├── ARID10/
        │   ├── table/top/mixed/seq03/
        │   │   ├── rgb/
        │   │   └── label/
        │   └── ...
        ├── ARID20/
        └── YCB10/
    """

    def __init__(
        self,
        root_dir: str = "data/OCID-dataset",
        split: str = "train",
        num_frames: int = 8,
        image_size: int = 1024,
        train_ratio: float = 0.8,
        random_seed: int = 42,
    ):
        """
        Args:
            root_dir: Path to OCID-dataset directory
            split: 'train' or 'val'
            num_frames: Number of frames to sample from each sequence
            image_size: Image size (will be resized to square)
            train_ratio: Ratio of sequences for training
            random_seed: Random seed for split
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_frames = num_frames
        self.image_size = image_size

        # Find all sequences
        self.sequences = self._find_sequences()

        # Train/val split
        random.seed(random_seed)
        random.shuffle(self.sequences)
        split_idx = int(len(self.sequences) * train_ratio)

        if split == "train":
            self.sequences = self.sequences[:split_idx]
        else:
            self.sequences = self.sequences[split_idx:]

        print(f"Loaded {len(self.sequences)} sequences for {split}")

    def _find_sequences(self) -> List[Path]:
        """Find all valid sequences in the dataset."""
        sequences = []

        # Search pattern: OCID-dataset/*/surface/view/category/seqXX/
        for dataset_type in ["ARID10", "ARID20", "YCB10"]:
            dataset_path = self.root_dir / dataset_type

            if not dataset_path.exists():
                continue

            # Recursively find directories containing 'rgb' and 'label' subdirs
            for seq_dir in dataset_path.rglob("seq*"):
                rgb_dir = seq_dir / "rgb"
                label_dir = seq_dir / "label"

                if rgb_dir.exists() and label_dir.exists():
                    # Check if there are matching files
                    rgb_files = sorted(list(rgb_dir.glob("*.png")))
                    label_files = sorted(list(label_dir.glob("*.png")))

                    if len(rgb_files) > 0 and len(label_files) > 0:
                        sequences.append(seq_dir)

        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def _get_frame_pairs(self, seq_dir: Path) -> List[Tuple[Path, Path]]:
        """Get matching RGB and label file pairs."""
        rgb_dir = seq_dir / "rgb"
        label_dir = seq_dir / "label"

        rgb_files = sorted(list(rgb_dir.glob("*.png")))
        label_files = sorted(list(label_dir.glob("*.png")))

        # Match by filename
        pairs = []
        for rgb_file in rgb_files:
            label_file = label_dir / rgb_file.name
            if label_file.exists():
                pairs.append((rgb_file, label_file))

        return pairs

    def _sample_frames(
        self,
        pairs: List[Tuple[Path, Path]],
        num_frames: int
    ) -> List[Tuple[Path, Path]]:
        """Sample frames from sequence."""
        if len(pairs) <= num_frames:
            return pairs

        # Evenly sample frames
        indices = np.linspace(0, len(pairs) - 1, num_frames, dtype=int)
        return [pairs[i] for i in indices]

    def _load_image(self, path: Path) -> torch.Tensor:
        """Load and preprocess RGB image."""
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW
        return img

    def _load_mask(self, path: Path) -> torch.Tensor:
        """Load and preprocess segmentation mask."""
        mask = Image.open(path).convert("L")  # Grayscale
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        mask = np.array(mask).astype(np.float32)

        # Binarize: 0 = background, 1 = object
        mask = (mask > 0).astype(np.float32)
        mask = torch.from_numpy(mask)
        return mask

    def _get_prompt_points(self, mask: torch.Tensor, num_points: int = 5) -> torch.Tensor:
        """
        Sample positive points from mask for prompting SAM2.

        Args:
            mask: Binary mask (H, W)
            num_points: Number of points to sample

        Returns:
            Points tensor (N, 2) in (x, y) format
        """
        # Find foreground pixels
        fg_coords = torch.nonzero(mask > 0.5)  # (N, 2) - (y, x)

        if len(fg_coords) == 0:
            # No foreground, return center point
            h, w = mask.shape
            return torch.tensor([[w // 2, h // 2]], dtype=torch.float32)

        # Randomly sample points
        num_points = min(num_points, len(fg_coords))
        indices = torch.randperm(len(fg_coords))[:num_points]
        sampled_coords = fg_coords[indices]

        # Convert to (x, y) format
        points = sampled_coords.flip(1).float()  # (y, x) -> (x, y)

        return points

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sequence sample.

        Returns:
            Dictionary containing:
                - frames: (T, C, H, W) RGB frames
                - masks: (T, H, W) binary masks
                - points: (N, 2) prompt points from first frame
                - seq_name: sequence name (string)
        """
        seq_dir = self.sequences[idx]

        # Get frame pairs
        pairs = self._get_frame_pairs(seq_dir)

        # Sample frames
        sampled_pairs = self._sample_frames(pairs, self.num_frames)

        # Load frames and masks
        frames = []
        masks = []

        for rgb_path, label_path in sampled_pairs:
            frame = self._load_image(rgb_path)
            mask = self._load_mask(label_path)

            frames.append(frame)
            masks.append(mask)

        frames = torch.stack(frames)  # (T, C, H, W)
        masks = torch.stack(masks)    # (T, H, W)

        # Get prompt points from first frame
        points = self._get_prompt_points(masks[0], num_points=5)

        return {
            "frames": frames,
            "masks": masks,
            "points": points,
            "seq_name": str(seq_dir.relative_to(self.root_dir)),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.

    Since sequences may have different lengths, we return lists.
    """
    return {
        "frames": torch.stack([item["frames"] for item in batch]),  # (B, T, C, H, W)
        "masks": torch.stack([item["masks"] for item in batch]),    # (B, T, H, W)
        "points": [item["points"] for item in batch],  # List of (N, 2)
        "seq_names": [item["seq_name"] for item in batch],
    }


def get_dataloaders(
    root_dir: str = "data/OCID-dataset",
    batch_size: int = 2,
    num_frames: int = 8,
    image_size: int = 1024,
    num_workers: int = 4,
    train_ratio: float = 0.8,
):
    """
    Create train and validation dataloaders.

    Args:
        root_dir: Path to OCID dataset
        batch_size: Batch size
        num_frames: Number of frames per sequence
        image_size: Image size
        num_workers: Number of dataloader workers
        train_ratio: Train/val split ratio

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = OCIDSequenceDataset(
        root_dir=root_dir,
        split="train",
        num_frames=num_frames,
        image_size=image_size,
        train_ratio=train_ratio,
    )

    val_dataset = OCIDSequenceDataset(
        root_dir=root_dir,
        split="val",
        num_frames=num_frames,
        image_size=image_size,
        train_ratio=train_ratio,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    print("Testing OCID Dataset Loader...")

    dataset = OCIDSequenceDataset(
        root_dir="data/OCID-dataset",
        split="train",
        num_frames=8,
        image_size=1024,
    )

    print(f"\nDataset size: {len(dataset)}")

    # Get first sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Sequence: {sample['seq_name']}")
    print(f"  Frames shape: {sample['frames'].shape}")
    print(f"  Masks shape: {sample['masks'].shape}")
    print(f"  Points shape: {sample['points'].shape}")
    print(f"  Num objects in first frame: {sample['masks'][0].max():.0f}")

    # Test dataloader
    print("\nTesting DataLoader...")
    train_loader, val_loader = get_dataloaders(
        batch_size=2,
        num_frames=4,
        num_workers=0,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Get first batch
    batch = next(iter(train_loader))
    print(f"\nBatch:")
    print(f"  Frames: {batch['frames'].shape}")  # (B, T, C, H, W)
    print(f"  Masks: {batch['masks'].shape}")    # (B, T, H, W)
    print(f"  Points: {len(batch['points'])} sequences")
    print(f"  Seq names: {batch['seq_names']}")

    print("\n✓ Dataset loader working correctly!")
