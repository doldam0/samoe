"""
OCID Dataset Loader for SAM2-MoE Training

Loads OCID (Object Clutter Indoor Dataset) for video object segmentation.
Each sequence contains RGB images and instance segmentation labels.

Instance-aware: Each sample tracks a single object instance across frames.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class OCIDInstanceDataset(Dataset):
    """
    OCID Dataset for instance-aware video object segmentation.

    Key difference from binary segmentation:
    - Each sample tracks ONE specific object instance
    - Points are sampled from that specific instance
    - Mask is binary for that instance only

    Dataset structure:
        OCID-dataset/
        ├── ARID10/
        │   ├── table/top/mixed/seq03/
        │   │   ├── rgb/
        │   │   └── label/  (instance IDs: 0=bg, 1,2,3...=objects)
        │   └── ...
        ├── ARID20/
        └── YCB10/
    """

    def __init__(
        self,
        root_dir: str = "data/OCID-dataset/ARID20",
        split: str = "train",
        num_frames: int = 8,
        image_size: int = 1024,
        train_ratio: float = 0.8,
        random_seed: int = 42,
        test_sequences: List[str] = None,
        num_points: int = 5,
    ):
        """
        Args:
            root_dir: Path to OCID-dataset directory
            split: 'train', 'val', or 'test'
            num_frames: Number of frames to sample from each sequence
            image_size: Image size (will be resized to square)
            train_ratio: Ratio of train sequences (from non-test sequences)
            random_seed: Random seed for train/val split
            test_sequences: List of sequence names for test (e.g., ["seq13"])
                           If None, defaults to ["seq13"]
            num_points: Number of prompt points to sample per instance
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_frames = num_frames
        self.image_size = image_size
        self.num_points = num_points

        # Default test sequences
        if test_sequences is None:
            test_sequences = ["seq13"]
        self.test_sequences = test_sequences

        # Find all sequences
        all_sequences = self._find_sequences()

        # Split into test and train/val
        test_seqs = []
        trainval_seqs = []

        for seq in all_sequences:
            seq_name = seq.name  # e.g., "seq13"
            if seq_name in self.test_sequences:
                test_seqs.append(seq)
            else:
                trainval_seqs.append(seq)

        if split == "test":
            sequences = test_seqs
        else:
            # Train/val split from non-test sequences
            random.seed(random_seed)
            random.shuffle(trainval_seqs)
            split_idx = int(len(trainval_seqs) * train_ratio)

            if split == "train":
                sequences = trainval_seqs[:split_idx]
            else:  # val
                sequences = trainval_seqs[split_idx:]

        # Build instance-aware samples: (sequence, object_id)
        self.samples = self._build_instance_samples(sequences)
        print(f"Loaded {len(self.samples)} instance samples from {len(sequences)} sequences for {split}")

    def _find_sequences(self) -> List[Path]:
        """Find all valid sequences in the dataset."""
        sequences = []

        # Search pattern: OCID-dataset/*/surface/view/category/seqXX/
        dataset_path = self.root_dir

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

    def _build_instance_samples(self, sequences: List[Path]) -> List[Tuple[Path, int]]:
        """
        Build list of (sequence, object_id) pairs.

        For each sequence, find all unique object IDs and create a sample for each.
        """
        samples = []

        for seq_dir in sequences:
            # Get the last frame's label to find all objects
            label_dir = seq_dir / "label"
            label_files = sorted(list(label_dir.glob("*.png")))

            if not label_files:
                continue

            # Use last frame to get maximum number of objects
            last_label = np.array(Image.open(label_files[-1]))
            unique_ids = np.unique(last_label)

            # Filter out background (0)
            object_ids = [int(oid) for oid in unique_ids if oid > 0]

            for obj_id in object_ids:
                samples.append((seq_dir, obj_id))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _get_frame_pairs(self, seq_dir: Path) -> List[Tuple[Path, Path]]:
        """Get matching RGB and label file pairs."""
        rgb_dir = seq_dir / "rgb"
        label_dir = seq_dir / "label"

        rgb_files = sorted(list(rgb_dir.glob("*.png")))

        # Match by filename
        pairs = []
        for rgb_file in rgb_files:
            label_file = label_dir / rgb_file.name
            if label_file.exists():
                pairs.append((rgb_file, label_file))

        return pairs

    def _sample_frames_for_instance(
        self,
        pairs: List[Tuple[Path, Path]],
        object_id: int,
        num_frames: int,
    ) -> List[Tuple[Path, Path, int]]:
        """
        Sample frames where the target object exists.

        Returns list of (rgb_path, label_path, start_frame_idx) tuples.
        start_frame_idx indicates when this object first appears.
        """
        # Find frames where object exists
        valid_pairs = []
        for i, (rgb_path, label_path) in enumerate(pairs):
            label = np.array(Image.open(label_path))
            if object_id in label:
                valid_pairs.append((rgb_path, label_path, i))

        if len(valid_pairs) == 0:
            return []

        if len(valid_pairs) >= num_frames:
            # Evenly sample from valid frames
            indices = np.linspace(0, len(valid_pairs) - 1, num_frames, dtype=int)
            return [valid_pairs[i] for i in indices]

        # Pad by repeating frames to reach num_frames
        # This ensures consistent tensor sizes for batching
        result = []
        for i in range(num_frames):
            idx = i % len(valid_pairs)
            result.append(valid_pairs[idx])
        return result

    def _load_image(self, path: Path) -> torch.Tensor:
        """Load and preprocess RGB image."""
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW
        return img

    def _load_instance_mask(self, path: Path, object_id: int) -> torch.Tensor:
        """
        Load instance segmentation mask for a specific object.

        Args:
            path: Path to label image
            object_id: Target object ID

        Returns:
            Binary mask (H, W) where 1 = target object, 0 = everything else
        """
        # Load as 16-bit to handle large instance IDs
        label = Image.open(path)
        label = label.resize((self.image_size, self.image_size), Image.NEAREST)
        label = np.array(label)

        # Create binary mask for this specific instance
        mask = (label == object_id).astype(np.float32)
        mask = torch.from_numpy(mask)
        return mask

    def _get_prompt_points(
        self, mask: torch.Tensor, num_points: int = 5
    ) -> torch.Tensor:
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

    def __getitem__(self, idx: int) -> Dict:
        """
        Get an instance-aware sequence sample.

        Returns:
            Dictionary containing:
                - frames: (T, C, H, W) RGB frames
                - masks: (T, H, W) binary masks for the specific instance
                - points: (N, 2) prompt points from first frame
                - seq_name: sequence name (string)
                - object_id: target object ID (int)
        """
        seq_dir, object_id = self.samples[idx]

        # Get frame pairs
        pairs = self._get_frame_pairs(seq_dir)

        # Sample frames where object exists
        sampled_data = self._sample_frames_for_instance(pairs, object_id, self.num_frames)

        if len(sampled_data) == 0:
            # Fallback: object doesn't exist, return empty sample
            # This shouldn't happen if _build_instance_samples is correct
            frames = torch.zeros(1, 3, self.image_size, self.image_size)
            masks = torch.zeros(1, self.image_size, self.image_size)
            points = torch.tensor([[self.image_size // 2, self.image_size // 2]], dtype=torch.float32)
            return {
                "frames": frames,
                "masks": masks,
                "points": points,
                "seq_name": str(seq_dir.relative_to(self.root_dir)),
                "object_id": object_id,
            }

        # Load frames and masks
        frames = []
        masks = []

        for rgb_path, label_path, _ in sampled_data:
            frame = self._load_image(rgb_path)
            mask = self._load_instance_mask(label_path, object_id)

            frames.append(frame)
            masks.append(mask)

        frames = torch.stack(frames)  # (T, C, H, W)
        masks = torch.stack(masks)  # (T, H, W)

        # Get prompt points from first frame's mask
        points = self._get_prompt_points(masks[0], num_points=self.num_points)

        return {
            "frames": frames,
            "masks": masks,
            "points": points,
            "seq_name": str(seq_dir.relative_to(self.root_dir)),
            "object_id": object_id,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for DataLoader.

    Since sequences may have different lengths, we return lists.
    """
    return {
        "frames": torch.stack(
            [item["frames"] for item in batch]
        ),  # (B, T, C, H, W)
        "masks": torch.stack([item["masks"] for item in batch]),  # (B, T, H, W)
        "points": [item["points"] for item in batch],  # List of (N, 2)
        "seq_names": [item["seq_name"] for item in batch],
        "object_ids": [item["object_id"] for item in batch],  # List of int
    }


def get_dataloaders(
    root_dir: str = "data/OCID-dataset",
    batch_size: int = 2,
    num_frames: int = 8,
    image_size: int = 1024,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    test_sequences: List[str] = None,
    include_test: bool = False,
    num_points: int = 5,
):
    """
    Create train, validation, and optionally test dataloaders.

    Args:
        root_dir: Path to OCID dataset
        batch_size: Batch size
        num_frames: Number of frames per sequence
        image_size: Image size
        num_workers: Number of dataloader workers
        train_ratio: Train/val split ratio (from non-test sequences)
        test_sequences: List of sequence names for test (default: ["seq13"])
        include_test: If True, returns (train_loader, val_loader, test_loader)
        num_points: Number of prompt points per instance

    Returns:
        (train_loader, val_loader) or (train_loader, val_loader, test_loader)
    """
    train_dataset = OCIDInstanceDataset(
        root_dir=root_dir,
        split="train",
        num_frames=num_frames,
        image_size=image_size,
        train_ratio=train_ratio,
        test_sequences=test_sequences,
        num_points=num_points,
    )

    val_dataset = OCIDInstanceDataset(
        root_dir=root_dir,
        split="val",
        num_frames=num_frames,
        image_size=image_size,
        train_ratio=train_ratio,
        test_sequences=test_sequences,
        num_points=num_points,
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

    if include_test:
        test_dataset = OCIDInstanceDataset(
            root_dir=root_dir,
            split="test",
            num_frames=num_frames,
            image_size=image_size,
            test_sequences=test_sequences,
            num_points=num_points,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        return train_loader, val_loader, test_loader

    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    print("Testing OCID Instance Dataset Loader...")

    dataset = OCIDInstanceDataset(
        root_dir="data/OCID-dataset/ARID20/table/bottom",
        split="train",
        num_frames=8,
        image_size=1024,
    )

    print(f"\nDataset size: {len(dataset)} instance samples")

    # Get first sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Sequence: {sample['seq_name']}")
    print(f"  Object ID: {sample['object_id']}")
    print(f"  Frames shape: {sample['frames'].shape}")
    print(f"  Masks shape: {sample['masks'].shape}")
    print(f"  Points shape: {sample['points'].shape}")
    print(f"  Mask is binary: min={sample['masks'].min():.0f}, max={sample['masks'].max():.0f}")

    # Test dataloader
    print("\nTesting DataLoader...")
    train_loader, val_loader = get_dataloaders(
        root_dir="data/OCID-dataset/ARID20/table/bottom",
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
    print(f"  Masks: {batch['masks'].shape}")  # (B, T, H, W)
    print(f"  Points: {len(batch['points'])} instances")
    print(f"  Seq names: {batch['seq_names']}")
    print(f"  Object IDs: {batch['object_ids']}")

    print("\n✓ Instance-aware dataset loader working correctly!")
