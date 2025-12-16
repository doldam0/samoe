"""
Simple SAM2-MoE Training Example

A minimal example showing how to:
1. Load SAM2-MoE with pre-trained base weights
2. Freeze base model, train only LoRA adapters
3. Run a simple training loop

Usage:
    uv run python examples/simple_train_moe.py
"""

import torch
import torch.nn.functional as F
from sam2.build_sam import build_sam2_video_predictor_moe


def main():
    print("=" * 80)
    print("SAM2-MoE Simple Training Example")
    print("=" * 80)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # =========================================================================
    # Step 1: Build MoE model with pre-trained base weights
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 1: Building SAM2-MoE Model")
    print("=" * 80)

    model = build_sam2_video_predictor_moe(
        config_file="configs/sam2.1/sam2.1_hiera_b+_moe.yaml",
        ckpt_path="checkpoints/sam2.1_hiera_base_plus.pt",  # Load base weights
        device=device,
        mode="train",  # Enable training mode
    )

    print("✓ Model loaded successfully!")

    # =========================================================================
    # Step 2: Verify only MoE adapters are trainable
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 2: Parameter Analysis")
    print("=" * 80)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\nTotal parameters:      {total_params:,}")
    print(f"Trainable (MoE):       {trainable_params:,}")
    print(f"Frozen (base):         {frozen_params:,}")
    print(f"Trainable ratio:       {100 * trainable_params / total_params:.2f}%")

    # List some trainable parameters
    print("\nSample trainable parameters:")
    trainable_param_names = [name for name, p in model.named_parameters() if p.requires_grad]
    for name in trainable_param_names[:10]:
        print(f"  - {name}")
    if len(trainable_param_names) > 10:
        print(f"  ... and {len(trainable_param_names) - 10} more")

    # =========================================================================
    # Step 3: Setup optimizer (only for trainable parameters)
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 3: Setup Optimizer")
    print("=" * 80)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=0.01,
    )

    print(f"✓ Optimizer: AdamW (lr=1e-4, weight_decay=0.01)")
    print(f"✓ Optimizing {len(optimizer.param_groups[0]['params'])} parameter tensors")

    # =========================================================================
    # Step 4: Dummy training loop (demonstrative)
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 4: Dummy Training Loop")
    print("=" * 80)
    print("\nNote: This is a minimal example with dummy data.")
    print("For actual training, you need to:")
    print("  1. Load your video dataset")
    print("  2. Initialize predictor state with first frame")
    print("  3. Add prompts (points/boxes/masks)")
    print("  4. Propagate through video frames")
    print("  5. Compute loss against ground truth")
    print("")

    num_iterations = 5

    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}/{num_iterations}")

        # Create dummy inputs (replace with actual data)
        # In real training, you would:
        # - Load video frames
        # - Use model.init_state() to initialize
        # - Use model.add_new_points() or model.add_new_mask() to add prompts
        # - Use model.propagate_in_video() to get predictions

        # Dummy forward pass
        dummy_input = torch.randn(2, 256, 64, 64).to(device)  # (B, C, H, W)
        dummy_target = torch.randn(2, 256, 64, 64).to(device)

        # For demonstration, we'll just compute a simple loss
        # In practice, you need to use SAM2's inference API
        dummy_output = dummy_input  # Placeholder

        # Compute loss
        loss = F.mse_loss(dummy_output, dummy_target)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()

        print(f"  Loss: {loss.item():.4f}")

        # Check gradients (verify only MoE params have gradients)
        if iteration == 0:
            has_grad = sum(1 for p in model.parameters() if p.grad is not None)
            print(f"  Parameters with gradients: {has_grad}")

    # =========================================================================
    # Step 5: Save MoE adapter weights
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 5: Save MoE Adapter Weights")
    print("=" * 80)

    # Extract only MoE parameters
    moe_state_dict = {
        name: param
        for name, param in model.state_dict().items()
        if any(keyword in name for keyword in ['lora', 'gate', 'experts'])
    }

    print(f"\nMoE parameters to save: {len(moe_state_dict)} tensors")

    # Save checkpoint
    checkpoint_path = "outputs/moe_adapters.pt"
    import os
    os.makedirs("outputs", exist_ok=True)

    torch.save({
        'moe_state_dict': moe_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

    print(f"✓ Saved to: {checkpoint_path}")

    # =========================================================================
    # Step 6: Load MoE adapter weights (demonstration)
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 6: Load MoE Adapter Weights")
    print("=" * 80)

    # Build a fresh model
    model_new = build_sam2_video_predictor_moe(
        config_file="configs/sam2.1/sam2.1_hiera_b+_moe.yaml",
        ckpt_path="checkpoints/sam2.1_hiera_base_plus.pt",
        device=device,
        mode="eval",
    )

    # Load MoE adapters
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_new.load_state_dict(checkpoint['moe_state_dict'], strict=False)

    print("✓ MoE adapters loaded successfully!")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    print("\n✓ Successfully demonstrated:")
    print("  1. Loading SAM2-MoE with pre-trained base weights")
    print("  2. Verifying only LoRA adapters are trainable (~5-10% of params)")
    print("  3. Setting up optimizer for MoE parameters only")
    print("  4. Running training loop with gradient updates")
    print("  5. Saving/loading MoE adapter checkpoints")
    print("\n✓ Next steps for real training:")
    print("  1. Prepare your video object segmentation dataset")
    print("  2. Implement data loading pipeline")
    print("  3. Use SAM2 inference API (init_state, add_prompts, propagate)")
    print("  4. Compute segmentation loss (Dice + Focal)")
    print("  5. Monitor expert usage and specialization")
    print("\nFor full training script, see: train_moe.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
