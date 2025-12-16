"""
SAM2 with Mixture of Prompt Experts (MoPE) - Demo Script

This script demonstrates how to use SAM2 with MoE-LoRA adapters
for video object segmentation with reduced catastrophic forgetting.

Usage:
    python examples/sam2_moe_demo.py
"""

import torch
import numpy as np
from sam2.build_sam import build_sam2_video_predictor_moe


def main():
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Build SAM2-MoE model
    print("\n" + "="*60)
    print("Building SAM2 with Mixture of Prompt Experts (MoPE)")
    print("="*60)

    # Option 1: Build from scratch (random initialization)
    print("\n[Option 1] Building MoE model from scratch...")
    predictor = build_sam2_video_predictor_moe(
        config_file="configs/sam2.1/sam2.1_hiera_b+_moe.yaml",
        ckpt_path=None,  # No checkpoint, random init
        device=device,
        mode="eval",
    )
    print("✓ Model built successfully!")

    # Option 2: Build with pre-trained base weights
    # Uncomment to load base SAM2 weights (LoRA adapters will be random)
    # print("\n[Option 2] Building MoE model with pre-trained base weights...")
    # predictor = build_sam2_video_predictor_moe(
    #     config_file="configs/sam2.1/sam2.1_hiera_b+_moe.yaml",
    #     ckpt_path="checkpoints/sam2.1_hiera_base_plus.pt",
    #     device=device,
    #     mode="eval",
    # )
    # print("✓ Model built with pre-trained weights!")

    # Print model architecture summary
    print("\n" + "="*60)
    print("Model Architecture Summary")
    print("="*60)

    # Count parameters
    total_params = sum(p.numel() for p in predictor.parameters())
    trainable_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters (MoE adapters): {trainable_params:,}")
    print(f"Frozen parameters (base model): {frozen_params:,}")
    print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")

    # Analyze MoE structure
    print("\n" + "="*60)
    print("MoE Structure Analysis")
    print("="*60)

    moe_modules = []
    for name, module in predictor.named_modules():
        if "MoERoPEAttention" in str(type(module).__name__):
            moe_modules.append((name, module))

    print(f"\nFound {len(moe_modules)} MoE-enhanced attention modules:")
    for name, module in moe_modules:
        print(f"\n  Module: {name}")
        print(f"    - Num experts: {module.num_experts}")
        print(f"    - LoRA rank: {module.lora_rank}")
        print(f"    - Top-k: {module.top_k}")

    # Example: Analyze expert usage (requires dummy input)
    print("\n" + "="*60)
    print("Expert Usage Analysis (with dummy input)")
    print("="*60)

    if len(moe_modules) > 0:
        # Create dummy input
        batch_size = 1
        seq_len = 64 * 64  # For 64x64 feature map
        d_model = 256

        dummy_q = torch.randn(batch_size, seq_len, d_model).to(device)
        dummy_k = torch.randn(batch_size, seq_len, d_model).to(device)
        dummy_v = torch.randn(batch_size, seq_len, d_model).to(device)

        # Get the first MoE attention module
        _, first_moe = moe_modules[0]

        with torch.no_grad():
            stats = first_moe.get_expert_statistics(dummy_q, dummy_k, dummy_v)

        print(f"\nExpert weights for q_proj (averaged over batch/sequence):")
        for i, weight in enumerate(stats['q_proj']):
            print(f"  Expert {i}: {weight.item():.4f}")

        print(f"\nExpert weights for k_proj (averaged over batch/sequence):")
        for i, weight in enumerate(stats['k_proj']):
            print(f"  Expert {i}: {weight.item():.4f}")

        print(f"\nExpert weights for v_proj (averaged over batch/sequence):")
        for i, weight in enumerate(stats['v_proj']):
            print(f"  Expert {i}: {weight.item():.4f}")

    # Model info
    print("\n" + "="*60)
    print("Model Ready for Training/Inference!")
    print("="*60)
    print("\nKey Features:")
    print("  ✓ Base model weights are frozen")
    print("  ✓ Only LoRA adapters and gating networks are trainable")
    print("  ✓ Top-k expert routing reduces computation")
    print("  ✓ Domain-specific experts mitigate catastrophic forgetting")
    print("\nNext Steps:")
    print("  1. Fine-tune on your specific domain/dataset")
    print("  2. Monitor expert specialization patterns")
    print("  3. Evaluate on multi-domain benchmarks")

    return predictor


if __name__ == "__main__":
    predictor = main()
    print("\n✓ Demo completed successfully!")
