# SAM2 with Mixture of Prompt Experts (MoPE)

This implementation extends [SAM2 (Segment Anything Model 2)](https://github.com/facebookresearch/sam2) with **Mixture of Prompt Experts (MoPE)** to mitigate catastrophic forgetting in memory attention during fine-tuning.

## Overview

### The Problem: Catastrophic Forgetting in Memory Attention

During fine-tuning for domain-specific tasks, SAM2's memory attention can suffer from **catastrophic forgetting** due to domain interference. When the model is adapted to new domains or object types, it may lose its ability to segment objects from previously learned domains.

### The Solution: MoPE with LoRA Experts

We implement a **Mixture of Prompt Experts** architecture where:

1. **Base projections are frozen** - Original SAM2 weights remain unchanged
2. **Multiple LoRA experts** - 10 lightweight adapters per projection layer
3. **Learned routing** - Top-k gating network selects relevant experts
4. **Domain specialization** - Different experts can specialize for different domains/objects

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Memory Attention                      │
│                                                           │
│  ┌─────────────────────────────────────────────────┐   │
│  │        RoPE Self-Attention (MoE-enhanced)       │   │
│  │                                                   │   │
│  │  Input x                                          │   │
│  │    ├──> Base Q Proj (frozen) ──┐                │   │
│  │    │                             │                │   │
│  │    └──> Gating Network          │                │   │
│  │           ├─> Expert 1 (LoRA) ──┤                │   │
│  │           ├─> Expert 2 (LoRA) ──┤                │   │
│  │           │         ...          │──> Output     │   │
│  │           └─> Expert 10 (LoRA) ─┘                │   │
│  │                                                   │   │
│  │  (Similar for K, V, Out projections)            │   │
│  └─────────────────────────────────────────────────┘   │
│                                                           │
│  ┌─────────────────────────────────────────────────┐   │
│  │       RoPE Cross-Attention (MoE-enhanced)       │   │
│  │              (same structure)                    │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Key Features

### 1. **LoRA Adapters** ([moe_modules.py:LoRAAdapter](sam2/modeling/moe_modules.py))
- Low-rank adaptation: `W' = W + BA` where `W` is frozen
- Configurable rank (default: 4)
- Minimal parameter overhead

### 2. **Top-k Gating Network** ([moe_modules.py:TopKGatingNetwork](sam2/modeling/moe_modules.py))
- Learns to route inputs to appropriate experts
- Top-k selection (default: k=2) for efficiency
- Softmax normalization over selected experts

### 3. **MoE-LoRA Linear Layer** ([moe_modules.py:MoELoRALinear](sam2/modeling/moe_modules.py))
- Wraps any linear projection with MoE
- Output: `y = W_base(x) + Σ w_i * LoRA_i(x)`
- Efficient expert parallelization

### 4. **MoE-Enhanced Attention** ([moe_attention.py:MoERoPEAttention](sam2/modeling/moe_attention.py))
- Drop-in replacement for `RoPEAttention`
- MoE applied to Q, K, V, and output projections
- Compatible with all SAM2 features (RoPE, multi-head, etc.)

## Installation

The MoE implementation is integrated into the SAM2 codebase. No additional dependencies required beyond standard SAM2.

```bash
# Standard SAM2 installation
pip install -e .
```

## Usage

### Quick Start

```python
from sam2.build_sam import build_sam2_video_predictor_moe

# Build MoE model from scratch
predictor = build_sam2_video_predictor_moe(
    config_file="configs/sam2.1/sam2.1_hiera_b+_moe.yaml",
    ckpt_path=None,  # Random initialization
    device="cuda",
)

# Or with pre-trained base weights
predictor = build_sam2_video_predictor_moe(
    config_file="configs/sam2.1/sam2.1_hiera_b+_moe.yaml",
    ckpt_path="checkpoints/sam2.1_hiera_base_plus.pt",  # Load base weights
    device="cuda",
)
```

### Running the Demo

```bash
python examples/sam2_moe_demo.py
```

This will:
- Build the MoE model
- Print architecture summary
- Analyze parameter counts
- Show expert usage statistics

### Configuration

The MoE model is configured via YAML files in [configs/sam2.1/](sam2/configs/sam2.1/):

**[sam2.1_hiera_b+_moe.yaml](sam2/configs/sam2.1/sam2.1_hiera_b+_moe.yaml)**

Key parameters:
```yaml
self_attention:
  _target_: sam2.modeling.moe_attention.MoERoPEAttention
  # Standard attention params
  embedding_dim: 256
  num_heads: 1
  dropout: 0.1
  # MoE-specific params
  num_experts: 10      # Number of LoRA experts
  lora_rank: 4         # LoRA bottleneck dimension
  top_k: 2             # Number of experts to activate
  lora_alpha: 1.0      # LoRA scaling factor
  lora_dropout: 0.1    # Dropout in LoRA layers
```

## Training

### Parameter Efficiency

By default, **only MoE adapters are trainable**:

```python
# Check parameter counts
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")
# Expected: ~5-10% of parameters are trainable
```

### Fine-tuning Example

```python
# Build model in training mode
predictor = build_sam2_video_predictor_moe(
    config_file="configs/sam2.1/sam2.1_hiera_b+_moe.yaml",
    ckpt_path="checkpoints/sam2.1_hiera_base_plus.pt",
    device="cuda",
    mode="train",  # Enable training mode
)

# Only MoE parameters will have gradients
optimizer = torch.optim.AdamW(
    [p for p in predictor.parameters() if p.requires_grad],
    lr=1e-4,
)

# Standard training loop
for batch in dataloader:
    optimizer.zero_grad()
    loss = compute_loss(predictor, batch)
    loss.backward()
    optimizer.step()
```

## Expert Analysis

### Monitoring Expert Usage

```python
# Get expert statistics for a sample input
stats = attention_module.get_expert_statistics(q, k, v)

# Stats contains expert weights for each projection:
# - stats['q_proj']: shape (num_experts,)
# - stats['k_proj']: shape (num_experts,)
# - stats['v_proj']: shape (num_experts,)

# Analyze specialization
import matplotlib.pyplot as plt
plt.bar(range(10), stats['q_proj'].cpu().numpy())
plt.xlabel('Expert ID')
plt.ylabel('Average Weight')
plt.title('Expert Usage in Q Projection')
plt.show()
```

## File Structure

```
sam2/
├── modeling/
│   ├── moe_modules.py          # Core MoE components
│   │   ├── LoRAAdapter         # Low-rank adaptation
│   │   ├── TopKGatingNetwork   # Expert routing
│   │   └── MoELoRALinear       # MoE projection wrapper
│   │
│   ├── moe_attention.py        # MoE-enhanced attention
│   │   ├── MoERoPEAttention    # RoPE attention with MoE
│   │   └── convert_rope_attention_to_moe()
│   │
│   └── memory_attention.py     # Original memory attention
│
├── configs/sam2.1/
│   └── sam2.1_hiera_b+_moe.yaml  # MoE model config
│
├── build_sam.py                # Build utilities
│   ├── build_sam2_moe()
│   └── build_sam2_video_predictor_moe()
│
└── examples/
    └── sam2_moe_demo.py        # Demo script
```

## Hyperparameter Tuning

### Number of Experts (`num_experts`)
- **Default**: 10
- More experts → better specialization, but higher memory
- Typical range: 4-16

### LoRA Rank (`lora_rank`)
- **Default**: 4
- Higher rank → more capacity, but more parameters
- Typical range: 2-8

### Top-k (`top_k`)
- **Default**: 2
- More experts activated → better accuracy, but slower
- Typical range: 1-4

### LoRA Alpha (`lora_alpha`)
- **Default**: 1.0
- Scaling factor for LoRA updates
- Higher values → stronger adaptation

## Expected Benefits

1. **Reduced Catastrophic Forgetting**
   - Maintain performance on original domains while adapting to new ones
   - Different experts specialize for different object types/domains

2. **Parameter Efficiency**
   - Only 5-10% of parameters are trainable
   - Faster fine-tuning compared to full model
   - Lower memory footprint

3. **Domain Specialization**
   - Experts can learn domain-specific features
   - Automatic routing based on input

4. **Modular Adaptation**
   - Easy to add/remove experts
   - Can train experts independently for different domains

## Limitations & Future Work

### Current Limitations
- Gating network requires training to specialize
- Memory overhead from multiple expert copies
- Top-k routing may not utilize all experts equally

### Future Improvements
- [ ] Expert pruning based on usage statistics
- [ ] Hierarchical expert structure
- [ ] Domain-aware expert initialization
- [ ] Load balancing loss for expert utilization
- [ ] Sparse expert activation patterns

## Citation

If you use this implementation, please cite both SAM2 and this work:

```bibtex
@article{sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  year={2024}
}
```

## License

This implementation follows the same license as SAM2. See [LICENSE](LICENSE) for details.

## Acknowledgments

- **SAM2 Team** at Meta AI for the base model
- **MoPE/LoRA** literature for the expert mixture framework
- Inspired by research on mitigating catastrophic forgetting in continual learning
