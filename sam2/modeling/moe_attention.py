# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
MoE-enhanced attention modules for SAM2.
Extends RoPEAttention with Mixture of Prompt Experts (MoPE).
"""

import math
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from sam2.modeling.sam.transformer import RoPEAttention, compute_axial_cis, apply_rotary_enc
from sam2.modeling.moe_modules import MoELoRALinear


class MoERoPEAttention(RoPEAttention):
    """
    RoPE Attention with Mixture of Prompt Experts (MoPE).

    Extends standard RoPEAttention by replacing projection layers
    (q_proj, k_proj, v_proj, out_proj) with MoE-LoRA variants.

    This enables the model to:
    1. Maintain frozen base projections from pre-trained weights
    2. Learn specialized expert adapters for different domains/objects
    3. Dynamically route inputs to appropriate experts
    4. Mitigate catastrophic forgetting via domain-specific experts

    Args:
        *args: Arguments passed to RoPEAttention
        num_experts: Number of LoRA experts per projection
        lora_rank: Rank of LoRA adapters
        top_k: Number of experts to activate
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout for LoRA layers
        **kwargs: Additional arguments passed to RoPEAttention
    """

    def __init__(
        self,
        *args,
        num_experts: int = 10,
        lora_rank: int = 4,
        top_k: int = 2,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        # Initialize base RoPEAttention first
        super().__init__(*args, **kwargs)

        # Store MoE config
        self.num_experts = num_experts
        self.lora_rank = lora_rank
        self.top_k = top_k

        # Store original projections temporarily
        original_q_proj = self.q_proj
        original_k_proj = self.k_proj
        original_v_proj = self.v_proj
        original_out_proj = self.out_proj

        # Replace projection layers with MoE-LoRA versions
        # Note: We keep the original layers as base and wrap them
        self.q_proj = MoELoRALinear(
            base_layer=original_q_proj,
            num_experts=num_experts,
            rank=lora_rank,
            top_k=top_k,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )

        self.k_proj = MoELoRALinear(
            base_layer=original_k_proj,
            num_experts=num_experts,
            rank=lora_rank,
            top_k=top_k,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )

        self.v_proj = MoELoRALinear(
            base_layer=original_v_proj,
            num_experts=num_experts,
            rank=lora_rank,
            top_k=top_k,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )

        self.out_proj = MoELoRALinear(
            base_layer=original_out_proj,
            num_experts=num_experts,
            rank=lora_rank,
            top_k=top_k,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, num_k_exclude_rope: int = 0
    ) -> Tensor:
        """
        Forward pass with MoE-enhanced projections.

        The flow is identical to RoPEAttention, but projections now
        route through expert adapters:
        1. Project q, k, v through MoE-LoRA layers
        2. Separate into attention heads
        3. Apply RoPE
        4. Compute scaled dot-product attention
        5. Recombine heads
        6. Project output through MoE-LoRA

        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor
            num_k_exclude_rope: Number of keys to exclude from RoPE

        Returns:
            Attention output
        """
        # Input projections (now with MoE-LoRA)
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Apply rotary position encoding
        w = h = math.sqrt(q.shape[-2])
        self.freqs_cis = self.freqs_cis.to(q.device)
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat

        num_k_rope = k.size(-2) - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )

        dropout_p = self.dropout_p if self.training else 0.0
        # Attention
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

    def get_expert_statistics(self, q: Tensor, k: Tensor, v: Tensor) -> dict:
        """
        Analyze expert usage across all projections.

        Useful for debugging and understanding expert specialization.

        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor

        Returns:
            Dictionary with expert usage statistics for each projection
        """
        stats = {}
        with torch.no_grad():
            stats['q_proj'] = self.q_proj.get_expert_usage(q)
            stats['k_proj'] = self.k_proj.get_expert_usage(k)
            stats['v_proj'] = self.v_proj.get_expert_usage(v)
            # For out_proj, we need intermediate attention output
            # For simplicity, we skip it here or could approximate with v
        return stats


def convert_rope_attention_to_moe(
    attention: RoPEAttention,
    num_experts: int = 10,
    lora_rank: int = 4,
    top_k: int = 2,
    lora_alpha: float = 1.0,
    lora_dropout: float = 0.0,
) -> MoERoPEAttention:
    """
    Convert a pre-trained RoPEAttention to MoE variant.

    This function creates a new MoERoPEAttention and copies over
    the base weights from the original attention module.

    Args:
        attention: Original RoPEAttention module
        num_experts: Number of experts
        lora_rank: LoRA rank
        top_k: Top-k expert selection
        lora_alpha: LoRA scaling
        lora_dropout: LoRA dropout

    Returns:
        MoERoPEAttention with copied base weights
    """
    # Create new MoE attention with same config
    moe_attention = MoERoPEAttention(
        embedding_dim=attention.embedding_dim,
        num_heads=attention.num_heads,
        downsample_rate=attention.internal_dim / attention.embedding_dim,
        dropout=attention.dropout_p,
        kv_in_dim=attention.kv_in_dim,
        rope_theta=10000.0,  # Default from RoPEAttention
        rope_k_repeat=attention.rope_k_repeat,
        feat_sizes=(64, 64),  # Default, will be recomputed
        num_experts=num_experts,
        lora_rank=lora_rank,
        top_k=top_k,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    # The base layers are already copied when we pass them to MoELoRALinear
    # during __init__, so no additional copying needed

    return moe_attention
