# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Mixture of Prompt Experts (MoPE) modules for SAM2.
Implements LoRA-based expert adapters with learned routing.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAAdapter(nn.Module):
    """
    Low-Rank Adaptation (LoRA) adapter.

    Implements the reparameterization: W' = W + BA, where
    - W is the original weight (frozen)
    - B is down-projection (d → r)
    - A is up-projection (r → d)
    - r is the rank (r << d)

    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: LoRA rank (bottleneck dimension)
        alpha: LoRA scaling factor
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        # LoRA down-projection (in_features → rank)
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        # LoRA up-projection (rank → out_features)
        self.lora_up = nn.Linear(rank, out_features, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

        # Scaling factor
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            LoRA adaptation of shape (..., out_features)
        """
        # x → down → dropout → up → scale
        return self.lora_up(self.dropout(self.lora_down(x))) * self.scaling


class TopKGatingNetwork(nn.Module):
    """
    Gating network with top-k expert selection.

    Computes expert weights using top-k softmax routing:
    1. Project input to expert logits
    2. Select top-k experts
    3. Apply softmax over top-k
    4. Zero out non-top-k experts

    Args:
        input_dim: Input feature dimension
        num_experts: Number of experts
        top_k: Number of experts to activate
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)

        # Gating network: input → expert logits
        self.gate = nn.Linear(input_dim, num_experts)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Expert weights of shape (batch_size, seq_len, num_experts)
            Only top-k experts have non-zero weights (softmax normalized)
        """
        # Compute gating logits
        # (B, N, input_dim) → (B, N, num_experts)
        logits = self.gate(self.dropout(x))

        # Top-k selection
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)

        # Apply softmax over top-k
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        # Scatter weights back to full expert dimension
        weights = torch.zeros_like(logits)
        weights.scatter_(-1, top_k_indices, top_k_weights)

        return weights


class MoELoRALinear(nn.Module):
    """
    Mixture of Experts with LoRA adapters for linear projections.

    Architecture:
        y = W_base(x) + ∑_{i=1}^{num_experts} w_i * LoRA_i(x)

    where:
    - W_base is the frozen base projection
    - LoRA_i are expert adapters
    - w_i are learned routing weights (top-k gated)

    Args:
        base_layer: Pre-trained linear layer (will be frozen)
        num_experts: Number of LoRA experts
        rank: LoRA rank
        top_k: Number of experts to activate
        alpha: LoRA scaling factor
        dropout: Dropout probability
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        num_experts: int = 10,
        rank: int = 4,
        top_k: int = 2,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Freeze base layer
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.num_experts = num_experts
        self.rank = rank
        self.top_k = top_k

        # Create LoRA experts
        self.experts = nn.ModuleList([
            LoRAAdapter(
                in_features=in_features,
                out_features=out_features,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            for _ in range(num_experts)
        ])

        # Gating network
        self.gate = TopKGatingNetwork(
            input_dim=in_features,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        # Base projection (frozen)
        base_output = self.base_layer(x)

        # Compute expert weights
        # (..., in_features) → (..., num_experts)
        expert_weights = self.gate(x)

        # Apply experts and weighted sum
        # For each expert, compute LoRA(x) and weight it
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_out = expert(x)  # (..., out_features)
            # Extract weight for this expert: (..., num_experts) → (..., 1)
            weight = expert_weights[..., i:i+1]
            expert_outputs.append(weight * expert_out)

        # Sum all expert contributions
        expert_output = torch.stack(expert_outputs, dim=0).sum(dim=0)

        # Combine base + expert outputs
        return base_output + expert_output

    def get_expert_usage(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get expert usage statistics for analysis.

        Args:
            x: Input tensor

        Returns:
            Expert weights averaged over batch and sequence
        """
        with torch.no_grad():
            expert_weights = self.gate(x)
            # Average over all dimensions except expert dimension
            return expert_weights.mean(dim=list(range(expert_weights.dim() - 1)))
