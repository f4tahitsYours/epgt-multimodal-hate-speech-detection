"""
fusion_layer.py — Component 3: Pragmatic Fusion Layer (Cross-Attention).

Mekanisme:
  Q = W_Q · text_emb        (ℝ⁷⁶⁸ → ℝ⁷⁶⁸)
  K = W_K · graph_emb       (ℝ²⁵⁶ → ℝ⁷⁶⁸)
  V = W_V · graph_emb       (ℝ²⁵⁶ → ℝ⁷⁶⁸)
  attn_out = softmax(QKᵀ/√768) · V
  combined = LayerNorm(text_emb + attn_out)  ← residual

Output: combined_repr ∈ ℝ⁷⁶⁸

Ablation support:
  bypass=True → concat(text, proj(graph)) → linear → 768 (ABL-2: no_fusion)

Blueprint Section 3.3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PragmaticFusionLayer(nn.Module):
    """
    Cross-attention fusion antara text embedding dan graph embedding.

    Args:
        text_dim     : dimensi text embedding (default 768)
        graph_dim    : dimensi graph embedding (default 256)
        output_dim   : dimensi output (default 768)
        dropout_rate : dropout rate (default 0.3)
        bypass       : jika True, skip cross-attention (ABL-2: no_fusion)
    """

    def __init__(
        self,
        text_dim    : int   = 768,
        graph_dim   : int   = 256,
        output_dim  : int   = 768,
        dropout_rate: float = 0.3,
        bypass      : bool  = False,
    ):
        super().__init__()

        self.text_dim   = text_dim
        self.graph_dim  = graph_dim
        self.output_dim = output_dim
        self.bypass     = bypass
        self.scale      = math.sqrt(text_dim)

        # Projection: text → Q
        self.W_Q = nn.Linear(text_dim,  text_dim,  bias=False)
        # Projection: graph → K, V
        self.W_K = nn.Linear(graph_dim, text_dim,  bias=False)
        self.W_V = nn.Linear(graph_dim, text_dim,  bias=False)

        # Output projection + normalization
        self.out_proj = nn.Linear(text_dim, output_dim)
        self.norm     = nn.LayerNorm(output_dim)
        self.dropout  = nn.Dropout(dropout_rate)

        # Bypass path (ABL-2): concat → linear → output_dim
        if bypass:
            self.bypass_proj = nn.Linear(text_dim + graph_dim, output_dim)
            self.bypass_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        text_embedding : torch.Tensor,
        graph_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            text_embedding  : (B, 768)
            graph_embedding : (B, 256)

        Returns:
            combined_repr   : (B, 768)
        """
        # Ablation ABL-2: bypass cross-attention
        if self.bypass:
            combined = torch.cat([text_embedding, graph_embedding], dim=-1)
            return self.bypass_norm(F.relu(self.bypass_proj(combined)))

        # Cross-attention:
        # text_embedding sebagai Query, graph_embedding sebagai Key dan Value

        # Q: (B, 768), K: (B, 768), V: (B, 768)
        Q = self.W_Q(text_embedding)    # (B, 768)
        K = self.W_K(graph_embedding)   # (B, 768)
        V = self.W_V(graph_embedding)   # (B, 768)

        # Scaled dot-product attention
        # Q, K, V: unsqueeze ke (B, 1, 768) untuk bmm
        Q_ = Q.unsqueeze(1)  # (B, 1, 768)
        K_ = K.unsqueeze(1)  # (B, 1, 768)
        V_ = V.unsqueeze(1)  # (B, 1, 768)

        # Attention score: (B, 1, 1)
        attn_score = torch.bmm(Q_, K_.transpose(1, 2)) / self.scale
        attn_weight= F.softmax(attn_score, dim=-1)     # (B, 1, 1)
        attn_out   = torch.bmm(attn_weight, V_)        # (B, 1, 768)
        attn_out   = attn_out.squeeze(1)               # (B, 768)

        # Residual connection + output projection
        combined   = text_embedding + self.dropout(attn_out)
        combined   = self.norm(combined)
        combined   = self.out_proj(combined)           # (B, 768)

        return combined

    def get_output_dim(self) -> int:
        return self.output_dim
