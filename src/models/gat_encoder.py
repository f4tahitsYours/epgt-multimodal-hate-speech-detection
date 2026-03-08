"""
gat_encoder.py — Component 2: Emoji Graph Encoder (2-layer GAT).

Arsitektur:
  Layer 1: GATConv(203 → 64, heads=4, concat=True) → 256-dim
  Layer 2: GATConv(256 → 64, heads=4, concat=True) → 256-dim
  Pooling : global_mean_pool → graph_embedding ∈ ℝ²⁵⁶

Edge weight modulation:
  Attention score dimodulasi dengan edge_weight dari blueprint:
  e_ij = LeakyReLU(aᵀ[W·h_i || W·h_j]) · w(v_i, v_j)

Ablation support:
  zero_output=True → return zero tensor (ABL-1: no_graph)

Blueprint Section 3.2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Optional


class EmojiGraphEncoder(nn.Module):
    """
    2-layer Graph Attention Network untuk encoding emoji graph.

    Input  : PyG Batch object (dari EmojiGraphBuilder.build_batch)
    Output : graph_embedding ∈ ℝ²⁵⁶ per sampel dalam batch

    Args:
        node_feat_dim : dimensi node feature (default 203)
        hidden_dim    : output dim per head × heads (default 256)
        n_heads       : jumlah attention heads (default 4)
        n_layers      : jumlah GAT layers (default 2)
        dropout_rate  : dropout rate (default 0.3)
        zero_output   : jika True, return zeros (ABL-1: no_graph)
    """

    OUTPUT_DIM = 256

    def __init__(
        self,
        node_feat_dim : int   = 203,
        hidden_dim    : int   = 256,
        n_heads       : int   = 4,
        n_layers      : int   = 2,
        dropout_rate  : float = 0.3,
        zero_output   : bool  = False,
    ):
        super().__init__()

        self.zero_output  = zero_output
        self.output_dim   = hidden_dim
        self.dropout_rate = dropout_rate
        self.n_heads      = n_heads

        # head_dim * n_heads = hidden_dim
        assert hidden_dim % n_heads == 0,             f"hidden_dim ({hidden_dim}) harus habis dibagi n_heads ({n_heads})"
        head_dim = hidden_dim // n_heads

        # Layer 1: node_feat_dim → head_dim * n_heads = hidden_dim
        self.gat1 = GATConv(
            in_channels  = node_feat_dim,
            out_channels = head_dim,
            heads        = n_heads,
            concat       = True,
            dropout      = dropout_rate,
            edge_dim     = 1,       # edge_weight sebagai edge feature
        )

        # Layer 2: hidden_dim → head_dim * n_heads = hidden_dim
        self.gat2 = GATConv(
            in_channels  = hidden_dim,
            out_channels = head_dim,
            heads        = n_heads,
            concat       = True,
            dropout      = dropout_rate,
            edge_dim     = 1,
        )

        self.norm1   = nn.LayerNorm(hidden_dim)
        self.norm2   = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        x          : torch.Tensor,
        edge_index : torch.Tensor,
        edge_weight: torch.Tensor,
        batch      : torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x           : node features  (N_total, 203)
            edge_index  : edge indices   (2, E_total)
            edge_weight : edge weights   (E_total,)
            batch       : batch vector   (N_total,) — node ke graph mapping

        Returns:
            graph_embedding : (B, 256)
        """
        B = batch.max().item() + 1 if batch.numel() > 0 else 1

        # Ablation ABL-1: return zero embedding
        if self.zero_output:
            return torch.zeros(
                B, self.output_dim,
                device=x.device, dtype=x.dtype
            )

        # Edge weight sebagai edge feature (reshape ke (E, 1))
        edge_attr = edge_weight.unsqueeze(-1)

        # GAT Layer 1
        h = self.gat1(x, edge_index, edge_attr=edge_attr)  # (N, 256)
        h = self.norm1(h)
        h = F.elu(h)
        h = self.dropout(h)

        # GAT Layer 2
        h = self.gat2(h, edge_index, edge_attr=edge_attr)  # (N, 256)
        h = self.norm2(h)
        h = F.elu(h)

        # Global Mean Pooling → graph-level embedding
        graph_embedding = global_mean_pool(h, batch)       # (B, 256)
        return graph_embedding

    def get_output_dim(self) -> int:
        return self.output_dim
