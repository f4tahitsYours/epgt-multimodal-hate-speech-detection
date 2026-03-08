"""
classification_head.py — Component 4: MTL Classification Head.

3 parallel classification heads:
  Head A: Emotion Intensity  → Linear → 3-class softmax
  Head B: Sarcasm Detection  → Linear → binary (logit)
  Head C: Emoji Role         → Linear → 4-class softmax

Input : combined_repr ∈ ℝ⁷⁶⁸
Output: Dict[str, Tensor] dengan logits per head

Blueprint Section 3.4.
"""

import torch
import torch.nn as nn
from typing import Dict


class MTLClassificationHead(nn.Module):
    """
    Multi-Task Learning classification head dengan 3 head paralel.

    Args:
        input_dim            : dimensi input (default 768)
        num_intensity_classes: jumlah kelas intensity (default 3)
        num_sarcasm_classes  : jumlah kelas sarcasm (default 2)
        num_role_classes     : jumlah kelas emoji role (default 4)
        dropout_rate         : dropout sebelum tiap head (default 0.3)
    """

    def __init__(
        self,
        input_dim            : int   = 768,
        num_intensity_classes: int   = 3,
        num_sarcasm_classes  : int   = 2,
        num_role_classes     : int   = 4,
        dropout_rate         : float = 0.3,
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout_rate)

        # Head A — Emotion Intensity (3-class)
        self.head_intensity = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim // 2, num_intensity_classes),
        )

        # Head B — Sarcasm Detection (binary, output 1 logit)
        self.head_sarcasm = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim // 2, num_sarcasm_classes),
        )

        # Head C — Emoji Pragmatic Role (4-class)
        self.head_role = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim // 2, num_role_classes),
        )

    def forward(
        self,
        combined_repr: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            combined_repr : (B, 768) — output dari PragmaticFusionLayer

        Returns:
            dict with:
              logits_intensity : (B, 3)
              logits_sarcasm   : (B, 2)
              logits_role      : (B, 4)
        """
        x = self.dropout(combined_repr)

        return {
            "logits_intensity": self.head_intensity(x),  # (B, 3)
            "logits_sarcasm"  : self.head_sarcasm(x),    # (B, 2)
            "logits_role"     : self.head_role(x),       # (B, 4)
        }
