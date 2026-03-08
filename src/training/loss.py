"""
loss.py — MTL Loss untuk EPGT.

L_total = λ_A · L_intensity + λ_B · L_sarcasm + λ_C · L_role
λ_A=0.40, λ_B=0.35, λ_C=0.25 (blueprint Section 4.1)

Semua sub-loss menggunakan CrossEntropyLoss dengan label_smoothing=0.1.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class EPGTLoss(nn.Module):
    """
    Weighted MTL loss untuk 3 task EPGT.

    Args:
        lambda_intensity : bobot L_intensity (default 0.40)
        lambda_sarcasm   : bobot L_sarcasm   (default 0.35)
        lambda_role      : bobot L_role      (default 0.25)
        label_smoothing  : label smoothing (default 0.10)
        ignore_index     : label yang diabaikan (default -1)
    """

    def __init__(
        self,
        lambda_intensity: float = 0.40,
        lambda_sarcasm  : float = 0.35,
        lambda_role     : float = 0.25,
        label_smoothing : float = 0.10,
        ignore_index    : int   = -1,
    ):
        super().__init__()

        total = lambda_intensity + lambda_sarcasm + lambda_role
        assert abs(total - 1.0) < 1e-6, f"Lambda harus berjumlah 1.0, got {total}"

        self.lambda_intensity = lambda_intensity
        self.lambda_sarcasm   = lambda_sarcasm
        self.lambda_role      = lambda_role

        self.criterion = nn.CrossEntropyLoss(
            label_smoothing = label_smoothing,
            ignore_index    = ignore_index,
        )

    def forward(
        self,
        logits_intensity : torch.Tensor,
        logits_sarcasm   : torch.Tensor,
        logits_role      : torch.Tensor,
        labels_intensity : torch.Tensor,
        labels_sarcasm   : torch.Tensor,
        labels_role      : torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Hitung total MTL loss.

        Returns:
            total_loss : scalar tensor untuk backward()
            loss_dict  : dict dengan sub-loss values (untuk logging)
        """
        L_intensity = self.criterion(logits_intensity, labels_intensity)
        L_sarcasm   = self.criterion(logits_sarcasm,   labels_sarcasm)
        L_role      = self.criterion(logits_role,      labels_role)

        total_loss = (
            self.lambda_intensity * L_intensity +
            self.lambda_sarcasm   * L_sarcasm   +
            self.lambda_role      * L_role
        )

        loss_dict = {
            "loss_total"    : total_loss.item(),
            "loss_intensity": L_intensity.item(),
            "loss_sarcasm"  : L_sarcasm.item(),
            "loss_role"     : L_role.item(),
        }

        return total_loss, loss_dict
