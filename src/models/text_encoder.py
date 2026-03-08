"""
text_encoder.py — Component 1: Text Semantic Encoder.

Model    : IndoBERT (indobenchmark/indobert-base-p1)
Fallback : bert-base-multilingual-cased
Pooling  : CLS token (index 0)
Output   : text_embedding ∈ ℝ⁷⁶⁸

Blueprint Section 3.1.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict, Optional


class TextSemanticEncoder(nn.Module):
    """
    IndoBERT-based text encoder.
    CLS token pooling → text_embedding ∈ ℝ⁷⁶⁸.

    Args:
        model_name   : HuggingFace model identifier
        dropout_rate : Dropout setelah CLS pooling
        freeze_layers: Jumlah layer BERT yang di-freeze (0 = tidak ada)
    """

    PRIMARY_MODEL  = "indobenchmark/indobert-base-p1"
    FALLBACK_MODEL = "bert-base-multilingual-cased"
    OUTPUT_DIM     = 768

    def __init__(
        self,
        model_name   : Optional[str] = None,
        dropout_rate : float = 0.3,
        freeze_layers: int   = 0,
    ):
        super().__init__()

        self.model_name  = model_name or self.PRIMARY_MODEL
        self.output_dim  = self.OUTPUT_DIM
        self.bert        = self._load_bert()
        self.dropout     = nn.Dropout(dropout_rate)

        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)

    def _load_bert(self) -> AutoModel:
        for name in [self.model_name, self.FALLBACK_MODEL]:
            try:
                model = AutoModel.from_pretrained(name)
                print(f"  [TextEncoder] Loaded: {name}")
                self.model_name = name
                return model
            except Exception as e:
                print(f"  [TextEncoder] Failed {name}: {e}")
        raise RuntimeError("Semua BERT model gagal dimuat.")

    def _freeze_layers(self, n_layers: int) -> None:
        """Freeze n_layers pertama dari BERT encoder."""
        freeze_list = [
            self.bert.embeddings,
            *self.bert.encoder.layer[:n_layers],
        ]
        for module in freeze_list:
            for param in module.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids     : torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids      : (B, seq_len)
            attention_mask : (B, seq_len)
            token_type_ids : (B, seq_len) optional

        Returns:
            text_embedding : (B, 768) — CLS token representation
        """
        kwargs = dict(
            input_ids      = input_ids,
            attention_mask = attention_mask,
        )
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs        = self.bert(**kwargs)
        # CLS token = index 0 dari last_hidden_state
        cls_embedding  = outputs.last_hidden_state[:, 0, :]  # (B, 768)
        text_embedding = self.dropout(cls_embedding)
        return text_embedding

    def get_output_dim(self) -> int:
        return self.output_dim
