"""
epgt.py — Full EPGT Model Assembly.

Mengintegrasikan 4 komponen arsitektur:
  Component 1: TextSemanticEncoder  (IndoBERT, output 768)
  Component 2: EmojiGraphEncoder    (2-layer GAT, output 256)
  Component 3: PragmaticFusionLayer (Cross-Attention, output 768)
  Component 4: MTLClassificationHead (3 parallel heads)

Ablation mode (single forward() untuk semua konfigurasi):
  None         → Full EPGT             (ABL-5 reference)
  "no_graph"   → skip GAT              (ABL-1)
  "no_fusion"  → skip cross-attention  (ABL-2)
  "no_emoji"   → skip semua emoji      (ABL-3, ≡ B1)
  "no_position"→ zero p_i di graph     (ABL-4)

Blueprint Section 3.1-3.4.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from typing import Dict, Optional

from models.text_encoder       import TextSemanticEncoder
from models.gat_encoder        import EmojiGraphEncoder
from models.fusion_layer       import PragmaticFusionLayer
from models.classification_head import MTLClassificationHead


VALID_ABLATION_MODES = {None, "no_graph", "no_fusion", "no_emoji", "no_position"}


class EPGTModel(nn.Module):
    """
    Emoji Pragmatic Graph Transformer — Full Model.

    Args:
        bert_model_name      : IndoBERT model name
        node_feat_dim        : dimensi node feature graph (203)
        graph_hidden_dim     : output dim GAT (256)
        gat_heads            : jumlah attention heads (4)
        gat_layers           : jumlah GAT layers (2)
        text_dim             : dimensi text embedding (768)
        num_intensity_classes: jumlah kelas intensity (3)
        num_sarcasm_classes  : jumlah kelas sarcasm (2)
        num_role_classes     : jumlah kelas emoji role (4)
        dropout_rate         : dropout global (0.3)
        freeze_bert_layers   : jumlah layer BERT yang di-freeze (0)
        ablation_mode        : None | "no_graph" | "no_fusion" |
                               "no_emoji" | "no_position"
    """

    def __init__(
        self,
        bert_model_name      : Optional[str] = None,
        node_feat_dim        : int   = 203,
        graph_hidden_dim     : int   = 256,
        gat_heads            : int   = 4,
        gat_layers           : int   = 2,
        text_dim             : int   = 768,
        num_intensity_classes: int   = 3,
        num_sarcasm_classes  : int   = 2,
        num_role_classes     : int   = 4,
        dropout_rate         : float = 0.3,
        freeze_bert_layers   : int   = 0,
        ablation_mode        : Optional[str] = None,
    ):
        super().__init__()

        assert ablation_mode in VALID_ABLATION_MODES,             f"ablation_mode harus salah satu dari: {VALID_ABLATION_MODES}"

        self.ablation_mode    = ablation_mode
        self.text_dim         = text_dim
        self.graph_hidden_dim = graph_hidden_dim

        # ── Component 1: Text Semantic Encoder ────────────────────
        # ABL-3 (no_emoji): tetap pakai text encoder
        self.text_encoder = TextSemanticEncoder(
            model_name    = bert_model_name,
            dropout_rate  = dropout_rate,
            freeze_layers = freeze_bert_layers,
        )

        # ── Component 2: Emoji Graph Encoder ──────────────────────
        # ABL-1 (no_graph): zero_output=True
        # ABL-3 (no_emoji): zero_output=True
        skip_graph = ablation_mode in ("no_graph", "no_emoji")
        self.graph_encoder = EmojiGraphEncoder(
            node_feat_dim = node_feat_dim,
            hidden_dim    = graph_hidden_dim,
            n_heads       = gat_heads,
            n_layers      = gat_layers,
            dropout_rate  = dropout_rate,
            zero_output   = skip_graph,
        )

        # ── Component 3: Pragmatic Fusion Layer ───────────────────
        # ABL-2 (no_fusion): bypass=True → concat instead of cross-attn
        # ABL-3 (no_emoji): bypass=True (no graph signal)
        use_bypass = ablation_mode in ("no_fusion", "no_emoji")
        self.fusion_layer = PragmaticFusionLayer(
            text_dim     = text_dim,
            graph_dim    = graph_hidden_dim,
            output_dim   = text_dim,
            dropout_rate = dropout_rate,
            bypass       = use_bypass,
        )

        # ── Component 4: MTL Classification Head ──────────────────
        self.classification_head = MTLClassificationHead(
            input_dim             = text_dim,
            num_intensity_classes = num_intensity_classes,
            num_sarcasm_classes   = num_sarcasm_classes,
            num_role_classes      = num_role_classes,
            dropout_rate          = dropout_rate,
        )

    def forward(
        self,
        input_ids     : torch.Tensor,
        attention_mask: torch.Tensor,
        graph_batch   : Batch,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass EPGT.

        Args:
            input_ids      : (B, 128)  — token IDs
            attention_mask : (B, 128)  — attention mask
            graph_batch    : PyG Batch — emoji graphs
            token_type_ids : (B, 128)  — optional

        Returns:
            dict with:
              logits_intensity : (B, 3)
              logits_sarcasm   : (B, 2)
              logits_role      : (B, 4)
              text_embedding   : (B, 768)  — untuk analisis
              graph_embedding  : (B, 256)  — untuk analisis
              combined_repr    : (B, 768)  — untuk analisis
        """
        # ── Component 1: Text Encoding ────────────────────────────
        text_embedding = self.text_encoder(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
        )  # (B, 768)

        # ── Component 2: Graph Encoding ───────────────────────────
        graph_embedding = self.graph_encoder(
            x           = graph_batch.x,
            edge_index  = graph_batch.edge_index,
            edge_weight = graph_batch.edge_weight,
            batch       = graph_batch.batch,
        )  # (B, 256)

        # ── Component 3: Pragmatic Fusion ─────────────────────────
        combined_repr = self.fusion_layer(
            text_embedding  = text_embedding,
            graph_embedding = graph_embedding,
        )  # (B, 768)

        # ── Component 4: MTL Classification ───────────────────────
        logits = self.classification_head(combined_repr)  # dict

        # Tambahkan intermediate representations untuk analisis
        logits["text_embedding"]  = text_embedding
        logits["graph_embedding"] = graph_embedding
        logits["combined_repr"]   = combined_repr

        return logits

    def count_parameters(self) -> Dict[str, int]:
        """Hitung jumlah parameter per komponen."""
        def count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "text_encoder"      : count(self.text_encoder),
            "graph_encoder"     : count(self.graph_encoder),
            "fusion_layer"      : count(self.fusion_layer),
            "classification_head": count(self.classification_head),
            "total"             : count(self),
        }

    def get_ablation_info(self) -> str:
        mode = self.ablation_mode or "None (Full EPGT)"
        descriptions = {
            None             : "Full EPGT — ABL-5 reference",
            "no_graph"       : "ABL-1: No Graph (zero graph embedding)",
            "no_fusion"      : "ABL-2: No Fusion (concat instead of cross-attn)",
            "no_emoji"       : "ABL-3: No Emoji (≡ IndoBERT text-only)",
            "no_position"    : "ABL-4: No Position (zero p_i in node features)",
        }
        return descriptions.get(self.ablation_mode, str(self.ablation_mode))
