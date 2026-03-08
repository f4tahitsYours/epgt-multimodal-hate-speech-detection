"""
emoji_graph.py — Emoji Pragmatic Graph Constructor untuk EPGT.

Membangun G=(V,E,W) sesuai blueprint Section 3.2:
  Node feature: x_i = Concat(e_i∈ℝ²⁰⁰, p_i∈ℝ¹, s_i∈ℝ¹, r_i∈ℝ¹) ∈ ℝ²⁰³
  Edge types  : Sequential, Repetition, Semantic
  Output      : torch_geometric.data.Data object
"""

import re
import logging
import numpy as np
import torch
import emoji
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

# ── KONSTANTA ─────────────────────────────────────────────────────────────────
EMOJI_EMBEDDING_DIM   = 200     # e_i dimensi
NODE_FEATURE_DIM      = 203     # total: 200 + 1 + 1 + 1
REPETITION_WINDOW_K   = 3       # window untuk repetition edge
SEMANTIC_THRESHOLD    = 0.7     # threshold cosine similarity semantic edge

# Edge type encoding
EDGE_TYPE_SEQUENTIAL  = 0
EDGE_TYPE_REPETITION  = 1
EDGE_TYPE_SEMANTIC    = 2


# ── EMOJI SENTIMENT LEXICON ───────────────────────────────────────────────────

class EmojiSentimentLexicon:
    """
    Sentiment polarity score s_i untuk tiap emoji.
    Sumber: EmojiNet-inspired manual mapping.
    Score range: [-1.0, 1.0]
    Default: 0.0 (netral) untuk emoji tidak dikenal.
    """

    SENTIMENT_MAP = {
        # Positif kuat (+0.8 s/d +1.0)
        "🔥": 0.9,   # 🔥
        "😍": 1.0,   # 😍
        "🤩": 1.0,   # 🤩
        "🥰": 0.9,   # 🥰
        "💯": 0.9,   # 💯
        "❤":    0.8,    # ❤
        "💖": 0.8,   # 💖
        "👏": 0.8,   # 👏
        "👍": 0.7,   # 👍
        "🙏": 0.7,   # 🙏
        "🌟": 0.8,   # 🌟
        "🎊": 0.7,   # 🎊
        "🎉": 0.7,   # 🎉
        # Positif sedang (+0.3 s/d +0.7)
        "😂": 0.7,   # 😂
        "🤣": 0.6,   # 🤣
        "😄": 0.6,   # 😄
        "😀": 0.5,   # 😀
        "😉": 0.5,   # 😉
        "😘": 0.6,   # 😘
        "🌞": 0.6,   # 🌞
        "🌿": 0.4,   # 🌿
        "🪴": 0.4,   # 🪴
        # Netral (sekitar 0.0)
        "🤔": 0.0,   # 🤔
        "👀": 0.0,   # 👀
        "💬": 0.0,   # 💬
        "📝": 0.0,   # 📝
        "👋": 0.1,   # 👋
        # Negatif sedang (-0.3 s/d -0.7)
        "😔": -0.4,  # 😔
        "😕": -0.3,  # 😕
        "😬": -0.5,  # 😬
        "😑": -0.4,  # 😑
        "🙄": -0.4,  # 🙄
        "😤": -0.5,  # 😤
        "😡": -0.7,  # 😡
        "😠": -0.6,  # 😠
        # Negatif kuat / ironi (-0.7 s/d -1.0)
        "😭": -0.7,  # 😭
        "💀": -0.8,  # 💀
        "🗿": -0.6,  # 🗿 (ironi/sarkasme)
        "🙂": -0.5,  # 🙂 (ironi implisit)
        "🤡": -0.6,  # 🤡
        "🫠": -0.5,  # 🫠
        "💔": -0.9,  # 💔
        "😩": -0.8,  # 😩
    }

    def get_score(self, emoji_char: str) -> float:
        return self.SENTIMENT_MAP.get(emoji_char, 0.0)

    def get_scores(self, emojis: List[str]) -> List[float]:
        return [self.get_score(e) for e in emojis]


# ── EMOJI EMBEDDING LOADER ────────────────────────────────────────────────────

class EmojiEmbeddingLoader:
    """
    Loader untuk emoji identity embedding e_i ∈ ℝ²⁰⁰.
    Prioritas: (1) pre-trained emoji2vec → (2) random init
    Random init: seeded hash dari unicode codepoint untuk konsistensi.
    """

    def __init__(self, embedding_dim: int = 200):
        self.embedding_dim = embedding_dim
        self._cache: Dict[str, np.ndarray] = {}

    def get_embedding(self, emoji_char: str) -> np.ndarray:
        """
        Ambil embedding untuk satu emoji.
        Menggunakan seeded random berdasarkan unicode codepoint
        untuk menghasilkan embedding yang konsisten dan deterministik.
        """
        if emoji_char in self._cache:
            return self._cache[emoji_char]

        # Seed dari hash unicode codepoint → deterministik & konsisten
        seed = sum(ord(c) for c in emoji_char) % (2**31)
        rng  = np.random.RandomState(seed)
        emb  = rng.randn(self.embedding_dim).astype(np.float32)

        # L2 normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        self._cache[emoji_char] = emb
        return emb

    def get_embeddings(self, emojis: List[str]) -> np.ndarray:
        """
        Returns: array of shape (n_emoji, embedding_dim)
        """
        if not emojis:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        return np.stack([self.get_embedding(e) for e in emojis])


# ── GRAPH BUILDER ─────────────────────────────────────────────────────────────

class EmojiGraphBuilder:
    """
    Membangun Emoji Pragmatic Graph G=(V,E,W) sesuai blueprint.

    Node feature x_i = Concat(e_i, p_i, s_i, r_i) ∈ ℝ²⁰³:
      e_i ∈ ℝ²⁰⁰ : identity embedding
      p_i ∈ ℝ¹   : normalized position = token_pos / seq_len
      s_i ∈ ℝ¹   : sentiment score ∈ [-1, 1]
      r_i ∈ ℝ¹   : repetition flag ∈ {0, 1}

    Edge types:
      0 = Sequential  : v_i → v_{i+1}
      1 = Repetition  : v_i → v_j (same emoji, |i-j| ≤ K=3)
      2 = Semantic    : v_i → v_j (cosine_sim ≥ θ=0.7)
    """

    def __init__(
        self,
        embedding_dim    : int   = EMOJI_EMBEDDING_DIM,
        repetition_window: int   = REPETITION_WINDOW_K,
        semantic_threshold: float = SEMANTIC_THRESHOLD,
    ):
        self.embedding_dim     = embedding_dim
        self.K                 = repetition_window
        self.theta             = semantic_threshold
        self.emb_loader        = EmojiEmbeddingLoader(embedding_dim)
        self.sentiment_lexicon = EmojiSentimentLexicon()

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def build_node_features(
        self,
        emojis           : List[str],
        positions        : List[float],
        repetition_flags : List[int],
        zero_position    : bool = False,
    ) -> torch.Tensor:
        """
        Bangun node feature matrix X ∈ ℝ^{n×203}.

        Args:
            zero_position: Jika True, set p_i=0 (untuk ablation ABL-4)

        Returns:
            Tensor shape (n_emoji, 203)
        """
        n = len(emojis)
        if n == 0:
            return torch.zeros((1, NODE_FEATURE_DIM), dtype=torch.float)

        # e_i: identity embeddings (n, 200)
        emb_matrix = self.emb_loader.get_embeddings(emojis)

        # p_i: normalized position (n, 1)
        pos_array = np.zeros((n, 1), dtype=np.float32)
        if not zero_position and positions:
            for i, p in enumerate(positions[:n]):
                pos_array[i, 0] = float(p)

        # s_i: sentiment score (n, 1)
        scores    = self.sentiment_lexicon.get_scores(emojis)
        sent_array= np.array(scores[:n], dtype=np.float32).reshape(-1, 1)

        # r_i: repetition flag (n, 1)
        rep_flags = repetition_flags[:n] if repetition_flags else [0] * n
        rep_array = np.array(rep_flags, dtype=np.float32).reshape(-1, 1)

        # Concatenate: (n, 200+1+1+1) = (n, 203)
        x = np.concatenate([emb_matrix, pos_array, sent_array, rep_array], axis=1)
        return torch.tensor(x, dtype=torch.float)

    def build_edges(
        self,
        emojis           : List[str],
        positions        : List[float],
        node_features    : torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Bangun edge_index, edge_weight, edge_type.

        Returns:
            edge_index  : LongTensor (2, E)
            edge_weight : FloatTensor (E,)
            edge_type   : LongTensor (E,)
        """
        n          = len(emojis)
        src_list   = []
        dst_list   = []
        weight_list= []
        type_list  = []

        # EDGE TYPE 0: Sequential v_i → v_{i+1}
        # w_seq = 1 / (1 + |pos_i - pos_j|)
        for i in range(n - 1):
            j   = i + 1
            p_i = positions[i] if i < len(positions) else 0.0
            p_j = positions[j] if j < len(positions) else 0.0
            w   = 1.0 / (1.0 + abs(p_i - p_j))
            src_list.append(i);    dst_list.append(j)
            weight_list.append(w); type_list.append(EDGE_TYPE_SEQUENTIAL)

        # EDGE TYPE 1: Repetition v_i → v_j (same emoji, |i-j| ≤ K)
        # w_rep = count(emoji_i in window) / K
        for i in range(n):
            window_start = max(0, i - self.K)
            window_end   = min(n, i + self.K + 1)
            same_in_window = sum(
                1 for j in range(window_start, window_end)
                if j != i and emojis[j] == emojis[i]
            )
            if same_in_window > 0:
                w = same_in_window / self.K
                for j in range(window_start, window_end):
                    if j != i and emojis[j] == emojis[i]:
                        src_list.append(i);    dst_list.append(j)
                        weight_list.append(w); type_list.append(EDGE_TYPE_REPETITION)

        # EDGE TYPE 2: Semantic v_i → v_j (cosine_sim(s_i, s_j) ≥ θ)
        # w_sem = cosine_sim(s_i, s_j)
        # Gunakan embedding vector untuk menghitung similarity
        nf = node_features.numpy()
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                e_i = nf[i, :self.embedding_dim]
                e_j = nf[j, :self.embedding_dim]
                sim = self._cosine_similarity(e_i, e_j)
                if sim >= self.theta:
                    src_list.append(i);      dst_list.append(j)
                    weight_list.append(sim); type_list.append(EDGE_TYPE_SEMANTIC)

        if not src_list:
            # Tidak ada edge → kembalikan tensor kosong
            return (
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0,),   dtype=torch.float),
                torch.zeros((0,),   dtype=torch.long),
            )

        edge_index  = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_weight = torch.tensor(weight_list,          dtype=torch.float)
        edge_type   = torch.tensor(type_list,            dtype=torch.long)

        return edge_index, edge_weight, edge_type

    def build_graph(
        self,
        emojis           : List[str],
        positions        : List[float],
        repetition_flags : List[int],
        label_intensity  : int = -1,
        label_sarcasm    : int = -1,
        label_emoji_role : int = -1,
        sample_id        : str = "",
        ablation_mode    : Optional[str] = None,
    ) -> Data:
        """
        Bangun satu PyG Data object.

        Args:
            ablation_mode: None | "no_position" (ABL-4: zero p_i)

        Returns:
            torch_geometric.data.Data dengan:
              x           : node features (n, 203)
              edge_index  : (2, E)
              edge_weight : (E,)
              edge_type   : (E,)
              num_nodes   : n
              y_intensity : label tensor
              y_sarcasm   : label tensor
              y_role      : label tensor
        """
        # Handle edge case: tidak ada emoji
        if not emojis:
            emojis           = ["😐"]  # 😐 placeholder
            positions        = [0.5]
            repetition_flags = [0]

        # Handle single emoji → self-loop
        add_self_loop = len(emojis) == 1

        zero_pos   = ablation_mode == "no_position"
        x          = self.build_node_features(emojis, positions, repetition_flags, zero_pos)
        edge_index, edge_weight, edge_type = self.build_edges(emojis, positions, x)

        # Self-loop untuk single emoji
        if add_self_loop or edge_index.shape[1] == 0:
            n          = x.shape[0]
            self_loops = torch.arange(n).unsqueeze(0).repeat(2, 1)
            self_wts   = torch.ones(n, dtype=torch.float)
            self_types = torch.zeros(n, dtype=torch.long)
            if edge_index.shape[1] > 0:
                edge_index  = torch.cat([edge_index,  self_loops], dim=1)
                edge_weight = torch.cat([edge_weight, self_wts])
                edge_type   = torch.cat([edge_type,   self_types])
            else:
                edge_index  = self_loops
                edge_weight = self_wts
                edge_type   = self_types

        data = Data(
            x            = x,
            edge_index   = edge_index,
            edge_weight  = edge_weight,
            edge_type    = edge_type,
            num_nodes    = x.shape[0],
        )

        # Label
        data.y_intensity = torch.tensor([label_intensity], dtype=torch.long)
        data.y_sarcasm   = torch.tensor([label_sarcasm],   dtype=torch.long)
        data.y_role      = torch.tensor([label_emoji_role],dtype=torch.long)
        data.sample_id   = sample_id

        return data

    def build_batch(
        self,
        df            : "pd.DataFrame",
        ablation_mode : Optional[str] = None,
        show_progress : bool = True,
    ) -> List[Data]:
        """
        Bangun list of Data objects dari DataFrame.
        Kolom yang dibutuhkan:
          emoji_list, emoji_positions, repetition_flags,
          label_intensity, label_sarcasm, label_emoji_role, id
        """
        from tqdm.auto import tqdm
        import ast

        graphs = []
        iterator = tqdm(df.iterrows(), total=len(df), desc="Building graphs",
                        disable=not show_progress)

        for _, row in iterator:
            # Parse list columns (mungkin tersimpan sebagai string)
            emoji_list = row.get("emoji_list", [])
            if isinstance(emoji_list, str):
                try:
                    emoji_list = ast.literal_eval(emoji_list)
                except Exception:
                    emoji_list = emoji_list.split(",") if emoji_list else []

            positions = row.get("emoji_positions", [])
            if isinstance(positions, str):
                try:
                    positions = ast.literal_eval(positions)
                except Exception:
                    positions = []

            rep_flags = row.get("repetition_flags", [])
            if isinstance(rep_flags, str):
                try:
                    rep_flags = ast.literal_eval(rep_flags)
                except Exception:
                    rep_flags = []

            graph = self.build_graph(
                emojis           = emoji_list,
                positions        = [float(p) for p in positions],
                repetition_flags = [int(r) for r in rep_flags],
                label_intensity  = int(row.get("label_intensity",  -1)),
                label_sarcasm    = int(row.get("label_sarcasm",    -1)),
                label_emoji_role = int(row.get("label_emoji_role", -1)),
                sample_id        = str(row.get("id", "")),
                ablation_mode    = ablation_mode,
            )
            graphs.append(graph)

        return graphs
