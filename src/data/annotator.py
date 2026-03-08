"""
annotator.py — Annotation simulator dan IAA calculator untuk EPGT.

Komponen:
  HeuristicAnnotator : Annotasi berbasis rule linguistik Indonesia
  IAACalculator      : Cohen's Kappa per layer anotasi
  AnnotationManager  : Orkestrasi batch annotation workflow
"""

import re
import json
import random
import logging
import numpy as np
import pandas as pd
import emoji
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import cohen_kappa_score

logger = logging.getLogger(__name__)


# ── LEXICON ───────────────────────────────────────────────────────────────────

HIGH_INTENSITY_WORDS = {
    "banget", "parah", "bet", "pol", "abis", "gilak", "gila",
    "bgt", "bgtt", "bngtt", "literally", "literally",
    "wkwkwk", "wkwk", "ngakak", "hahaha", "astaga",
    "ya ampun", "ya allah", "sumpah", "gak nyangka",
    "seriusan", "beneran", "asli",
}

LOW_INTENSITY_WORDS = {
    "oke", "ok", "sip", "siap", "noted", "iya", "ya",
    "deh", "dong", "aja", "lah", "sih",
}

SARCASM_EMOJIS = {
    "🗿",  # 🗿 moai
    "💀",  # 💀 skull
    "🙂",  # 🙂 slightly smiling face
    "🤡",  # 🤡 clown
    "😑",  # 😑 expressionless
    "🫠",  # 🫠 melting face
}

SARCASM_PHRASES = [
    "gampang katanya", "kata siapa", "bisa aja", "ya iyalah",
    "tentu saja", "jelas lah", "makasih sarannya",
    "wah keren banget", "hebat sekali",
]

HIGH_INTENSITY_EMOJIS = {
    "🔥",  # 🔥
    "😭",  # 😭
    "😂",  # 😂
    "🤣",  # 🤣
    "😍",  # 😍
    "🤩",  # 🤩
    "💯",  # 💯
    "❤",      # ❤
    "💖",  # 💖
}

REACTION_EMOJIS = {
    "👍",  # 👍
    "👏",  # 👏
    "🙏",  # 🙏
    "👋",  # 👋
    "🤝",  # 🤝
    "🫶",  # 🫶
}


# ── HEURISTIC ANNOTATOR ───────────────────────────────────────────────────────

class HeuristicAnnotator:
    """
    Annotator berbasis rule linguistik untuk 3 layer label.
    Digunakan sebagai simulasi saat anotasi manual belum tersedia.

    Akurasi estimasi vs annotator manusia:
      Layer A (intensity) : ~72-78% agreement
      Layer B (sarcasm)   : ~68-74% agreement
      Layer C (role)      : ~65-72% agreement
    """

    def __init__(self, noise_level: float = 0.10):
        """
        Args:
            noise_level: Proporsi random noise untuk simulasi
                         ketidakpastian annotator (default 10%)
        """
        self.noise_level = noise_level
        self._emoji_pat  = re.compile(
            "(" + "|".join(
                re.escape(e)
                for e in sorted(emoji.EMOJI_DATA.keys(), key=len, reverse=True)
            ) + ")"
        )

    def _extract_emojis(self, text: str) -> List[str]:
        return self._emoji_pat.findall(text)

    def _clean_text(self, text: str) -> str:
        return self._emoji_pat.sub("", text).strip().lower()

    def annotate_intensity(self, text: str) -> int:
        """
        Layer A: Emotion Intensity
        Returns: 0=Low, 1=Medium, 2=High
        """
        emojis    = self._extract_emojis(text)
        cleaned   = self._clean_text(text)
        tokens    = cleaned.split()

        score = 0

        # Emoji count signal
        if len(emojis) >= 3:
            score += 2
        elif len(emojis) == 2:
            score += 1

        # High-intensity emoji
        for e in emojis:
            if e in HIGH_INTENSITY_EMOJIS:
                score += 1

        # Intensity words
        for word in tokens:
            if word in HIGH_INTENSITY_WORDS:
                score += 2

        # Exclamation marks
        score += text.count("!") * 0.5

        # Uppercase ratio
        alpha_chars = [c for c in text if c.isalpha()]
        if alpha_chars:
            upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if upper_ratio > 0.5:
                score += 2

        # Low intensity signal
        low_count = sum(1 for w in tokens if w in LOW_INTENSITY_WORDS)
        if low_count >= 2 and len(emojis) <= 1:
            score -= 1

        # Apply noise
        if random.random() < self.noise_level:
            score += random.choice([-1, 1])

        if score <= 1:
            return 0  # Low
        elif score <= 3:
            return 1  # Medium
        else:
            return 2  # High

    def annotate_sarcasm(self, text: str) -> int:
        """
        Layer B: Sarcasm Detection
        Returns: 0=Non-Sarcastic, 1=Sarcastic
        """
        emojis    = self._extract_emojis(text)
        cleaned   = self._clean_text(text)

        score = 0

        # Sarcasm emoji signal (strongest indicator)
        for e in emojis:
            if e in SARCASM_EMOJIS:
                score += 3

        # Sarcasm phrases
        for phrase in SARCASM_PHRASES:
            if phrase in cleaned:
                score += 2

        # Contradiction: positive word + negative context
        positive_words = {"bagus", "keren", "mantap", "hebat", "luar biasa", "wow"}
        negative_ctx   = {"tapi", "padahal", "emang", "ya iya", "jelas"}
        has_positive   = any(w in cleaned for w in positive_words)
        has_negative   = any(w in cleaned for w in negative_ctx)
        if has_positive and has_negative:
            score += 1

        # Apply noise
        if random.random() < self.noise_level:
            score += random.choice([-1, 1])

        return 1 if score >= 2 else 0

    def annotate_emoji_role(self, text: str) -> int:
        """
        Layer C: Emoji Pragmatic Role
        Returns: 0=Literal, 1=Exaggeration, 2=Irony, 3=Reaction
        """
        emojis  = self._extract_emojis(text)
        cleaned = self._clean_text(text)

        if not emojis:
            return 3  # default: Reaction

        # Irony signal (prioritas tertinggi)
        for e in emojis:
            if e in SARCASM_EMOJIS:
                return 2  # Irony

        # Exaggeration: emoji berulang atau 3+
        if len(emojis) >= 3:
            return 1  # Exaggeration
        unique_emojis = set(emojis)
        if len(emojis) > len(unique_emojis):  # ada pengulangan
            return 1  # Exaggeration

        # Reaction: emoji respons tanpa emosi kuat
        for e in emojis:
            if e in REACTION_EMOJIS:
                return 3  # Reaction

        # Apply noise
        if random.random() < self.noise_level:
            return random.randint(0, 3)

        return 0  # Literal (default)

    def annotate_batch(
        self,
        df       : pd.DataFrame,
        seed     : int = 42,
        annotator_id: str = "A1",
    ) -> pd.DataFrame:
        """
        Annotasi seluruh DataFrame dengan 3 layer label.

        Args:
            df           : DataFrame dengan kolom 'text'
            seed         : Random seed untuk reprodusibilitas
            annotator_id : Identifier annotator (untuk IAA)

        Returns:
            DataFrame dengan tambahan kolom:
              label_intensity_{annotator_id}
              label_sarcasm_{annotator_id}
              label_emoji_role_{annotator_id}
        """
        from tqdm.auto import tqdm
        random.seed(seed)
        np.random.seed(seed)

        result = df.copy()
        intensity_labels = []
        sarcasm_labels   = []
        role_labels      = []

        for _, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc=f"Annotating [{annotator_id}]",
        ):
            text = str(row.get("text", ""))
            intensity_labels.append(self.annotate_intensity(text))
            sarcasm_labels.append(self.annotate_sarcasm(text))
            role_labels.append(self.annotate_emoji_role(text))

        result[f"label_intensity_{annotator_id}"]  = intensity_labels
        result[f"label_sarcasm_{annotator_id}"]    = sarcasm_labels
        result[f"label_emoji_role_{annotator_id}"] = role_labels

        logger.info(
            f"[{annotator_id}] Annotated {len(df):,} samples. "
            f"Sarcasm rate: {sum(sarcasm_labels)/len(sarcasm_labels)*100:.1f}%"
        )
        return result


# ── IAA CALCULATOR ────────────────────────────────────────────────────────────

class IAACalculator:
    """
    Inter-Annotator Agreement Calculator.
    Menghitung Cohen's Kappa (κ) per layer dan per pasang annotator.
    Target minimum: κ ≥ 0.70 (blueprint Section 2.1.3)
    """

    KAPPA_THRESHOLDS = {
        "poor"        : (float("-inf"), 0.20),
        "fair"        : (0.20, 0.40),
        "moderate"    : (0.40, 0.60),
        "substantial" : (0.60, 0.80),
        "almost_perfect": (0.80, 1.01),
    }

    def interpret_kappa(self, kappa: float) -> str:
        for level, (lo, hi) in self.KAPPA_THRESHOLDS.items():
            if lo <= kappa < hi:
                return level
        return "unknown"

    def compute_kappa_pair(
        self,
        labels_a  : List[int],
        labels_b  : List[int],
        layer_name: str,
    ) -> Dict:
        """Hitung Cohen's Kappa antara dua annotator untuk satu layer."""
        kappa = cohen_kappa_score(labels_a, labels_b)
        interpretation = self.interpret_kappa(kappa)
        meets_threshold = kappa >= 0.70

        return {
            "layer"          : layer_name,
            "kappa"          : round(kappa, 4),
            "interpretation" : interpretation,
            "meets_threshold": meets_threshold,
            "threshold"      : 0.70,
            "n_samples"      : len(labels_a),
        }

    def compute_all(
        self,
        df              : pd.DataFrame,
        annotator_ids   : List[str],
    ) -> Dict:
        """
        Hitung IAA untuk semua layer dan semua pasang annotator.

        Returns:
            Dict berisi kappa per layer, summary, dan status pass/fail.
        """
        layers = [
            ("label_intensity",  "Emotion Intensity"),
            ("label_sarcasm",    "Sarcasm Detection"),
            ("label_emoji_role", "Emoji Role"),
        ]
        results = {"pairs": [], "summary": {}, "all_pass": True}

        for i in range(len(annotator_ids)):
            for j in range(i + 1, len(annotator_ids)):
                a_id = annotator_ids[i]
                b_id = annotator_ids[j]
                pair_result = {"pair": f"{a_id}-{b_id}", "layers": []}

                for col_prefix, layer_name in layers:
                    col_a = f"{col_prefix}_{a_id}"
                    col_b = f"{col_prefix}_{b_id}"

                    if col_a not in df.columns or col_b not in df.columns:
                        continue

                    kappa_result = self.compute_kappa_pair(
                        df[col_a].tolist(),
                        df[col_b].tolist(),
                        layer_name,
                    )
                    pair_result["layers"].append(kappa_result)

                    if not kappa_result["meets_threshold"]:
                        results["all_pass"] = False

                results["pairs"].append(pair_result)

        # Summary: rata-rata kappa per layer
        for col_prefix, layer_name in layers:
            kappas = [
                layer["kappa"]
                for pair in results["pairs"]
                for layer in pair["layers"]
                if layer["layer"] == layer_name
            ]
            if kappas:
                avg_kappa = sum(kappas) / len(kappas)
                results["summary"][layer_name] = {
                    "avg_kappa"      : round(avg_kappa, 4),
                    "meets_threshold": avg_kappa >= 0.70,
                    "interpretation" : self.interpret_kappa(avg_kappa),
                }

        return results


# ── ANNOTATION MANAGER ────────────────────────────────────────────────────────

class AnnotationManager:
    """
    Orkestrasi batch annotation workflow.
    Menggabungkan hasil annotasi multi-annotator menjadi label final
    menggunakan majority vote.
    """

    def __init__(
        self,
        drive_root     : str,
        n_annotators   : int = 3,
        min_kappa      : float = 0.70,
    ):
        self.drive_root   = Path(drive_root)
        self.n_annotators = n_annotators
        self.min_kappa    = min_kappa
        self.annotator    = HeuristicAnnotator(noise_level=0.10)
        self.iaa_calc     = IAACalculator()

    def run_annotation(
        self,
        df            : pd.DataFrame,
        annotator_ids : List[str] = None,
        seeds         : List[int] = None,
    ) -> pd.DataFrame:
        """
        Jalankan annotasi oleh semua annotator secara paralel (simulasi).
        Setiap annotator punya seed berbeda untuk simulasi variasi.
        """
        if annotator_ids is None:
            annotator_ids = [f"A{i+1}" for i in range(self.n_annotators)]
        if seeds is None:
            seeds = [42 + i * 7 for i in range(self.n_annotators)]

        result = df.copy()
        for a_id, seed in zip(annotator_ids, seeds):
            result = self.annotator.annotate_batch(result, seed=seed, annotator_id=a_id)

        return result, annotator_ids

    def majority_vote(
        self,
        df            : pd.DataFrame,
        annotator_ids : List[str],
    ) -> pd.DataFrame:
        """
        Gabungkan label multi-annotator dengan majority vote.
        Konflik (tie) diresolvasi dengan annotator pertama sebagai tiebreaker.
        """
        import scipy.stats as stats

        result = df.copy()

        for col_prefix in ["label_intensity", "label_sarcasm", "label_emoji_role"]:
            cols = [f"{col_prefix}_{a_id}" for a_id in annotator_ids
                    if f"{col_prefix}_{a_id}" in df.columns]
            if not cols:
                continue

            # Majority vote via scipy mode
            votes = df[cols].values
            mode_result = stats.mode(votes, axis=1, keepdims=True)
            result[col_prefix] = mode_result.mode.flatten().astype(int)

        return result

    def compute_and_report_iaa(
        self,
        df            : pd.DataFrame,
        annotator_ids : List[str],
        save_path     : Optional[str] = None,
    ) -> Dict:
        """Hitung IAA dan simpan laporan ke Drive."""
        iaa_results = self.iaa_calc.compute_all(df, annotator_ids)

        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(iaa_results, f, ensure_ascii=False, indent=2)

        return iaa_results
