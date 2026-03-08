"""
preprocessor.py — 7-Stage Text Preprocessing Pipeline untuk EPGT.

Stage 1 : Unicode normalization (NFKC)
Stage 2 : Emoji extraction (pada raw text, sebelum cleaning)
Stage 3 : Text normalization (URL, mention, repeated chars)
Stage 4 : IndoBERT tokenization (max_length=128)
Stage 5 : Inclusion filter verification
Stage 6 : Feature engineering (positions, repetition, sentiment)
Stage 7 : Batch processing + save
"""

import re
import unicodedata
import logging
import numpy as np
import pandas as pd
import emoji
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# ── STAGE 1 & 3: Text Normalizer ─────────────────────────────────────────────

class IndonesianTextNormalizer:
    """
    Normalisasi teks Indonesia untuk NLP.
    Urutan operasi penting: emoji diekstrak DULU sebelum normalisasi.
    """

    def __init__(self):
        self._emoji_pat = re.compile(
            "(" + "|".join(
                re.escape(e)
                for e in sorted(emoji.EMOJI_DATA.keys(), key=len, reverse=True)
            ) + ")"
        )
        # Patterns normalisasi
        self._url_pat       = re.compile(r"https?://\S+|www\.\S+")
        self._mention_pat   = re.compile(r"@\w+")
        self._hashtag_pat   = re.compile(r"#(\w+)")
        self._repeated_char = re.compile(r"(.)\1{2,}")
        self._repeated_punc = re.compile(r"([!?.]){2,}")
        self._whitespace    = re.compile(r"\s+")

    def normalize_unicode(self, text: str) -> str:
        """Stage 1: Unicode NFKC normalization."""
        return unicodedata.normalize("NFKC", text)

    def extract_emojis(self, text: str) -> List[str]:
        """Stage 2: Ekstrak emoji dari raw text (sebelum normalisasi)."""
        return self._emoji_pat.findall(text)

    def remove_emojis(self, text: str) -> str:
        """Hapus emoji dari teks."""
        return self._emoji_pat.sub(" ", text)

    def normalize_text(self, text: str) -> str:
        """
        Stage 3: Normalisasi teks.
        Urutan: URL → mention → hashtag → repeated chars → whitespace
        Emoji sudah dihapus sebelum fungsi ini dipanggil.
        """
        text = text.lower()
        text = self._url_pat.sub("", text)               # hapus URL
        text = self._mention_pat.sub("", text)           # hapus @mention
        text = self._hashtag_pat.sub(r"\1", text)      # #kata → kata
        text = self._repeated_char.sub(r"\1\1", text) # bangettt → bangett
        text = self._repeated_punc.sub(r"\1", text)    # !!! → !
        text = self._whitespace.sub(" ", text).strip()  # whitespace
        return text

    def get_emoji_positions(
        self,
        text: str,
        emojis: List[str],
    ) -> List[float]:
        """
        Hitung posisi normalized tiap emoji dalam teks.
        Normalized position = char_position / len(text)
        Sesuai blueprint: p_i ∈ ℝ¹
        """
        if not text or not emojis:
            return []

        positions = []
        search_start = 0

        for e in emojis:
            idx = text.find(e, search_start)
            if idx == -1:
                idx = text.find(e, 0)
            if idx != -1:
                normalized_pos = idx / max(len(text), 1)
                positions.append(round(normalized_pos, 6))
                search_start = idx + len(e)
            else:
                positions.append(0.0)

        return positions

    def get_repetition_flags(self, emojis: List[str]) -> List[int]:
        """
        Hitung repetition flag r_i untuk tiap emoji.
        r_i = 1 jika emoji_i == emoji_{i-1}, else 0.
        Sesuai blueprint: r_i ∈ ℝ¹
        """
        if not emojis:
            return []
        flags = [0]  # emoji pertama tidak bisa berulang
        for i in range(1, len(emojis)):
            flags.append(1 if emojis[i] == emojis[i-1] else 0)
        return flags

    def process(self, text: str) -> Dict:
        """
        Full preprocessing satu sampel.
        Returns dict dengan semua field yang dibutuhkan graph builder.
        """
        # Stage 1: Unicode
        text_unicode = self.normalize_unicode(str(text))

        # Stage 2: Ekstrak emoji dari raw text
        emojis        = self.extract_emojis(text_unicode)
        emoji_count   = len(emojis)
        emoji_sequence= ",".join(emojis)

        # Stage 3: Normalisasi teks (hapus emoji dulu)
        text_no_emoji = self.remove_emojis(text_unicode)
        cleaned_text  = self.normalize_text(text_no_emoji)

        # Stage 6: Feature engineering
        emoji_positions   = self.get_emoji_positions(text_unicode, emojis)
        repetition_flags  = self.get_repetition_flags(emojis)

        return {
            "raw_text"         : text,
            "cleaned_text"     : cleaned_text,
            "emoji_sequence"   : emoji_sequence,
            "emoji_count"      : emoji_count,
            "emoji_list"       : emojis,
            "emoji_positions"  : emoji_positions,
            "repetition_flags" : repetition_flags,
        }


# ── STAGE 4: IndoBERT Tokenizer ───────────────────────────────────────────────

class EPGTTokenizer:
    """
    Wrapper IndoBERT tokenizer untuk EPGT.
    Model: indobenchmark/indobert-base-p1
    Fallback: bert-base-multilingual-cased
    Output: input_ids, attention_mask, token_type_ids (max_length=128)
    """

    PRIMARY_MODEL  = "indobenchmark/indobert-base-p1"
    FALLBACK_MODEL = "bert-base-multilingual-cased"

    def __init__(self, max_length: int = 128):
        self.max_length = max_length
        self.tokenizer  = self._load_tokenizer()

    def _load_tokenizer(self) -> AutoTokenizer:
        for model_name in [self.PRIMARY_MODEL, self.FALLBACK_MODEL]:
            try:
                tok = AutoTokenizer.from_pretrained(model_name)
                logger.info(f"Tokenizer loaded: {model_name}")
                print(f"  Tokenizer: {model_name}")
                return tok
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
        raise RuntimeError("Semua tokenizer gagal dimuat.")

    def tokenize(self, text: str) -> Dict:
        """
        Tokenize satu teks.
        Returns: input_ids, attention_mask, token_type_ids (semua list int)
        """
        encoded = self.tokenizer(
            text,
            max_length      = self.max_length,
            padding         = "max_length",
            truncation      = True,
            return_tensors  = None,
        )
        return {
            "input_ids"      : encoded["input_ids"],
            "attention_mask" : encoded["attention_mask"],
            "token_type_ids" : encoded.get("token_type_ids", [0] * self.max_length),
        }

    def tokenize_batch(
        self,
        texts     : List[str],
        batch_size: int = 64,
    ) -> Dict[str, List]:
        """
        Tokenize batch teks secara efisien.
        Returns dict of lists: input_ids, attention_mask, token_type_ids
        """
        from tqdm.auto import tqdm
        all_input_ids       = []
        all_attention_masks = []
        all_token_type_ids  = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing"):
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                max_length      = self.max_length,
                padding         = "max_length",
                truncation      = True,
                return_tensors  = None,
            )
            all_input_ids.extend(encoded["input_ids"])
            all_attention_masks.extend(encoded["attention_mask"])
            all_token_type_ids.extend(
                encoded.get("token_type_ids", [[0]*self.max_length]*len(batch))
            )

        return {
            "input_ids"      : all_input_ids,
            "attention_mask" : all_attention_masks,
            "token_type_ids" : all_token_type_ids,
        }


# ── FULL PIPELINE ─────────────────────────────────────────────────────────────

class EPGTPreprocessor:
    """
    Orkestrasi 7-stage preprocessing pipeline.
    Input : DataFrame dengan kolom 'text'
    Output: DataFrame dengan semua feature columns + tokenization
    """

    def __init__(self, max_length: int = 128):
        self.normalizer = IndonesianTextNormalizer()
        self.tokenizer  = EPGTTokenizer(max_length=max_length)
        self.max_length = max_length

    def process_dataframe(
        self,
        df          : pd.DataFrame,
        text_col    : str = "text",
        batch_size  : int = 64,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Proses seluruh DataFrame melalui 7-stage pipeline.

        Returns:
            DataFrame dengan tambahan kolom:
              cleaned_text, emoji_sequence, emoji_count,
              emoji_list, emoji_positions, repetition_flags,
              input_ids, attention_mask, token_type_ids
        """
        from tqdm.auto import tqdm

        result = df.copy()

        # Stage 1-2-3-6: Normalisasi + feature engineering
        print("Stage 1-3-6: Text normalization + emoji extraction...")
        processed_rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing",
                           disable=not show_progress):
            proc = self.normalizer.process(str(row.get(text_col, "")))
            processed_rows.append(proc)

        proc_df = pd.DataFrame(processed_rows)

        # Update kolom yang sudah ada di df atau tambahkan yang baru
        for col in ["cleaned_text", "emoji_sequence", "emoji_count",
                    "emoji_list", "emoji_positions", "repetition_flags"]:
            if col in proc_df.columns:
                result[col] = proc_df[col].values

        # Stage 4: Tokenisasi cleaned_text
        print("Stage 4: IndoBERT tokenization...")
        cleaned_texts = result["cleaned_text"].fillna("").tolist()
        token_data    = self.tokenizer.tokenize_batch(
            cleaned_texts,
            batch_size=batch_size,
        )
        result["input_ids"]       = token_data["input_ids"]
        result["attention_mask"]  = token_data["attention_mask"]
        result["token_type_ids"]  = token_data["token_type_ids"]

        # Stage 5: Inclusion filter — drop samples yang tidak memenuhi syarat
        before = len(result)
        result = result[result["emoji_count"] >= 1].reset_index(drop=True)
        after  = len(result)
        if before != after:
            logger.warning(f"Stage 5 filter: removed {before-after} samples without emoji")
            print(f"Stage 5: Filtered {before-after} samples (no emoji)")
        else:
            print(f"Stage 5: All {after} samples passed inclusion filter")

        print(f"Preprocessing complete: {len(result):,} samples")
        return result
