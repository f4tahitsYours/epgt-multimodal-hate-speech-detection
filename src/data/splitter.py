"""
splitter.py — Stratified Dataset Split untuk EPGT.

Stratifikasi adaptif berdasarkan ukuran dataset:
  - Dataset besar (≥5000) : composite key 4 dimensi
    {intensity}_{sarcasm}_{platform}_{density}
  - Dataset sedang (≥500) : composite key 3 dimensi
    {intensity}_{sarcasm}_{platform}
  - Dataset kecil (<500)  : composite key 1 dimensi
    {intensity} saja (paling stabil)

Rare combinations (< min_strat_count) digabung ke bucket "other".
Split ratio: 70/15/15 (train/val/test).
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class StratifiedSplitter:
    """
    Stratified split dengan composite stratification key adaptif.
    Secara otomatis menyesuaikan granularitas key berdasarkan
    ukuran dataset agar train_test_split tidak error.
    """

    def __init__(
        self,
        train_ratio     : float = 0.70,
        val_ratio       : float = 0.15,
        test_ratio      : float = 0.15,
        random_seed     : int   = 42,
        min_strat_count : int   = 2,
    ):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, \
            "Rasio harus berjumlah 1.0"

        self.train_ratio     = train_ratio
        self.val_ratio       = val_ratio
        self.test_ratio      = test_ratio
        self.random_seed     = random_seed
        self.min_strat_count = min_strat_count

    def _build_composite_key(self, df: pd.DataFrame) -> pd.Series:
        """
        Bangun composite stratification key secara adaptif.
        Granularitas dikurangi jika dataset terlalu kecil.
        """
        n = len(df)

        if n >= 5000:
            # Full 4-dimensi: intensity + sarcasm + platform + density
            key = (
                df["label_intensity"].astype(str) + "_" +
                df["label_sarcasm"].astype(str)   + "_" +
                df["platform"].astype(str)         + "_" +
                df["emoji_density"].astype(str)
            )
            level = "4D (intensity+sarcasm+platform+density)"

        elif n >= 500:
            # 3-dimensi: intensity + sarcasm + platform
            key = (
                df["label_intensity"].astype(str) + "_" +
                df["label_sarcasm"].astype(str)   + "_" +
                df["platform"].astype(str)
            )
            level = "3D (intensity+sarcasm+platform)"

        elif n >= 100:
            # 2-dimensi: intensity + sarcasm
            key = (
                df["label_intensity"].astype(str) + "_" +
                df["label_sarcasm"].astype(str)
            )
            level = "2D (intensity+sarcasm)"

        else:
            # 1-dimensi: intensity saja (paling stabil untuk dataset kecil)
            key   = df["label_intensity"].astype(str)
            level = "1D (intensity only)"

        logger.info(f"Stratification key: {level} | n={n:,}")
        print(f"  Stratification level : {level}")
        return key

    def _handle_rare_strata(self, strat_key: pd.Series) -> pd.Series:
        """
        Gabungkan strata dengan count < min_strat_count ke "other".
        Diperlukan agar setiap strata punya minimal 2 samples
        untuk bisa di-split menjadi train dan test.
        """
        counts    = strat_key.value_counts()
        rare_keys = counts[counts < self.min_strat_count].index

        if len(rare_keys) > 0:
            logger.info(
                f"Merging {len(rare_keys)} rare strata "
                f"(< {self.min_strat_count} samples) into 'other'"
            )
            print(f"  Rare strata merged   : {len(rare_keys)} → 'other'")

        return strat_key.where(~strat_key.isin(rare_keys), other="other")

    def _safe_split(
        self,
        df       : pd.DataFrame,
        test_size: float,
        strat_key: pd.Series,
        label    : str = "",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Wrapper train_test_split dengan fallback ke non-stratified
        jika strata masih terlalu kecil setelah handling rare strata.
        """
        try:
            return train_test_split(
                df,
                test_size    = test_size,
                random_state = self.random_seed,
                stratify     = strat_key,
                shuffle      = True,
            )
        except ValueError as e:
            logger.warning(
                f"Stratified split failed ({label}): {e}. "
                f"Falling back to random split."
            )
            print(f"  WARNING: Stratified split failed ({label}), using random split.")
            return train_test_split(
                df,
                test_size    = test_size,
                random_state = self.random_seed,
                shuffle      = True,
            )

    def split(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Lakukan stratified split 70/15/15.

        Steps:
          1. Bangun composite key (adaptif)
          2. Handle rare strata
          3. Split train (70%) vs temp (30%)
          4. Split temp → val (50%) + test (50%) = 15%/15%
          5. Fallback ke random split jika stratified gagal

        Returns:
          (df_train, df_val, df_test)
        """
        n = len(df)
        print(f"\nSplitting {n:,} samples (70/15/15)...")

        df = df.reset_index(drop=True).copy()

        # Step 1 & 2: Composite key + handle rare strata
        strat_key         = self._build_composite_key(df)
        strat_key_cleaned = self._handle_rare_strata(strat_key)
        df["_strat_key"]  = strat_key_cleaned

        # Step 3: Train (70%) vs Temp (30%)
        temp_ratio   = self.val_ratio + self.test_ratio  # 0.30
        df_train, df_temp = self._safe_split(
            df        = df,
            test_size = temp_ratio,
            strat_key = df["_strat_key"],
            label     = "train/temp",
        )

        # Step 4: Val (50% of temp) vs Test (50% of temp)
        val_from_temp = self.val_ratio / temp_ratio       # 0.50
        df_val, df_test = self._safe_split(
            df        = df_temp,
            test_size = 1 - val_from_temp,
            strat_key = df_temp["_strat_key"],
            label     = "val/test",
        )

        # Bersihkan helper column
        for split_df in [df_train, df_val, df_test]:
            split_df.drop(columns=["_strat_key"], inplace=True, errors="ignore")

        df_train = df_train.reset_index(drop=True)
        df_val   = df_val.reset_index(drop=True)
        df_test  = df_test.reset_index(drop=True)

        return df_train, df_val, df_test

    def verify_split(
        self,
        df_train : pd.DataFrame,
        df_val   : pd.DataFrame,
        df_test  : pd.DataFrame,
        df_full  : pd.DataFrame,
    ) -> Dict:
        """
        Verifikasi kualitas split:
          1. Rasio aktual mendekati target
          2. Tidak ada ID yang bocor antar split
          3. Distribusi label terjaga
        """
        total   = len(df_full)
        results = {
            "sizes": {
                "train": len(df_train),
                "val"  : len(df_val),
                "test" : len(df_test),
                "total": total,
            },
            "ratios": {
                "train": round(len(df_train) / total, 4),
                "val"  : round(len(df_val)   / total, 4),
                "test" : round(len(df_test)  / total, 4),
            },
            "no_leakage"   : True,
            "label_balance": {},
        }

        # Leakage check
        train_ids = set(df_train["id"].astype(str))
        val_ids   = set(df_val["id"].astype(str))
        test_ids  = set(df_test["id"].astype(str))

        if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
            results["no_leakage"] = False
            logger.error("DATA LEAKAGE DETECTED between splits!")

        # Label distribution
        for col in ["label_intensity", "label_sarcasm", "label_emoji_role"]:
            if col in df_train.columns:
                results["label_balance"][col] = {
                    "train": df_train[col].value_counts(normalize=True).round(3).to_dict(),
                    "val"  : df_val[col].value_counts(normalize=True).round(3).to_dict(),
                    "test" : df_test[col].value_counts(normalize=True).round(3).to_dict(),
                }

        return results
