"""
dataset.py — PyTorch Dataset untuk EPGT.

Menggabungkan text tokenization data dari preprocessor.py
dengan graph objects dari emoji_graph.py.

Output per item:
  input_ids      : LongTensor (128,)
  attention_mask : LongTensor (128,)
  token_type_ids : LongTensor (128,)
  graph          : torch_geometric.data.Data
  label_intensity: LongTensor ()
  label_sarcasm  : LongTensor ()
  label_emoji_role: LongTensor ()
"""

import ast
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch


class EPGTDataset(Dataset):
    """
    Dataset yang menggabungkan text features dan graph objects.

    Args:
        df     : processed DataFrame dari preprocessor
        graphs : list of torch_geometric.data.Data objects
    """

    def __init__(
        self,
        df    : pd.DataFrame,
        graphs: List[Data],
    ):
        assert len(df) == len(graphs),             f"DataFrame ({len(df)}) dan graphs ({len(graphs)}) harus sama panjang"

        self.df     = df.reset_index(drop=True)
        self.graphs = graphs

        # Parse tokenization columns dari string
        self._input_ids       = self._parse_col("input_ids")
        self._attention_mask  = self._parse_col("attention_mask")
        self._token_type_ids  = self._parse_col("token_type_ids")

    def _parse_col(self, col: str) -> List[List[int]]:
        """Parse kolom yang tersimpan sebagai string list."""
        result = []
        for val in self.df[col]:
            if isinstance(val, str):
                try:
                    parsed = ast.literal_eval(val)
                except Exception:
                    parsed = [0] * 128
            elif isinstance(val, list):
                parsed = val
            else:
                parsed = [0] * 128
            result.append(parsed)
        return result

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]

        return {
            "input_ids"       : torch.tensor(self._input_ids[idx],      dtype=torch.long),
            "attention_mask"  : torch.tensor(self._attention_mask[idx], dtype=torch.long),
            "token_type_ids"  : torch.tensor(self._token_type_ids[idx], dtype=torch.long),
            "graph"           : self.graphs[idx],
            "label_intensity" : torch.tensor(int(row["label_intensity"]),  dtype=torch.long),
            "label_sarcasm"   : torch.tensor(int(row["label_sarcasm"]),    dtype=torch.long),
            "label_emoji_role": torch.tensor(int(row["label_emoji_role"]), dtype=torch.long),
        }


def epgt_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function untuk DataLoader.
    Graph objects di-batch menggunakan PyG Batch.from_data_list.
    """
    return {
        "input_ids"       : torch.stack([b["input_ids"]        for b in batch]),
        "attention_mask"  : torch.stack([b["attention_mask"]   for b in batch]),
        "token_type_ids"  : torch.stack([b["token_type_ids"]   for b in batch]),
        "graph"           : Batch.from_data_list([b["graph"]   for b in batch]),
        "label_intensity" : torch.stack([b["label_intensity"]  for b in batch]),
        "label_sarcasm"   : torch.stack([b["label_sarcasm"]    for b in batch]),
        "label_emoji_role": torch.stack([b["label_emoji_role"] for b in batch]),
    }


def build_dataloaders(
    df_train : pd.DataFrame,
    df_val   : pd.DataFrame,
    df_test  : pd.DataFrame,
    graphs_train: List[Data],
    graphs_val  : List[Data],
    graphs_test : List[Data],
    batch_size  : int = 16,
    num_workers : int = 0,
) -> Dict[str, DataLoader]:
    """
    Buat DataLoaders untuk train, val, test.

    Args:
        batch_size  : batch size untuk training (default 16)
        num_workers : num_workers DataLoader (Colab: 0)
    """
    train_dataset = EPGTDataset(df_train, graphs_train)
    val_dataset   = EPGTDataset(df_val,   graphs_val)
    test_dataset  = EPGTDataset(df_test,  graphs_test)

    return {
        "train": DataLoader(
            train_dataset,
            batch_size  = batch_size,
            shuffle     = True,
            collate_fn  = epgt_collate_fn,
            num_workers = num_workers,
            drop_last   = False,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size  = batch_size * 2,
            shuffle     = False,
            collate_fn  = epgt_collate_fn,
            num_workers = num_workers,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size  = batch_size * 2,
            shuffle     = False,
            collate_fn  = epgt_collate_fn,
            num_workers = num_workers,
        ),
    }
