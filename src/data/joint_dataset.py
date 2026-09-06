"""
src/data/joint_dataset.py

Dataset classes for joint multi-task training (T9 / T10).

ImplicitDataset
---------------
  Same as AspectDetectionDataset but also returns per-category implicit labels.
  Used by T10's implicit head — only ACOS rows have meaningful is_implicit values,
  but SemEval / MAMS rows return all-zeros (valid negatives for the head).

  Columns consumed from the DataFrame:
    text               : str
    categories_coarse  : list[str]  (Python repr string)
    is_implicit_flags  : list[bool] (Python repr string)
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from config.taxonomy import COARSE_CATEGORIES, COARSE_TO_ID


class ImplicitDataset(Dataset):
    """
    Aspect-detection dataset that also produces implicit-aspect labels.

    Each sample returns a dict with:
        input_ids      : (max_length,)
        attention_mask : (max_length,)
        token_type_ids : (max_length,) — zeros if tokenizer doesn't produce them
        labels         : (N,) float   — multi-label aspect presence (same as AspectDetectionDataset)
        implicit_labels: (N,) float   — per-category is_implicit flag

    where N = len(COARSE_CATEGORIES) = 5.
    """

    def __init__(
        self,
        df:         pd.DataFrame,
        tokenizer:  PreTrainedTokenizerBase,
        max_length: int = 128,
    ):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.records: list[dict] = []

        for _, row in df.iterrows():
            text = str(row.get("text", "")).strip()
            if not text:
                continue

            # ── Aspect presence labels ─────────────────────────────────────
            asp_label = torch.zeros(len(COARSE_CATEGORIES), dtype=torch.float)
            imp_label = torch.zeros(len(COARSE_CATEGORIES), dtype=torch.float)

            try:
                cats     = ast.literal_eval(str(row.get("categories_coarse", "[]")))
                implicits = ast.literal_eval(str(row.get("is_implicit_flags",  "[]")))
            except Exception:
                cats, implicits = [], []

            for cat, is_imp in zip(cats, implicits):
                idx = COARSE_TO_ID.get(cat)
                if idx is not None:
                    asp_label[idx] = 1.0
                    if is_imp:
                        imp_label[idx] = 1.0

            self.records.append({
                "text":           text,
                "labels":         asp_label,
                "implicit_labels": imp_label,
            })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        enc = self.tokenizer(
            rec["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        result = {
            "input_ids":       enc["input_ids"].squeeze(0),
            "attention_mask":  enc["attention_mask"].squeeze(0),
            "labels":          rec["labels"],
            "implicit_labels": rec["implicit_labels"],
        }
        if "token_type_ids" in enc:
            result["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        return result
