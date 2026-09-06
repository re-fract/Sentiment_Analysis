"""
src/data/dataset.py

PyTorch Dataset classes for transformer training.

Two dataset classes:
  AspectDetectionDataset  — multi-label: input = review text
                             label = binary vector over coarse categories
  SentimentDataset        — single-label: input = [CLS] text [SEP] CATEGORY [SEP]
                             label = sentiment index (0=positive, 1=negative, 2=neutral)
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import Dataset
import pandas as pd

from config.taxonomy import (
    COARSE_CATEGORIES, SENTIMENT_LABELS,
    COARSE_TO_ID, SENTIMENT_TO_ID, normalise_sentiment,
)


# ── Aspect Detection Dataset ──────────────────────────────────────────────────

class AspectDetectionDataset(Dataset):
    """
    Multi-label classification: given a review text, predict which coarse
    aspect categories are discussed.

    Label: FloatTensor of shape [num_coarse] with 0/1 per category.
    """

    def __init__(
        self,
        df:        pd.DataFrame,
        tokenizer: Any,
        max_length:int = 128,
    ):
        self.samples   = []
        self.tokenizer = tokenizer
        self.max_length= max_length

        for _, row in df.iterrows():
            text = str(row["text"])
            try:
                cats = list(set(ast.literal_eval(row["categories_coarse"])))
            except Exception:
                continue
            if not cats:
                continue
            # Build multi-hot label
            label = torch.zeros(len(COARSE_CATEGORIES))
            for cat in cats:
                if cat in COARSE_TO_ID:
                    label[COARSE_TO_ID[cat]] = 1.0
            self.samples.append((text, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text, label = self.samples[idx]
        enc = self.tokenizer(
            text,
            max_length     = self.max_length,
            truncation     = True,
            padding        = "max_length",
            return_tensors = "pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         label,
        }


# ── Sentiment Classification Dataset ─────────────────────────────────────────

class SentimentDataset(Dataset):
    """
    Single-label classification: predict sentiment for a specific aspect.

    Input format:  "[CLS] {review_text} [SEP] {coarse_category} [SEP]"

    This aspect-conditioned format is the core design decision — it forces
    the model to focus sentiment prediction on the specified aspect, which
    is critical for multi-aspect sentences.

    Label: LongTensor — index into SENTIMENT_LABELS.
    """

    def __init__(
        self,
        df:        pd.DataFrame,
        tokenizer: Any,
        max_length:int = 128,
    ):
        self.samples   = []
        self.tokenizer = tokenizer
        self.max_length= max_length

        for _, row in df.iterrows():
            text = str(row["text"])
            try:
                cats  = ast.literal_eval(row["categories_coarse"])
                sents = ast.literal_eval(row["sentiments"])
            except Exception:
                continue
            for cat, sent in zip(cats, sents):
                sent_norm = normalise_sentiment(sent)
                if sent_norm not in SENTIMENT_TO_ID:
                    continue
                label = SENTIMENT_TO_ID[sent_norm]
                self.samples.append((text, cat, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text, aspect_cat, label = self.samples[idx]

        # Aspect-conditioned input: text [SEP] category
        enc = self.tokenizer(
            text,
            aspect_cat,
            max_length     = self.max_length,
            truncation     = True,
            padding        = "max_length",
            return_tensors = "pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get("token_type_ids",
                                      torch.zeros(self.max_length, dtype=torch.long)).squeeze(0),
            "labels":         torch.tensor(label, dtype=torch.long),
        }
