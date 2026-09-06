"""
src/data/splits.py

Utilities for managing train/val/test splits across all datasets.

- SemEval: no official val set -> carve 10% from combined_train (stratified)
- MAMS: pre-split (train/val/test)
- ACOS: pre-split (train/dev/test); dev -> val for consistency

Also builds the final merged training sets used by each model variant:
  - semeval_only_train   : combined_train.csv
  - full_train           : combined_train + mams_train + acos_train
  - augmented_train      : full_train + synthetic_filtered (when available)
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import numpy as np
from src.utils.io import load_csv, save_csv, load_config
from src.utils.logging import get_logger
from src.data.text_cleaning import clean_series

log = get_logger("splits")


def _clean_split(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text cleaning to the 'text' column in-place."""
    if "text" in df.columns:
        df = df.copy()
        df["text"] = clean_series(df["text"])
    return df


# ── Carve val set from SemEval combined_train ─────────────────────────────────

def carve_semeval_val(
    df: pd.DataFrame,
    val_fraction: float = 0.10,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split combined_train into train/val (stratified on first aspect coarse category).
    Returns (train_df, val_df).
    """
    # Extract first coarse category as stratification key
    def first_coarse(row):
        try:
            cats = ast.literal_eval(row["categories_coarse"])
            return cats[0] if cats else "General"
        except Exception:
            return "General"

    df = df.copy()
    df["_strat_key"] = df.apply(first_coarse, axis=1)

    val_parts  = []
    train_parts = []
    rng = np.random.default_rng(seed)

    for key, group in df.groupby("_strat_key"):
        n_val = max(1, int(round(len(group) * val_fraction)))
        idx   = rng.permutation(len(group))
        val_idx   = idx[:n_val]
        train_idx = idx[n_val:]
        val_parts.append(group.iloc[val_idx])
        train_parts.append(group.iloc[train_idx])

    val_df   = pd.concat(val_parts).drop(columns=["_strat_key"]).reset_index(drop=True)
    train_df = pd.concat(train_parts).drop(columns=["_strat_key"]).reset_index(drop=True)
    log.info(f"SemEval split: {len(train_df)} train / {len(val_df)} val")
    return train_df, val_df


# ── Build merged training sets ────────────────────────────────────────────────

def build_splits(cfg: dict, augmented: bool = False) -> dict[str, pd.DataFrame]:
    """
    Load all processed CSVs and return a dict of split DataFrames.

    Keys:
        semeval_train, semeval_val, semeval_test
        mams_train, mams_val, mams_test
        acos_train, acos_val, acos_test
        full_train        (semeval + mams + acos)
        full_val          (semeval_val + mams_val + acos_val)
        augmented_train   (full_train + synthetic, if augmented=True)
    """
    proc = cfg["paths"]["processed"]
    splits: dict[str, pd.DataFrame] = {}

    # ── SemEval ──────────────────────────────────────────────────────────────
    combined = _clean_split(load_csv(proc["combined_train"]))
    semeval_train, semeval_val = carve_semeval_val(
        combined, val_fraction=cfg["evaluation"]["semeval_val_fraction"])
    splits["semeval_train"] = semeval_train
    splits["semeval_val"]   = semeval_val
    # Merge all SemEval test sets
    test_parts = []
    for key in ["semeval14_test", "semeval15_test", "semeval16_test"]:
        p = proc.get(key)
        if p and Path(p).exists():
            test_parts.append(_clean_split(load_csv(p)))
    splits["semeval_test"]  = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()

    # ── MAMS ─────────────────────────────────────────────────────────────────
    splits["mams_train"] = _clean_split(load_csv(proc["mams_train"]))
    splits["mams_val"]   = _clean_split(load_csv(proc["mams_val"]))
    splits["mams_test"]  = _clean_split(load_csv(proc["mams_test"]))

    # ── ACOS ─────────────────────────────────────────────────────────────────
    splits["acos_train"] = _clean_split(load_csv(proc["acos_train"]))
    splits["acos_val"]   = _clean_split(load_csv(proc["acos_dev"]))
    splits["acos_test"]  = _clean_split(load_csv(proc["acos_test"]))

    # ── Merged sets ───────────────────────────────────────────────────────────
    splits["full_train"] = pd.concat(
        [splits["semeval_train"], splits["mams_train"], splits["acos_train"]],
        ignore_index=True,
    )
    splits["full_val"] = pd.concat(
        [splits["semeval_val"], splits["mams_val"], splits["acos_val"]],
        ignore_index=True,
    )

    # ── Augmented train ───────────────────────────────────────────────────────
    if augmented:
        aug_path = cfg["paths"]["augmented"]["filtered"]
        if Path(aug_path).exists():
            aug_df = _clean_split(load_csv(aug_path))
            splits["augmented_train"] = pd.concat(
                [splits["full_train"], aug_df], ignore_index=True)
            log.info(f"Augmented train: {len(splits['augmented_train'])} rows "
                     f"({len(aug_df)} synthetic)")
        else:
            log.warning(f"Augmented data not found at {aug_path}. "
                        "Using full_train as augmented_train.")
            splits["augmented_train"] = splits["full_train"]

    log.info(f"Split sizes: " + ", ".join(
        f"{k}={len(v)}" for k, v in splits.items()))
    return splits
