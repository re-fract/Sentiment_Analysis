"""
parse_all.py — Parse all raw datasets into unified CSVs.

Usage:
    python scripts/parse_all.py
    python scripts/parse_all.py --config config/config.yaml

Outputs (in data/processed/):
    semeval14_{train,test}.csv
    semeval15_{train,test}.csv
    semeval16_{train,test}.csv
    mams_{train,val,test}.csv
    acos_{train,dev,test}.csv
    combined_train.csv          ← SemEval 14+15+16 merged training set
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.data.parsers import PyABSASegParser, MAMSParser, ACOSParser
from src.data.unified_schema import samples_to_dataframe
from src.utils.io import load_config, save_csv
from src.utils.logging import get_logger

log = get_logger("parse_all")


def parse_semeval(cfg: dict) -> dict[str, pd.DataFrame]:
    """Parse SemEval 14 / 15 / 16 .seg and .raw files."""
    raw = cfg["paths"]["raw"]
    out: dict[str, pd.DataFrame] = {}

    configs = [
        ("semeval14", raw["semeval14_train"], raw["semeval14_test"]),
        ("semeval15", raw["semeval15_train"], raw["semeval15_test"]),
        ("semeval16", raw["semeval16_train"], raw["semeval16_test"]),
    ]

    for source, train_path, test_path in configs:
        parser = PyABSASegParser(source=source)

        log.info(f"Parsing {source} train: {train_path}")
        train_samples = parser.parse_file(train_path, split="train")
        df_train = samples_to_dataframe(train_samples)
        out[f"{source}_train"] = df_train
        log.info(f"  -> {len(df_train)} rows")

        log.info(f"Parsing {source} test:  {test_path}")
        test_samples = parser.parse_file(test_path, split="test")
        df_test = samples_to_dataframe(test_samples)
        out[f"{source}_test"] = df_test
        log.info(f"  -> {len(df_test)} rows")

    return out


def parse_mams(cfg: dict) -> dict[str, pd.DataFrame]:
    """Parse MAMS-ACSA XML files."""
    raw = cfg["paths"]["raw"]
    parser = MAMSParser()
    out: dict[str, pd.DataFrame] = {}

    for split, key in [("train", "mams_train"), ("val", "mams_val"), ("test", "mams_test")]:
        path = raw[key]
        log.info(f"Parsing MAMS {split}: {path}")
        samples = parser.parse_file(path, split=split)
        df = samples_to_dataframe(samples)
        out[f"mams_{split}"] = df
        log.info(f"  -> {len(df)} rows")

    return out


def parse_acos(cfg: dict) -> dict[str, pd.DataFrame]:
    """Parse Restaurant-ACOS JSONL files."""
    raw = cfg["paths"]["raw"]
    parser = ACOSParser()
    out: dict[str, pd.DataFrame] = {}

    for split, key in [("train", "acos_train"), ("dev", "acos_dev"), ("test", "acos_test")]:
        path = raw[key]
        log.info(f"Parsing ACOS {split}: {path}")
        samples = parser.parse_file(path, split=split)
        df = samples_to_dataframe(samples)
        out[f"acos_{split}"] = df
        log.info(f"  -> {len(df)} rows  |  "
                 f"implicit: {df['has_implicit'].sum()}")

    return out


def build_combined_train(*dfs: pd.DataFrame) -> pd.DataFrame:
    """Merge training DataFrames and reset sample IDs."""
    combined = pd.concat(list(dfs), ignore_index=True)
    # Reassign unique IDs
    combined["sample_id"] = [f"combined_train_{i:06d}" for i in range(len(combined))]
    return combined


def print_summary(all_dfs: dict[str, pd.DataFrame]) -> None:
    log.info("\n" + "=" * 60)
    log.info("DATASET SUMMARY")
    log.info("=" * 60)
    total = 0
    for name, df in all_dfs.items():
        n = len(df)
        total += n
        implicit = int(df["has_implicit"].sum()) if "has_implicit" in df.columns else 0
        conflict = int(df["has_conflict"].sum()) if "has_conflict" in df.columns else 0
        log.info(f"  {name:<30s}  {n:>5d} rows  "
                 f"implicit={implicit}  multi-sentiment={conflict}")
    log.info(f"  {'TOTAL':<30s}  {total:>5d} rows")
    log.info("=" * 60)


def main(config_path: str = "config/config.yaml") -> None:
    cfg = load_config(config_path)
    proc = cfg["paths"]["processed"]

    all_dfs: dict[str, pd.DataFrame] = {}

    # ── SemEval ──────────────────────────────────────────────────────────────
    semeval_dfs = parse_semeval(cfg)
    all_dfs.update(semeval_dfs)

    # ── MAMS ─────────────────────────────────────────────────────────────────
    mams_dfs = parse_mams(cfg)
    all_dfs.update(mams_dfs)

    # ── ACOS ─────────────────────────────────────────────────────────────────
    acos_dfs = parse_acos(cfg)
    all_dfs.update(acos_dfs)

    # ── Combined SemEval training set ────────────────────────────────────────
    combined = build_combined_train(
        semeval_dfs["semeval14_train"],
        semeval_dfs["semeval15_train"],
        semeval_dfs["semeval16_train"],
    )
    all_dfs["combined_train"] = combined

    # ── Save all CSVs ─────────────────────────────────────────────────────────
    key_to_proc = {
        "semeval14_train": proc["semeval14_train"],
        "semeval14_test":  proc["semeval14_test"],
        "semeval15_train": proc["semeval15_train"],
        "semeval15_test":  proc["semeval15_test"],
        "semeval16_train": proc["semeval16_train"],
        "semeval16_test":  proc["semeval16_test"],
        "mams_train":      proc["mams_train"],
        "mams_val":        proc["mams_val"],
        "mams_test":       proc["mams_test"],
        "acos_train":      proc["acos_train"],
        "acos_dev":        proc["acos_dev"],
        "acos_test":       proc["acos_test"],
        "combined_train":  proc["combined_train"],
    }

    for key, out_path in key_to_proc.items():
        if key in all_dfs:
            save_csv(all_dfs[key], out_path)
            log.info(f"Saved -> {out_path}")

    print_summary(all_dfs)
    log.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse all raw ABSA datasets to unified CSVs.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
