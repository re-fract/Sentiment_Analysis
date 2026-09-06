"""
scripts/train_baselines.py — Train and evaluate all TF-IDF baseline models.

Usage:
    .venv/Scripts/python scripts/train_baselines.py
    .venv/Scripts/python scripts/train_baselines.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.splits import build_splits
from src.models.baseline import run_baselines
from src.utils.io import load_config
from src.utils.logging import get_logger

log = get_logger("train_baselines")


def main(config_path: str = "config/config.yaml") -> None:
    cfg    = load_config(config_path)
    splits = build_splits(cfg, augmented=False)
    log.info(f"Splits loaded. Running baselines...")
    run_baselines(splits, output_dir=cfg["paths"]["outputs"]["results"])
    log.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TF-IDF baseline models.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
