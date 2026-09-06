"""
scripts/run_llm_classifier.py — Run LLM-as-Classifier experiments.

Usage:
    .venv/Scripts/python scripts/run_llm_classifier.py
    .venv/Scripts/python scripts/run_llm_classifier.py --test mams_test --n 100
    .venv/Scripts/python scripts/run_llm_classifier.py --mode few_shot

Outputs:
    outputs/results/llm_{model}_{mode}_{test_name}_results.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from src.data.splits import build_splits
from src.augmentation.llm_client import CerebrasClient, RateLimiter, make_clients
from src.models.llm_classifier import run_llm_classifier
from src.utils.io import load_config, save_json
from src.utils.logging import get_logger

log = get_logger("run_llm_classifier")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="config/config.yaml")
    parser.add_argument("--test",    default="mams_test",
                        choices=["semeval_test", "mams_test", "acos_test"])
    parser.add_argument("--mode",    default="both",
                        choices=["zero_shot", "few_shot", "both"])
    parser.add_argument("--n",       type=int, default=200,
                        help="Number of test samples to classify")
    args = parser.parse_args()

    cfg = load_config(args.config)
    api_key = os.environ.get("CEREBRAS_API_KEY", "")
    if not api_key:
        log.error("Set CEREBRAS_API_KEY in .env or environment.")
        sys.exit(1)

    cfg["cerebras"]["api_key"] = api_key
    generator, _ = make_clients(cfg)

    splits = build_splits(cfg)
    test_df = splits.get(args.test)
    if test_df is None or test_df.empty:
        log.error(f"Test set '{args.test}' is empty or not found.")
        sys.exit(1)

    log.info(f"LLM-as-Classifier on {args.test} (n={args.n})")
    output_dir = Path(cfg["paths"]["outputs"]["results"])
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = ["zero_shot", "few_shot"] if args.mode == "both" else [args.mode]
    model_name = cfg["cerebras"]["generator_model"]

    for mode in modes:
        log.info(f"\n--- {mode.upper()} ---")
        results = run_llm_classifier(
            client    = generator,
            test_df   = test_df,
            mode      = mode,
            model     = model_name,
            n_samples = args.n,
        )
        out_path = output_dir / f"llm_{model_name.replace('-','_')}_{mode}_{args.test}.json"
        save_json(results, str(out_path))
        log.info(f"Saved -> {out_path}")

    log.info("Done.")


if __name__ == "__main__":
    main()
