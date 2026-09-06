"""
scripts/run_augmentation.py — Master augmentation orchestrator.

Steps (run in order, or specify --step to run one at a time):

  Step 1: filter-yelp   Filter the raw Yelp dataset to restaurant reviews.
  Step 2: generate      Generate synthetic reviews for imbalanced buckets.
  Step 3: label         Label sampled Yelp reviews with ABSA annotations.
  Step 4: filter        Run quality filter + cross-model consistency check.
  Step 5: verify        Launch the 100-sample manual verification CLI.

Usage examples:
    # Full pipeline (all steps):
    .venv/Scripts/python scripts/run_augmentation.py --api-key YOUR_KEY

    # Single step:
    .venv/Scripts/python scripts/run_augmentation.py --api-key YOUR_KEY --step generate
    .venv/Scripts/python scripts/run_augmentation.py --step verify   # no API key needed

    # Skip consistency check (dry run / no API key):
    .venv/Scripts/python scripts/run_augmentation.py --step filter --skip-consistency

Environment variable alternative to --api-key:
    set CEREBRAS_API_KEY=YOUR_KEY
    .venv/Scripts/python scripts/run_augmentation.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env file (CEREBRAS_API_KEY etc.) before anything else
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass  # dotenv not installed; fall back to env vars / --api-key flag

from src.utils.io import load_config
from src.utils.logging import get_logger

log = get_logger("run_augmentation")

VALID_STEPS = ["filter-yelp", "generate", "label", "filter", "verify", "all"]


def get_api_key(args_key: str | None, cfg: dict) -> str:
    key = (
        args_key
        or os.environ.get("GROQ_API_KEY", "")
        or os.environ.get("CEREBRAS_API_KEY", "")
        or cfg.get("cerebras", {}).get("api_key", "")
    )
    if not key or key == "YOUR_CEREBRAS_API_KEY_HERE":
        raise ValueError(
            "API key not set. Set GROQ_API_KEY or CEREBRAS_API_KEY in .env, or use --api-key YOUR_KEY"
        )
    return key


def step_filter_yelp(cfg: dict) -> None:
    log.info("=" * 60)
    log.info("STEP 1: Filtering Yelp dataset")
    log.info("=" * 60)
    from scripts.filter_yelp import filter_yelp
    filter_yelp(cfg)


def step_generate(cfg: dict, api_key: str, out_file: str | None = None) -> None:
    log.info("=" * 60)
    log.info("STEP 2: Generating synthetic reviews (Mode A)")
    log.info("=" * 60)
    from src.augmentation.llm_client import make_clients
    from src.augmentation.generator import run_generate_mode

    # Load a few real training examples as seeds for the few-shot prompt
    import pandas as pd, ast, random
    seed_data = []
    try:
        df = pd.read_csv("data/processed/acos_train.csv")
        for _, row in df.sample(min(20, len(df)), random_state=42).iterrows():
            cats  = ast.literal_eval(row["categories_fine"])
            sents = ast.literal_eval(row["sentiments"])
            terms = ast.literal_eval(row["aspect_terms"])
            impl  = ast.literal_eval(row["is_implicit_flags"])
            seed_data.append({
                "text": row["text"],
                "labels": [
                    {"aspect_term": t if str(t) != "None" else None,
                     "category": c, "sentiment": s, "is_implicit": bool(im)}
                    for c, s, t, im in zip(cats, sents, terms, impl)
                ]
            })
        log.info(f"  Loaded {len(seed_data)} seed examples from ACOS train")
    except Exception as e:
        log.warning(f"  Could not load seed examples: {e}")

    # Patch API key into cfg
    cfg = dict(cfg)
    cfg["cerebras"] = dict(cfg["cerebras"])
    cfg["cerebras"]["api_key"] = api_key

    out_path = out_file or cfg.get("paths", {}).get("augmented", {}).get(
        "generated_raw", "data/augmented/synthetic_generated_raw.jsonl"
    )

    generator, _ = make_clients(cfg)
    run_generate_mode(
        generator    = generator,
        targets_path = "data/augmented/augmentation_targets.json",
        out_path     = out_path,
        seed_data    = seed_data,
    )


def step_label(cfg: dict, api_key: str, max_to_label: int = 2000, batch_size: int = 2) -> None:
    log.info("=" * 60)
    log.info(f"STEP 3: Labeling Yelp reviews (Mode B, batch_size={batch_size})")
    log.info("=" * 60)
    from src.augmentation.llm_client import make_clients
    from src.augmentation.generator import run_label_mode

    cfg = dict(cfg)
    cfg["cerebras"] = dict(cfg["cerebras"])
    cfg["cerebras"]["api_key"] = api_key

    generator, _ = make_clients(cfg)
    run_label_mode(
        generator         = generator,
        yelp_sampled_path = "data/raw/yelp_restaurants_sampled.jsonl",
        max_to_label      = max_to_label,
        batch_size        = batch_size,
    )


def step_filter(cfg: dict, api_key: str | None, skip_consistency: bool) -> None:
    log.info("=" * 60)
    log.info("STEP 4: Quality filtering")
    log.info("=" * 60)
    from src.augmentation.quality_filter import run_filter

    validator = None
    if not skip_consistency and api_key:
        from src.augmentation.llm_client import make_clients
        cfg = dict(cfg)
        cfg["cerebras"] = dict(cfg["cerebras"])
        cfg["cerebras"]["api_key"] = api_key
        _, validator = make_clients(cfg)

    run_filter(validator=validator, cfg=cfg, skip_consistency=skip_consistency)


def step_verify() -> None:
    log.info("=" * 60)
    log.info("STEP 5: Manual verification")
    log.info("=" * 60)
    from src.augmentation.verify import verify_samples
    verify_samples(
        input_csv  = "data/augmented/synthetic_filtered.csv",
        output_csv = "data/augmented/gold_verified_100.csv",
        n          = 100,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ABSA augmentation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--step",
        choices=VALID_STEPS,
        default="all",
        help="Which step to run (default: all)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (Groq or Cerebras, overrides config and env var)",
    )
    parser.add_argument(
        "--skip-consistency",
        action="store_true",
        help="Skip cross-model consistency check in the filter step",
    )
    parser.add_argument(
        "--max-label",
        type=int,
        default=2000,
        help="Max Yelp reviews to label in the label step (default: 2000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Number of reviews to label per LLM prompt (default: 2)",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default=None,
        help="Custom output file path for generated or labeled reviews",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    from src.utils.io import load_config
    cfg = load_config(args.config)

    # Resolve API key (only needed for API steps)
    needs_api = args.step in ("generate", "label", "filter", "all")
    api_key   = None
    if needs_api and not args.skip_consistency:
        try:
            api_key = get_api_key(args.api_key, cfg)
            log.info(f"API key loaded: {'*' * (len(api_key)-4)}{api_key[-4:]}")
        except ValueError as e:
            log.error(str(e))
            if args.step in ("generate", "label"):
                sys.exit(1)
            else:
                log.warning("Continuing without API key — consistency check will be skipped.")

    step = args.step

    if step in ("filter-yelp", "all"):
        step_filter_yelp(cfg)

    if step in ("generate", "all"):
        if not api_key:
            log.error("API key required for generate step. Aborting.")
            sys.exit(1)
        step_generate(cfg, api_key, out_file=args.out_file)

    if step in ("label", "all"):
        if not api_key:
            log.error("API key required for label step. Aborting.")
            sys.exit(1)
        step_label(cfg, api_key, max_to_label=args.max_label, batch_size=args.batch_size)

    if step in ("filter", "all"):
        step_filter(cfg, api_key, skip_consistency=args.skip_consistency)

    if step in ("verify", "all"):
        step_verify()

    log.info("Augmentation pipeline complete.")


if __name__ == "__main__":
    main()
