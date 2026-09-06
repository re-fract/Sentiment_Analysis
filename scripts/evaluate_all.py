"""
scripts/evaluate_all.py — Run evaluation across all trained experiments.

Usage:
    # Evaluate one experiment:
    .venv/Scripts/python scripts/evaluate_all.py --exp T1

    # Evaluate all (after training all):
    .venv/Scripts/python scripts/evaluate_all.py --exp all

    # Only print summary (skip re-evaluating, reads existing JSONs):
    .venv/Scripts/python scripts/evaluate_all.py --summary-only

Outputs:
    outputs/results/{exp_id}_eval.json       — per-experiment, per-test-set metrics
    outputs/results/all_results_summary.csv  — comparison table (for dissertation)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from src.data.splits import build_splits
from src.evaluation.evaluator import evaluate_experiment
from src.training.hyperparams import EXPERIMENTS
from src.utils.io import load_config, load_json, save_json
from src.utils.logging import get_logger

log = get_logger("evaluate_all")


def build_summary_table(output_dir: Path) -> pd.DataFrame:
    """Read all *_eval.json files and produce a comparison table."""
    rows = []
    for json_path in sorted(output_dir.glob("T*_eval.json")):
        data = load_json(json_path)
        exp  = data.get("experiment", json_path.stem)
        model= data.get("model", "")

        for test_name in ["semeval_test", "mams_test", "acos_test"]:
            tdata = data.get(test_name, {})
            if not tdata:
                continue
            rows.append({
                "Experiment":      exp,
                "Model":           model,
                "Test set":        test_name,
                "Aspect macro-F1": tdata.get("aspect",    {}).get("macro_f1",    ""),
                "Aspect EM":       tdata.get("aspect",    {}).get("exact_match", ""),
                "Sent macro-F1":   tdata.get("sentiment", {}).get("macro_f1",    ""),
                "Sent accuracy":   tdata.get("sentiment", {}).get("accuracy",    ""),
                "Pair-F1":         tdata.get("pipeline",  {}).get("pair_f1",     ""),
            })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",          nargs="+", default=["all"],
                        help="Experiment ID(s) or 'all'. E.g. --exp T9 T10")
    parser.add_argument("--config",       default="config/config.yaml")
    parser.add_argument("--gpu",          type=int, default=0)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["paths"]["outputs"]["results"])
    model_dir  = Path(cfg["paths"]["outputs"]["models"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Summary-only mode ─────────────────────────────────────────────────────
    if args.summary_only:
        df = build_summary_table(output_dir)
        if df.empty:
            log.warning("No *_eval.json files found. Run evaluate_all.py first.")
        else:
            csv_path = output_dir / "all_results_summary.csv"
            df.to_csv(csv_path, index=False)
            log.info(f"Summary saved -> {csv_path}")
            print(df.to_string(index=False))
        return

    # ── Device ────────────────────────────────────────────────────────────────
    device = (torch.device(f"cuda:{args.gpu}")
              if torch.cuda.is_available() else torch.device("cpu"))
    log.info(f"Device: {device}")

    # ── Splits ────────────────────────────────────────────────────────────────
    splits = build_splits(cfg, augmented=False)

    # ── Experiments ───────────────────────────────────────────────────────────
    tokens = [t.upper() for t in args.exp]
    if len(tokens) == 1 and tokens[0] == "ALL":
        exp_ids = list(EXPERIMENTS.keys())
    else:
        exp_ids = tokens

    all_results = {}
    for exp_id in exp_ids:
        model_name = EXPERIMENTS[exp_id]["model_name"]
        asp_ckpt   = model_dir / f"{exp_id}_aspect_best.pt"

        if not asp_ckpt.exists():
            log.warning(f"Skipping {exp_id} — checkpoint not found at {asp_ckpt}")
            continue

        results = evaluate_experiment(
            exp_id     = exp_id,
            model_name = model_name,
            splits     = splits,
            model_dir  = str(model_dir),
            output_dir = str(output_dir),
            device     = device,
        )
        all_results[exp_id] = results

    # ── Cross-experiment summary ───────────────────────────────────────────────
    df = build_summary_table(output_dir)
    if not df.empty:
        csv_path = output_dir / "all_results_summary.csv"
        df.to_csv(csv_path, index=False)
        log.info(f"\nSummary table saved -> {csv_path}")
        print("\n" + df.to_string(index=False))

    log.info("Done.")


if __name__ == "__main__":
    main()
