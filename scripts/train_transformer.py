"""
scripts/train_transformer.py — Run transformer training experiments.

Usage:
    # Run a single experiment
    python scripts/train_transformer.py --exp T3

    # Run multiple experiments in one command
    python scripts/train_transformer.py --exp T7 T8

    # Run all experiments
    python scripts/train_transformer.py --exp all

Experiments:
    T1: BERT-base       + SemEval only       (aspect + sentiment)
    T2: BERT-base       + SemEval + Augmented
    T3: RoBERTa-base    + SemEval only
    T4: RoBERTa-base    + SemEval + Augmented
    T5: DistilBERT-base + SemEval only
    T6: DistilBERT-base + SemEval + Augmented
    T7: RoBERTa-large   + SemEval only       [improved: text cleaning + label smoothing]
    T8: RoBERTa-large   + SemEval + Augmented [improved]

Each experiment trains two models (aspect detector + sentiment classifier)
and saves:
    outputs/models/{exp_id}_aspect_best.pt
    outputs/models/{exp_id}_sentiment_best.pt
    outputs/models/{exp_id}_aspect_log.jsonl
    outputs/models/{exp_id}_sentiment_log.jsonl
    outputs/results/{exp_id}_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data.splits import build_splits
from src.models.aspect_detector import train_aspect_detector
from src.models.sentiment_classifier import train_sentiment_classifier
from src.training.hyperparams import get_config, EXPERIMENTS
from src.utils.io import load_config, save_json
from src.utils.logging import get_logger

log = get_logger("train_transformer")

# ── Hugging Face model cache: store on server's fast disk ────────────────────
import os
os.environ.setdefault("HF_HOME", str(Path("outputs/hf_cache").resolve()))
os.environ.setdefault("TRANSFORMERS_CACHE", str(Path("outputs/hf_cache").resolve()))


def run_experiment(
    exp_id:    str,
    splits:    dict,
    aug_splits:dict | None,
    cfg_yaml:  dict,
    device:    torch.device,
) -> dict:
    """Train both models for one experiment and return results dict."""
    cfg = get_config(exp_id)

    # Select train/val splits
    use_aug = cfg.data_config == "augmented"
    if use_aug and aug_splits:
        train_df = aug_splits["augmented_train"]
        val_df   = aug_splits["full_val"]
    else:
        train_df = splits["full_train"]
        val_df   = splits["full_val"]

    log.info(f"\n{'='*60}")
    log.info(f"EXPERIMENT {exp_id}: {cfg.model_name}  data={cfg.data_config}")
    log.info(f"train={len(train_df)}  val={len(val_df)}")
    log.info(f"{'='*60}")

    # Override checkpoint dir from yaml
    cfg.checkpoint_dir = cfg_yaml["paths"]["outputs"]["models"]
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    results = {"experiment": exp_id, "model": cfg.model_name,
               "data_config": cfg.data_config}

    # ── Train aspect detector ─────────────────────────────────────────────────
    log.info(f"\n[{exp_id}] Training aspect detector...")
    asp_results = train_aspect_detector(cfg, train_df, val_df, device=device)
    results["aspect"] = asp_results
    log.info(f"[{exp_id}] Aspect best val F1: {asp_results['best_val_f1']:.4f}")

    # ── Tune per-category thresholds on val set ───────────────────────────────
    log.info(f"\n[{exp_id}] Tuning aspect thresholds on val set...")
    from src.models.aspect_detector import AspectDetector
    detector = AspectDetector(cfg.model_name, device=device)
    detector.load_checkpoint(asp_results["checkpoint"])
    tuned_thresholds = detector.tune_thresholds(val_df, max_length=cfg.max_seq_length)
    results["aspect"]["tuned_thresholds"] = tuned_thresholds

    thr_path = Path(cfg.checkpoint_dir) / f"{exp_id}_thresholds.json"
    with thr_path.open("w") as fh:
        json.dump(tuned_thresholds, fh, indent=2)
    log.info(f"[{exp_id}] Thresholds saved -> {thr_path}")

    # ── Train sentiment classifier ────────────────────────────────────────────
    log.info(f"\n[{exp_id}] Training sentiment classifier...")
    sent_results = train_sentiment_classifier(cfg, train_df, val_df, device=device)
    results["sentiment"] = sent_results
    log.info(f"[{exp_id}] Sentiment best val F1: {sent_results['best_val_f1']:.4f}")

    # ── Save combined results ─────────────────────────────────────────────────
    out_path = Path(cfg_yaml["paths"]["outputs"]["results"]) / f"{exp_id}_results.json"
    save_json(results, str(out_path))
    log.info(f"[{exp_id}] Results saved -> {out_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train transformer ABSA models.")
    parser.add_argument("--exp", nargs="+", default=["all"],
                        help="Experiment ID(s): T1 T2 ... or 'all'. Example: --exp T5 T6")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--gpu",    type=int, default=0,
                        help="CUDA device index (default: 0)")
    args = parser.parse_args()

    cfg_yaml = load_config(args.config)

    # Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        log.info(f"GPU: {torch.cuda.get_device_name(device)}  "
                 f"({torch.cuda.get_device_properties(device).total_memory // 1024**3}GB)")
    else:
        device = torch.device("cpu")
        log.warning("No GPU detected — training on CPU (very slow).")

    # Splits
    log.info("Loading splits...")
    splits      = build_splits(cfg_yaml, augmented=False)
    aug_splits  = None
    aug_path    = cfg_yaml["paths"]["augmented"]["filtered"]
    if Path(aug_path).exists():
        aug_splits = build_splits(cfg_yaml, augmented=True)
        log.info("Augmented splits loaded.")
    else:
        log.warning(f"No augmented data found at {aug_path}. "
                    "Experiments T2/T4/T6 will fall back to full_train.")

    # Select experiments
    tokens = [t.upper() for t in args.exp]
    if len(tokens) == 1 and tokens[0] == "ALL":
        exp_ids = list(EXPERIMENTS.keys())
    else:
        unknown = [t for t in tokens if t not in EXPERIMENTS]
        if unknown:
            log.error(f"Unknown experiment(s): {unknown}. Valid: {list(EXPERIMENTS.keys())}")
            sys.exit(1)
        exp_ids = tokens

    log.info(f"Running experiments: {exp_ids}")

    # Run
    all_results = {}
    for exp_id in exp_ids:
        try:
            results = run_experiment(exp_id, splits, aug_splits, cfg_yaml, device)
            all_results[exp_id] = results
        except Exception as e:
            log.error(f"Experiment {exp_id} failed: {e}", exc_info=True)

    # Summary table
    log.info("\n" + "="*60)
    log.info("EXPERIMENT SUMMARY")
    log.info("="*60)
    log.info(f"{'ID':<4} {'Model':<26} {'Data':<12} {'AspF1':>7} {'SentF1':>7}")
    log.info("-"*60)
    for exp_id, res in all_results.items():
        asp_f1  = res.get("aspect",    {}).get("best_val_f1", 0)
        sent_f1 = res.get("sentiment", {}).get("best_val_f1", 0)
        log.info(f"{exp_id:<4} {res['model']:<26} {res['data_config']:<12} "
                 f"{asp_f1:>7.4f} {sent_f1:>7.4f}")

    log.info("Done.")


if __name__ == "__main__":
    main()
