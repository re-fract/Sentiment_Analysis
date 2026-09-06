"""
scripts/train_joint.py — Joint multi-task training for T9 and T10.

Usage:
    python scripts/train_joint.py --exp T9
    python scripts/train_joint.py --exp T9 T10
    python scripts/train_joint.py --exp T9 --smoke   # 200-sample sanity check

Experiments:
    T9  : RoBERTa-base  joint (aspect + sentiment heads)        semeval_only
    T10 : RoBERTa-base  joint (aspect + sentiment + implicit)   full_train

Output per experiment:
    outputs/models/{exp_id}_aspect_best.pt     ← compatible with AspectDetector
    outputs/models/{exp_id}_sentiment_best.pt  ← compatible with SentimentClassifier
    outputs/models/{exp_id}_thresholds.json
    outputs/results/{exp_id}_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config.taxonomy import COARSE_CATEGORIES
from src.data.splits import build_splits
from src.data.dataset import AspectDetectionDataset, SentimentDataset
from src.data.joint_dataset import ImplicitDataset
from src.models.joint_model import build_joint_model
from src.models.aspect_detector import AspectDetector
from src.training.joint_trainer import JointTrainer
from src.training.hyperparams import TrainingConfig, EXPERIMENTS, get_config
from src.utils.io import load_config, save_json
from src.utils.logging import get_logger

import transformers
transformers.logging.set_verbosity_error()

log = get_logger("train_joint")

# Joint experiments only
JOINT_EXPERIMENTS = {k: v for k, v in EXPERIMENTS.items() if k in ("T9", "T10")}


# ── Data selection ────────────────────────────────────────────────────────────

def _select_splits(exp_id: str, splits: dict, cfg: TrainingConfig) -> tuple:
    """Return (train_df, val_df) based on cfg.data_config."""
    dc = cfg.data_config
    if dc == "semeval_only":
        return splits["semeval_train"], splits["semeval_val"]
    elif dc == "augmented":
        return splits.get("augmented_train", splits["full_train"]), splits["full_val"]
    elif dc == "full_train":
        return splits["full_train"], splits["full_val"]
    else:
        raise ValueError(f"Unknown data_config: {dc}")


# ── Experiment runner ─────────────────────────────────────────────────────────

def run_joint_experiment(
    exp_id:   str,
    splits:   dict,
    cfg_yaml: dict,
    device:   torch.device,
    smoke:    bool = False,
) -> dict:

    cfg = get_config(exp_id)
    log.info(f"\n{'='*60}")
    log.info(f"JOINT EXPERIMENT {exp_id}: {cfg.model_name}  "
             f"data={cfg.data_config}  implicit={cfg.with_implicit}")
    log.info(f"{'='*60}\n")

    train_df, val_df = _select_splits(exp_id, splits, cfg)

    if smoke:
        train_df = train_df.sample(n=min(200, len(train_df)), random_state=42).reset_index(drop=True)
        val_df   = val_df.sample(n=min(50,  len(val_df)),   random_state=42).reset_index(drop=True)
        cfg.num_epochs = 1

    # For sentiment training, expand the dataframe (one row per aspect pair)
    # Re-use the SentimentDataset which handles this internally
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # ── Aspect DataLoaders ────────────────────────────────────────────────────
    asp_train_ds  = AspectDetectionDataset(train_df, tokenizer, max_length=cfg.max_seq_length)
    asp_val_ds    = AspectDetectionDataset(val_df,   tokenizer, max_length=cfg.max_seq_length)

    asp_train_loader = DataLoader(asp_train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=0)
    asp_val_loader   = DataLoader(asp_val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    log.info(f"[{exp_id}] Aspect  train={len(asp_train_ds)}  val={len(asp_val_ds)}")

    # ── Sentiment DataLoaders ─────────────────────────────────────────────────
    sent_train_ds  = SentimentDataset(train_df, tokenizer, max_length=cfg.max_seq_length)
    sent_val_ds    = SentimentDataset(val_df,   tokenizer, max_length=cfg.max_seq_length)

    sent_train_loader = DataLoader(sent_train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=0)
    sent_val_loader   = DataLoader(sent_val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    log.info(f"[{exp_id}] Sentiment train={len(sent_train_ds)}  val={len(sent_val_ds)}")

    # ── Implicit DataLoader (T10 only, ACOS subset) ───────────────────────────
    imp_loader = None
    if cfg.with_implicit:
        # Only ACOS has meaningful is_implicit flags; all others have False.
        # We train on full_train so MAMS/SemEval still provide useful negatives.
        imp_train_ds   = ImplicitDataset(train_df, tokenizer, max_length=cfg.max_seq_length)
        imp_loader     = DataLoader(imp_train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
        log.info(f"[{exp_id}] Implicit train={len(imp_train_ds)}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_joint_model(cfg.model_name, with_implicit=cfg.with_implicit)

    # ── Train ────────────────────────────────────────────────────────────────
    trainer = JointTrainer(
        model=model,
        cfg=cfg,
        aspect_loader=asp_train_loader,
        sentiment_loader=sent_train_loader,
        asp_val_loader=asp_val_loader,
        sent_val_loader=sent_val_loader,
        device=device,
        implicit_loader=imp_loader,
    )
    results = trainer.train()

    # ── Tune aspect thresholds ────────────────────────────────────────────────
    log.info(f"\n[{exp_id}] Tuning aspect thresholds on val set...")
    asp_ckpt = str(Path(cfg.checkpoint_dir) / f"{exp_id}_aspect_best.pt")
    detector = AspectDetector.from_checkpoint(asp_ckpt, cfg.model_name, device=device)
    tuned_thresholds = detector.tune_thresholds(val_df, max_length=cfg.max_seq_length)
    thr_path = Path(cfg.checkpoint_dir) / f"{exp_id}_thresholds.json"
    with open(thr_path, "w") as fh:
        json.dump(tuned_thresholds, fh, indent=2)
    log.info(f"[{exp_id}] Thresholds saved -> {thr_path}")

    # ── Save results ──────────────────────────────────────────────────────────
    results["model"]       = cfg.model_name
    results["data_config"] = cfg.data_config
    results["with_implicit"] = cfg.with_implicit
    out_path = Path(cfg_yaml["paths"]["outputs"]["results"]) / f"{exp_id}_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, str(out_path))
    log.info(f"[{exp_id}] Results saved -> {out_path}")

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train joint ABSA models (T9/T10).")
    parser.add_argument("--exp",    nargs="+", default=["T9", "T10"],
                        help="Experiment ID(s): T9 T10")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--gpu",    type=int, default=0)
    parser.add_argument("--smoke",  action="store_true",
                        help="Smoke test: tiny data + 1 epoch")
    args = parser.parse_args()

    cfg_yaml = load_config(args.config)

    # Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        log.info(f"GPU: {torch.cuda.get_device_name(device)}  "
                 f"({torch.cuda.get_device_properties(device).total_memory // 1024**3}GB)")
    else:
        device = torch.device("cpu")
        log.warning("No GPU — running on CPU (slow).")

    # Validate experiment IDs
    tokens = [t.upper() for t in args.exp]
    valid  = list(JOINT_EXPERIMENTS.keys())
    bad    = [t for t in tokens if t not in valid]
    if bad:
        log.error(f"Unknown joint experiments: {bad}. Valid: {valid}")
        sys.exit(1)

    # Load splits once (shared across experiments)
    log.info("Loading splits...")
    splits = build_splits(cfg_yaml, augmented=False)

    # Run
    all_results = {}
    for exp_id in tokens:
        try:
            r = run_joint_experiment(exp_id, splits, cfg_yaml, device, smoke=args.smoke)
            all_results[exp_id] = r
        except Exception as e:
            log.error(f"Experiment {exp_id} failed: {e}", exc_info=True)

    # Summary
    log.info("\n" + "="*60)
    log.info("JOINT EXPERIMENT SUMMARY")
    log.info("="*60)
    log.info(f"{'ID':<4} {'Model':<20} {'Data':<12} {'AspF1':>7} {'SentF1':>7} {'CombF1':>7}")
    log.info("-"*60)
    for exp_id, res in all_results.items():
        log.info(
            f"{exp_id:<4} {res.get('model',''):<20} {res.get('data_config',''):<12} "
            f"{res.get('best_asp_f1', 0):>7.4f} "
            f"{res.get('best_sent_f1', 0):>7.4f} "
            f"{res.get('best_combined_f1', 0):>7.4f}"
        )
    log.info("Done.")


if __name__ == "__main__":
    main()
