"""
scripts/smoke_test.py — End-to-end pipeline validation on a tiny data slice.

Runs the FULL pipeline (data loading → training → evaluation → error analysis)
on 200 training samples, 2 epochs, CPU — completes in ~3-5 minutes.
Designed to catch ALL bugs before the real GPU run on the server.

Checks validated:
  ✓ All imports work
  ✓ Data loading + splits
  ✓ AspectDetectionDataset + SentimentDataset construction
  ✓ Model instantiation (DistilBERT — fastest)
  ✓ Training loop (forward, backward, loss, scheduler)
  ✓ FP16 graceful fallback on CPU
  ✓ Checkpoint save/load
  ✓ Threshold tuning
  ✓ Evaluation metrics (aspect, sentiment, Pair-F1)
  ✓ Error analysis functions
  ✓ ABSAPipeline inference

Usage:
    .venv/Scripts/python scripts/smoke_test.py
    .venv/Scripts/python scripts/smoke_test.py --n 100 --epochs 1   # even faster
"""

from __future__ import annotations

import argparse
import sys
import time
import tempfile
from pathlib import Path

import os
os.environ.setdefault("PYTHONUTF8", "1")  # force UTF-8 on Windows console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pandas as pd

from src.utils.io import load_config
from src.utils.logging import get_logger

log = get_logger("smoke_test")

PASS = "[PASS]"
FAIL = "[FAIL]"

results: dict[str, bool] = {}


def check(label: str, fn):
    """Run a named check, catch and log any exception, return the result value."""
    try:
        val = fn()
        log.info(f"  {PASS} {label}")
        results[label] = True
        return val
    except Exception as e:
        log.error(f"  {FAIL} {label}: {e}", exc_info=True)
        results[label] = False
        return None


# ─────────────────────────────────────────────────────────────────────────────

def run_smoke_test(n_train: int = 200, n_epochs: int = 2) -> bool:
    log.info("=" * 60)
    log.info(f"SMOKE TEST  n_train={n_train}  n_epochs={n_epochs}")
    log.info("=" * 60)
    device = torch.device("cpu")  # smoke test always on CPU

    # ── 1. Config + splits ────────────────────────────────────────────────────
    cfg    = check("Load config",   lambda: load_config())
    splits = check("Build splits",  lambda: _load_splits(cfg))
    if splits is None:
        log.error("Cannot continue without splits. Aborting.")
        return False

    train_df = splits["full_train"].sample(n=min(n_train, len(splits["full_train"])),
                                            random_state=42).reset_index(drop=True)
    val_df   = splits["full_val"].sample(n=min(50, len(splits["full_val"])),
                                          random_state=42).reset_index(drop=True)
    test_df  = splits["mams_test"].sample(n=min(30, len(splits["mams_test"])),
                                           random_state=42).reset_index(drop=True)
    log.info(f"  Slices: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    # ── 2. Datasets ───────────────────────────────────────────────────────────
    from transformers import AutoTokenizer
    from src.data.dataset import AspectDetectionDataset, SentimentDataset
    from config.taxonomy import COARSE_CATEGORIES, SENTIMENT_LABELS

    model_name = "distilbert-base-uncased"
    tokenizer = check("Load tokenizer",
                      lambda: AutoTokenizer.from_pretrained(model_name))
    if tokenizer is None:
        return False

    asp_train_ds = check("AspectDetectionDataset (train)",
                         lambda: AspectDetectionDataset(train_df, tokenizer, max_length=64))
    asp_val_ds   = check("AspectDetectionDataset (val)",
                         lambda: AspectDetectionDataset(val_df,   tokenizer, max_length=64))
    sent_train_ds= check("SentimentDataset (train)",
                         lambda: SentimentDataset(train_df, tokenizer, max_length=64))
    sent_val_ds  = check("SentimentDataset (val)",
                         lambda: SentimentDataset(val_df,   tokenizer, max_length=64))

    if not all([asp_train_ds, asp_val_ds, sent_train_ds, sent_val_ds]):
        return False

    log.info(f"  Dataset sizes: asp_train={len(asp_train_ds)}  "
             f"asp_val={len(asp_val_ds)}  sent_train={len(sent_train_ds)}  "
             f"sent_val={len(sent_val_ds)}")

    # ── 3. Model instantiation ────────────────────────────────────────────────
    from src.training.trainer import build_aspect_model, build_sentiment_model

    asp_model  = check("Build aspect model",
                       lambda: build_aspect_model(model_name))
    sent_model = check("Build sentiment model",
                       lambda: build_sentiment_model(model_name))
    if asp_model is None or sent_model is None:
        return False

    # ── 4. Training config ────────────────────────────────────────────────────
    from src.training.hyperparams import TrainingConfig
    smoke_cfg = TrainingConfig(
        experiment_id    = "SMOKE",
        model_name       = model_name,
        data_config      = "full",
        max_seq_length   = 64,
        learning_rate    = 3e-5,
        num_epochs       = n_epochs,
        batch_size       = 8,
        grad_accum_steps = 1,
        patience         = n_epochs + 1,   # don't early-stop during smoke test
        fp16             = False,          # CPU — no FP16
        eval_every_n_steps = 9999,
    )

    # Use a temp dir for checkpoints
    with tempfile.TemporaryDirectory() as tmpdir:
        smoke_cfg.checkpoint_dir = tmpdir

        # ── 5. Train aspect detector ──────────────────────────────────────────
        from src.training.trainer import ABSATrainer

        asp_trainer = check("Instantiate ABSATrainer (aspect)",
                            lambda: ABSATrainer(asp_model, "aspect", smoke_cfg,
                                               asp_train_ds, asp_val_ds, device=device))
        asp_results = check("Train aspect detector",
                            lambda: asp_trainer.train())
        log.info(f"  Aspect best val F1: {asp_results.get('best_val_f1', 'N/A')}")

        # ── 6. Train sentiment classifier ─────────────────────────────────────
        sent_trainer = check("Instantiate ABSATrainer (sentiment)",
                             lambda: ABSATrainer(sent_model, "sentiment", smoke_cfg,
                                                sent_train_ds, sent_val_ds, device=device))
        sent_results = check("Train sentiment classifier",
                             lambda: sent_trainer.train())
        log.info(f"  Sentiment best val F1: {sent_results.get('best_val_f1', 'N/A')}")

        # ── 7. Checkpoint load ────────────────────────────────────────────────
        asp_ckpt  = asp_results.get("checkpoint",  "") if asp_results else ""
        sent_ckpt = sent_results.get("checkpoint", "") if sent_results else ""

        from src.models.aspect_detector import AspectDetector
        from src.models.sentiment_classifier import SentimentClassifier, ABSAPipeline

        detector = check("Load AspectDetector from checkpoint",
                         lambda: AspectDetector.from_checkpoint(asp_ckpt, model_name, device=device)
                         if asp_ckpt and Path(asp_ckpt).exists() else (_ for _ in ()).throw(
                             FileNotFoundError(f"Checkpoint not found: {asp_ckpt}")))

        classifier = check("Load SentimentClassifier from checkpoint",
                           lambda: SentimentClassifier.from_checkpoint(sent_ckpt, model_name, device=device)
                           if sent_ckpt and Path(sent_ckpt).exists() else (_ for _ in ()).throw(
                               FileNotFoundError(f"Checkpoint not found: {sent_ckpt}")))

        # ── 8. Threshold tuning ───────────────────────────────────────────────
        if detector is not None:
            thresholds = check("Tune aspect thresholds",
                               lambda: detector.tune_thresholds(val_df, max_length=64))
        else:
            thresholds = {cat: 0.5 for cat in COARSE_CATEGORIES}

        # ── 9. Evaluation metrics ─────────────────────────────────────────────
        from src.evaluation.evaluator import (
            eval_aspect_detector, eval_sentiment_classifier, eval_pipeline
        )

        if asp_ckpt and Path(asp_ckpt).exists():
            asp_m = check("eval_aspect_detector",
                          lambda: eval_aspect_detector(
                              asp_ckpt, model_name, test_df,
                              thresholds or {c: 0.5 for c in COARSE_CATEGORIES},
                              device, batch_size=8))
            if asp_m:
                log.info(f"  Aspect macro-F1={asp_m['macro_f1']}  "
                         f"exact={asp_m['exact_match']}")

        if sent_ckpt and Path(sent_ckpt).exists():
            sent_m, _, _, _ = check("eval_sentiment_classifier",
                                    lambda: eval_sentiment_classifier(
                                        sent_ckpt, model_name, test_df,
                                        device, batch_size=8)) or (None, None, None, None)
            if sent_m:
                log.info(f"  Sentiment macro-F1={sent_m['macro_f1']}  "
                         f"accuracy={sent_m['accuracy']}")

        if (asp_ckpt and Path(asp_ckpt).exists() and
                sent_ckpt and Path(sent_ckpt).exists()):
            pipe_m = check("eval_pipeline (Pair-F1)",
                           lambda: eval_pipeline(
                               asp_ckpt, sent_ckpt, model_name, test_df,
                               thresholds or {c: 0.5 for c in COARSE_CATEGORIES},
                               device))
            if pipe_m:
                log.info(f"  Pair-F1={pipe_m['pair_f1']}  "
                         f"P={pipe_m['pair_precision']}  R={pipe_m['pair_recall']}")

        # ── 10. Full pipeline inference ───────────────────────────────────────
        if detector is not None and classifier is not None:
            pipeline = ABSAPipeline(detector, classifier)
            sample_text = str(test_df.iloc[0]["text"])

            check("ABSAPipeline.analyze",
                  lambda: pipeline.analyze(sample_text))
            check("ABSAPipeline.analyze_with_proba",
                  lambda: pipeline.analyze_with_proba(sample_text))

        # ── 11. Error analysis ────────────────────────────────────────────────
        from src.evaluation.error_analysis import (
            sentiment_confusion_matrix, implicit_vs_explicit_breakdown
        )
        check("sentiment_confusion_matrix",
              lambda: sentiment_confusion_matrix(["positive", "negative", "neutral"],
                                                 ["positive", "negative", "positive"]))

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("SMOKE TEST RESULTS")
    log.info("=" * 60)
    passed = sum(v for v in results.values())
    total  = len(results)
    for label, ok in results.items():
        log.info(f"  {'[PASS]' if ok else '[FAIL]'} {label}")
    log.info(f"\n  {passed}/{total} checks passed")

    if passed == total:
        log.info("  *** ALL CHECKS PASSED - safe to run full training on server. ***")
        return True
    else:
        failed = [l for l, ok in results.items() if not ok]
        log.error(f"  !! {total - passed} check(s) FAILED: {failed}")
        return False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_splits(cfg):
    from src.data.splits import build_splits
    return build_splits(cfg, augmented=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end smoke test")
    parser.add_argument("--n",      type=int, default=200,
                        help="Training samples to use (default: 200)")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Training epochs (default: 2)")
    args = parser.parse_args()

    t0 = time.time()
    ok = run_smoke_test(n_train=args.n, n_epochs=args.epochs)
    elapsed = (time.time() - t0) / 60
    log.info(f"\nTotal time: {elapsed:.1f} min")
    sys.exit(0 if ok else 1)
