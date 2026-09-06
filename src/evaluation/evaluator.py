"""
src/evaluation/evaluator.py

Loads trained checkpoints and evaluates them on all test sets.

Produces:
  outputs/results/{exp_id}_eval.json        — full metrics per test set
  outputs/results/all_results_summary.csv   — comparison table across all experiments

Run via:
  .venv/Scripts/python scripts/evaluate_all.py [--exp T1]
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import transformers
transformers.logging.set_verbosity_error()   # suppress LOAD REPORT / weight warnings

from config.taxonomy import COARSE_CATEGORIES, COARSE_TO_ID, SENTIMENT_LABELS
from src.data.dataset import AspectDetectionDataset, SentimentDataset
from src.evaluation.metrics import (
    aspect_metrics, sentiment_metrics, pair_f1, multi_vs_single_aspect_sentiment
)
from src.models.aspect_detector import build_aspect_model
from src.models.sentiment_classifier import build_sentiment_model, ABSAPipeline, SentimentClassifier
from src.models.aspect_detector import AspectDetector
from src.utils.io import save_json, load_json
from src.utils.logging import get_logger

log = get_logger("evaluator")

NUM_COARSE    = len(COARSE_CATEGORIES)
NUM_SENTIMENT = len(SENTIMENT_LABELS)
SENT_TO_STR   = {0: "positive", 1: "negative", 2: "neutral"}


# ── Aspect detection evaluation ───────────────────────────────────────────────

def eval_aspect_detector(
    checkpoint:   str,
    model_name:   str,
    test_df:      pd.DataFrame,
    thresholds:   Dict[str, float],
    device:       torch.device,
    batch_size:   int = 32,
) -> Dict:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset   = AspectDetectionDataset(test_df, tokenizer, max_length=128)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = build_aspect_model(model_name)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.to(device).eval()

    all_probs  = []
    all_labels = []

    supports_tti = hasattr(model.config, "type_vocab_size")
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch  = {k: v.to(device) for k, v in batch.items()
                      if k != "token_type_ids" or supports_tti}
            logits = model(**batch).logits
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    probs_cat  = np.concatenate(all_probs,  axis=0)
    labels_cat = np.concatenate(all_labels, axis=0)

    # Apply per-category thresholds
    thr_vec = np.array([thresholds.get(cat, 0.5) for cat in COARSE_CATEGORIES])
    preds   = (probs_cat >= thr_vec).astype(int)

    return aspect_metrics(labels_cat, preds)


# ── Sentiment evaluation ──────────────────────────────────────────────────────

def eval_sentiment_classifier(
    checkpoint:   str,
    model_name:   str,
    test_df:      pd.DataFrame,
    device:       torch.device,
    batch_size:   int = 32,
) -> Tuple[Dict, List[str], List[str], List[int]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset   = SentimentDataset(test_df, tokenizer, max_length=128)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = build_sentiment_model(model_name)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.to(device).eval()

    all_pred  = []
    all_true  = []
    all_n_asp = []

    supports_tti = hasattr(model.config, "type_vocab_size")
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch  = {k: v.to(device) for k, v in batch.items()
                      if k != "token_type_ids" or supports_tti}
            logits = model(**batch).logits
            preds  = torch.argmax(logits, dim=-1).cpu().numpy()
            all_pred.extend([SENT_TO_STR[p] for p in preds])
            all_true.extend([SENT_TO_STR[int(l)] for l in labels.numpy()])

    # Extract num_aspects per (text, aspect) pair from test_df
    for _, row in test_df.iterrows():
        try:
            cats = ast.literal_eval(row["categories_coarse"])
        except Exception:
            cats = []
        n = len(cats)
        for _ in cats:
            all_n_asp.append(n)

    # Trim to match dataset size (dataset may have dropped rows with invalid labels)
    min_len = min(len(all_pred), len(all_n_asp))
    all_pred  = all_pred[:min_len]
    all_true  = all_true[:min_len]
    all_n_asp = all_n_asp[:min_len]

    metrics = sentiment_metrics(all_true, all_pred)
    metrics["multi_vs_single"] = multi_vs_single_aspect_sentiment(
        all_true, all_pred, all_n_asp)

    return metrics, all_true, all_pred, all_n_asp


# ── End-to-end Pipeline evaluation ───────────────────────────────────────────

def eval_pipeline(
    aspect_ckpt:   str,
    sentiment_ckpt:str,
    model_name:    str,
    test_df:       pd.DataFrame,
    thresholds:    Dict[str, float],
    device:        torch.device,
) -> Dict:
    """
    Evaluate the full pipeline (aspect detection → sentiment classification)
    using Pair-F1.
    """
    detector   = AspectDetector.from_checkpoint(aspect_ckpt, model_name,
                                                thresholds=thresholds, device=device)
    classifier = SentimentClassifier.from_checkpoint(sentiment_ckpt, model_name, device=device)
    pipeline   = ABSAPipeline(detector, classifier)

    gold_pairs: List[List[Tuple[str, str]]] = []
    pred_pairs: List[List[Tuple[str, str]]] = []

    for _, row in test_df.iterrows():
        text = str(row["text"])
        try:
            gold_cats  = ast.literal_eval(row["categories_coarse"])
            gold_sents = ast.literal_eval(row["sentiments"])
        except Exception:
            gold_cats, gold_sents = [], []

        gold = [(c, s) for c, s in zip(gold_cats, gold_sents)]
        gold_pairs.append(gold)

        # Pipeline prediction
        results = pipeline.analyze(text)
        pred = [(r["category"], r["sentiment"]) for r in results]
        pred_pairs.append(pred)

    return pair_f1(gold_pairs, pred_pairs)


# ── Main evaluator ────────────────────────────────────────────────────────────

def evaluate_experiment(
    exp_id:     str,
    model_name: str,
    splits:     Dict[str, pd.DataFrame],
    model_dir:  str,
    output_dir: str,
    device:     torch.device,
) -> Dict:
    """
    Run full evaluation for one experiment across all test sets.
    """
    model_dir  = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    asp_ckpt   = model_dir / f"{exp_id}_aspect_best.pt"
    sent_ckpt  = model_dir / f"{exp_id}_sentiment_best.pt"
    thr_path   = model_dir / f"{exp_id}_thresholds.json"

    if not asp_ckpt.exists():
        log.error(f"Aspect checkpoint not found: {asp_ckpt}")
        return {}
    if not sent_ckpt.exists():
        log.error(f"Sentiment checkpoint not found: {sent_ckpt}")
        return {}

    thresholds = load_json(thr_path) if thr_path.exists() else {c: 0.5 for c in COARSE_CATEGORIES}
    log.info(f"\n[{exp_id}] Evaluating {model_name}")
    log.info(f"  Aspect checkpoint:    {asp_ckpt}")
    log.info(f"  Sentiment checkpoint: {sent_ckpt}")

    results: Dict = {"experiment": exp_id, "model": model_name}

    # Test sets to evaluate on
    test_sets = {
        "semeval_test": splits.get("semeval_test", pd.DataFrame()),
        "mams_test":    splits.get("mams_test",    pd.DataFrame()),
        "acos_test":    splits.get("acos_test",    pd.DataFrame()),
    }

    for test_name, test_df in test_sets.items():
        if test_df.empty:
            log.warning(f"  Skipping {test_name} (empty)")
            continue

        log.info(f"\n  [{test_name}] n={len(test_df)}")

        # Aspect
        asp_m = eval_aspect_detector(
            str(asp_ckpt), model_name, test_df, thresholds, device)
        log.info(f"    Aspect   macro-F1={asp_m['macro_f1']:.4f}  "
                 f"exact={asp_m['exact_match']:.4f}")

        # Sentiment
        sent_m, _, _, _ = eval_sentiment_classifier(
            str(sent_ckpt), model_name, test_df, device)
        log.info(f"    Sentiment macro-F1={sent_m['macro_f1']:.4f}  "
                 f"accuracy={sent_m['accuracy']:.4f}")

        # Pipeline Pair-F1
        pipe_m = eval_pipeline(
            str(asp_ckpt), str(sent_ckpt), model_name, test_df, thresholds, device)
        log.info(f"    Pair-F1={pipe_m['pair_f1']:.4f}  "
                 f"P={pipe_m['pair_precision']:.4f}  R={pipe_m['pair_recall']:.4f}")

        results[test_name] = {
            "aspect":    asp_m,
            "sentiment": sent_m,
            "pipeline":  pipe_m,
        }

    out_path = output_dir / f"{exp_id}_eval.json"
    save_json(results, str(out_path))
    log.info(f"\n  Results saved -> {out_path}")
    return results
