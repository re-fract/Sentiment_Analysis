"""
src/evaluation/metrics.py

Core evaluation metrics for ABSA.

Key metrics used in the dissertation:

Aspect Detection (multi-label):
  - Macro-F1 over coarse categories
  - Per-category Precision / Recall / F1
  - Exact Match (all categories predicted correctly)

Sentiment Classification (single-label per aspect):
  - Macro-F1 over {positive, negative, neutral}
  - Accuracy
  - Per-class P/R/F1

Pipeline (end-to-end):
  - Pair-F1: the gold standard for ABSA evaluation
    A prediction is correct only if BOTH aspect category and sentiment match.

All functions take numpy arrays or lists and return plain dicts.
"""

from __future__ import annotations

import ast
from typing import List, Dict, Tuple

import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    hamming_loss, classification_report,
)

from config.taxonomy import COARSE_CATEGORIES, SENTIMENT_LABELS


# ── Aspect detection metrics ──────────────────────────────────────────────────

def aspect_metrics(
    y_true: np.ndarray,   # shape (N, num_coarse), binary
    y_pred: np.ndarray,   # shape (N, num_coarse), binary
) -> Dict:
    """Compute aspect detection metrics."""
    macro_f1  = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    micro_f1  = f1_score(y_true, y_pred, average="micro",  zero_division=0)
    macro_p   = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r   = recall_score(y_true, y_pred, average="macro",    zero_division=0)
    hamming   = hamming_loss(y_true, y_pred)

    # Exact match: all categories correct for a sample
    exact_match = float((y_true == y_pred).all(axis=1).mean())

    # Per-category
    per_cat_f1 = f1_score(y_true, y_pred, average=None, zero_division=0,
                          labels=list(range(len(COARSE_CATEGORIES))))
    per_cat_p  = precision_score(y_true, y_pred, average=None, zero_division=0,
                                 labels=list(range(len(COARSE_CATEGORIES))))
    per_cat_r  = recall_score(y_true, y_pred, average=None, zero_division=0,
                               labels=list(range(len(COARSE_CATEGORIES))))

    per_category = {}
    for i, cat in enumerate(COARSE_CATEGORIES):
        per_category[cat] = {
            "precision": round(float(per_cat_p[i]), 4),
            "recall":    round(float(per_cat_r[i]), 4),
            "f1":        round(float(per_cat_f1[i]), 4),
            "support":   int(y_true[:, i].sum()),
        }

    return {
        "macro_f1":    round(float(macro_f1), 4),
        "micro_f1":    round(float(micro_f1), 4),
        "macro_p":     round(float(macro_p), 4),
        "macro_r":     round(float(macro_r), 4),
        "hamming":     round(float(hamming), 4),
        "exact_match": round(exact_match, 4),
        "per_category":per_category,
    }


# ── Sentiment classification metrics ─────────────────────────────────────────

def sentiment_metrics(
    y_true: List[str],   # list of gold sentiment strings
    y_pred: List[str],   # list of predicted sentiment strings
) -> Dict:
    """Compute sentiment classification metrics."""
    labels = SENTIMENT_LABELS
    macro_f1  = f1_score(y_true, y_pred, average="macro",  labels=labels, zero_division=0)
    acc       = accuracy_score(y_true, y_pred)

    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_class_p  = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_class_r  = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)

    per_class = {}
    for i, label in enumerate(labels):
        per_class[label] = {
            "precision": round(float(per_class_p[i]), 4),
            "recall":    round(float(per_class_r[i]), 4),
            "f1":        round(float(per_class_f1[i]), 4),
            "support":   int(y_true.count(label) if isinstance(y_true, list) else
                            (np.array(y_true) == label).sum()),
        }

    return {
        "macro_f1":  round(float(macro_f1), 4),
        "accuracy":  round(float(acc), 4),
        "per_class": per_class,
    }


# ── Pipeline (end-to-end) Pair-F1 ────────────────────────────────────────────

def pair_f1(
    gold_pairs: List[List[Tuple[str, str]]],   # [[(cat, sentiment), ...], ...]
    pred_pairs: List[List[Tuple[str, str]]],   # [[(cat, sentiment), ...], ...]
) -> Dict:
    """
    Compute Pair-F1 for end-to-end ABSA evaluation.

    A (category, sentiment) pair is correct only if both match exactly.
    This is the strictest — and most informative — evaluation for full pipelines.

    gold_pairs[i] = list of (coarse_category, sentiment) tuples for sentence i
    pred_pairs[i] = list of (coarse_category, sentiment) tuples predicted

    Returns macro-averaged Pair-F1, plus per-category breakdown.
    """
    total_tp = total_fp = total_fn = 0

    for gold, pred in zip(gold_pairs, pred_pairs):
        gold_set = set(gold)
        pred_set = set(pred)
        tp = len(gold_set & pred_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    eps   = 1e-8
    prec  = total_tp / (total_tp + total_fp + eps)
    rec   = total_tp / (total_tp + total_fn + eps)
    f1    = 2 * prec * rec / (prec + rec + eps)

    # Per (category, sentiment) breakdown
    pair_counts: Dict[tuple, Dict] = {}
    for gold, pred in zip(gold_pairs, pred_pairs):
        gold_set = set(gold)
        pred_set = set(pred)
        for pair in gold_set | pred_set:
            if pair not in pair_counts:
                pair_counts[pair] = {"tp": 0, "fp": 0, "fn": 0}
            if pair in gold_set and pair in pred_set:
                pair_counts[pair]["tp"] += 1
            elif pair in pred_set:
                pair_counts[pair]["fp"] += 1
            else:
                pair_counts[pair]["fn"] += 1

    return {
        "pair_precision": round(float(prec), 4),
        "pair_recall":    round(float(rec), 4),
        "pair_f1":        round(float(f1), 4),
        "total_tp":       total_tp,
        "total_fp":       total_fp,
        "total_fn":       total_fn,
    }


# ── Multi-aspect split metrics ────────────────────────────────────────────────

def multi_vs_single_aspect_sentiment(
    y_true:      List[str],
    y_pred:      List[str],
    num_aspects: List[int],   # number of aspects for each (text, aspect) pair
) -> Dict:
    """
    Break down sentiment accuracy by whether the source sentence
    was single-aspect vs multi-aspect.
    """
    y_true      = np.array(y_true)
    y_pred      = np.array(y_pred)
    num_aspects = np.array(num_aspects)

    results = {}
    for label, mask in [("single", num_aspects == 1), ("multi", num_aspects > 1)]:
        if mask.sum() == 0:
            results[label] = {}
            continue
        results[label] = {
            "n":       int(mask.sum()),
            "macro_f1": round(float(f1_score(
                y_true[mask], y_pred[mask],
                average="macro", labels=SENTIMENT_LABELS, zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(y_true[mask], y_pred[mask])), 4),
        }
    return results
