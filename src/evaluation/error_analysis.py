"""
src/evaluation/error_analysis.py

Qualitative error analysis helpers.

Produces structured outputs for dissertation Section 5 (Analysis):
  1. Confusion matrix for sentiment (per test set)
  2. Most common error patterns (false positives / false negatives per aspect)
  3. Multi-aspect vs single-aspect error breakdown
  4. Hardest examples (lowest pipeline confidence)
  5. Per-star-rating accuracy (Yelp silver labels)
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from config.taxonomy import COARSE_CATEGORIES, SENTIMENT_LABELS
from src.utils.logging import get_logger

log = get_logger("error_analysis")


def sentiment_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
) -> pd.DataFrame:
    """Return labelled confusion matrix as a DataFrame."""
    cm = confusion_matrix(y_true, y_pred, labels=SENTIMENT_LABELS)
    return pd.DataFrame(cm, index=SENTIMENT_LABELS, columns=SENTIMENT_LABELS)


def aspect_error_patterns(
    test_df:    pd.DataFrame,
    y_pred_bin: np.ndarray,   # (N, num_coarse) binary predictions
) -> Dict:
    """
    Find the most common False Positive (FP) and False Negative (FN)
    aspect-category errors.

    Returns dict with:
      false_positives[cat] = count of times cat was predicted but not in gold
      false_negatives[cat] = count of times cat was in gold but not predicted
    """
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=COARSE_CATEGORIES)

    gold_cats = []
    for _, row in test_df.iterrows():
        try:
            cats = list(set(ast.literal_eval(row["categories_coarse"])))
        except Exception:
            cats = []
        gold_cats.append(cats)

    y_true = mlb.fit_transform(gold_cats)

    fp_counts = {cat: 0 for cat in COARSE_CATEGORIES}
    fn_counts = {cat: 0 for cat in COARSE_CATEGORIES}

    for i, cat in enumerate(COARSE_CATEGORIES):
        fp_counts[cat] = int(((y_pred_bin[:, i] == 1) & (y_true[:, i] == 0)).sum())
        fn_counts[cat] = int(((y_pred_bin[:, i] == 0) & (y_true[:, i] == 1)).sum())

    return {"false_positives": fp_counts, "false_negatives": fn_counts}


def find_hard_examples(
    test_df:    pd.DataFrame,
    pipeline,   # ABSAPipeline instance
    n:          int = 20,
) -> pd.DataFrame:
    """
    Find the n hardest examples: sentences where the pipeline has lowest
    max aspect probability (i.e., the detector was most uncertain).
    """
    rows = []
    for _, row in test_df.iterrows():
        text = str(row["text"])
        try:
            gold_cats  = ast.literal_eval(row["categories_coarse"])
            gold_sents = ast.literal_eval(row["sentiments"])
        except Exception:
            gold_cats, gold_sents = [], []

        results  = pipeline.analyze_with_proba(text)
        pred_cats = [r["category"]  for r in results]
        pred_sents= [r["sentiment"] for r in results]
        avg_conf  = np.mean([r["aspect_probability"] for r in results]) if results else 0.0

        rows.append({
            "text":       text,
            "gold_cats":  gold_cats,
            "gold_sents": gold_sents,
            "pred_cats":  pred_cats,
            "pred_sents": pred_sents,
            "avg_confidence": avg_conf,
            "correct": (sorted(zip(gold_cats, gold_sents)) ==
                        sorted(zip(pred_cats, pred_sents))),
        })

    df = pd.DataFrame(rows)
    return df.nsmallest(n, "avg_confidence")


def implicit_vs_explicit_breakdown(
    test_df:    pd.DataFrame,
    y_true_sent: List[str],
    y_pred_sent: List[str],
) -> Dict:
    """
    Compare sentiment accuracy on explicit vs implicit aspect mentions.
    Requires acos_test which has is_implicit_flags.
    """
    if "is_implicit_flags" not in test_df.columns:
        return {}

    from sklearn.metrics import f1_score
    impl_true, impl_pred = [], []
    expl_true, expl_pred = [], []

    idx = 0
    for _, row in test_df.iterrows():
        try:
            sents = ast.literal_eval(row["sentiments"])
            flags = ast.literal_eval(row["is_implicit_flags"])
        except Exception:
            sents, flags = [], []

        for s, flag in zip(sents, flags):
            if idx >= len(y_true_sent):
                break
            if flag:
                impl_true.append(y_true_sent[idx])
                impl_pred.append(y_pred_sent[idx])
            else:
                expl_true.append(y_true_sent[idx])
                expl_pred.append(y_pred_sent[idx])
            idx += 1

    results = {}
    for label, true, pred in [("explicit", expl_true, expl_pred),
                               ("implicit", impl_true, impl_pred)]:
        if not true:
            continue
        results[label] = {
            "n":        len(true),
            "macro_f1": round(float(f1_score(
                true, pred, average="macro",
                labels=SENTIMENT_LABELS, zero_division=0)), 4),
        }
    return results


def print_error_report(
    exp_id:     str,
    test_name:  str,
    cm:         pd.DataFrame,
    patterns:   Dict,
    implicit:   Dict,
) -> None:
    """Pretty-print an error analysis report to the logger."""
    log.info(f"\n{'='*60}")
    log.info(f"ERROR ANALYSIS: {exp_id} on {test_name}")
    log.info(f"{'='*60}")

    log.info("\nSentiment Confusion Matrix (rows=gold, cols=pred):")
    log.info("\n" + str(cm))

    log.info("\nAspect False Positives (predicted but not gold):")
    for cat, cnt in sorted(patterns["false_positives"].items(), key=lambda x: -x[1]):
        if cnt > 0:
            log.info(f"  {cat:<12}: {cnt}")

    log.info("\nAspect False Negatives (in gold but not predicted):")
    for cat, cnt in sorted(patterns["false_negatives"].items(), key=lambda x: -x[1]):
        if cnt > 0:
            log.info(f"  {cat:<12}: {cnt}")

    if implicit:
        log.info("\nImplicit vs Explicit breakdown (ACOS):")
        for kind, stats in implicit.items():
            log.info(f"  {kind:<10}: n={stats['n']}  macro-F1={stats['macro_f1']:.4f}")
