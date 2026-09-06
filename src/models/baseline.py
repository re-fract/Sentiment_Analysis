"""
src/models/baseline.py

Classical ML baselines for ABSA using TF-IDF + scikit-learn.

Two tasks:
  1. Aspect detection   — multi-label classification: which coarse categories
                          are mentioned in a review?
  2. Sentiment classification — given (text, aspect), predict sentiment.

We train separate models per task (not a joint model) since the baselines
serve as a performance floor to compare transformer models against.

Models evaluated:
  - Logistic Regression   (strong baseline, fast)
  - LinearSVC             (often best for text classification)
  - Naive Bayes           (fast, interpretable)

Outputs:
  outputs/results/baseline_aspect_results.json
  outputs/results/baseline_sentiment_results.json
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score,
    hamming_loss,
)

from config.taxonomy import COARSE_CATEGORIES, SENTIMENT_LABELS, normalise_sentiment
from src.utils.io import save_json
from src.utils.logging import get_logger

log = get_logger("baseline")

COARSE_LIST   = COARSE_CATEGORIES           # ["Food","Service","Ambience","Price","General"]
SENTIMENT_LIST= SENTIMENT_LABELS            # ["positive","negative","neutral"]


# ── Data preparation ──────────────────────────────────────────────────────────

def prepare_aspect_data(df: pd.DataFrame) -> tuple[List[str], List[List[str]]]:
    """Return (texts, list_of_coarse_cat_sets) for multi-label aspect detection."""
    texts  = []
    labels = []
    for _, row in df.iterrows():
        try:
            coarse = list(set(ast.literal_eval(row["categories_coarse"])))
        except Exception:
            coarse = []
        if not coarse:
            continue
        texts.append(str(row["text"]))
        labels.append(coarse)
    return texts, labels


def prepare_sentiment_data(df: pd.DataFrame) -> tuple[List[str], List[str]]:
    """
    Return (aspect_conditioned_texts, sentiments) for sentiment classification.

    The input text is formatted as:  "[ASPECT] {coarse_category} [SEP] {review_text}"
    This mimics the transformer approach at the classical level.
    """
    texts = []
    sents = []
    for _, row in df.iterrows():
        try:
            coarse = ast.literal_eval(row["categories_coarse"])
            sentiment = ast.literal_eval(row["sentiments"])
        except Exception:
            continue
        text = str(row["text"])
        for cat, sent in zip(coarse, sentiment):
            sent_norm = normalise_sentiment(sent)
            if sent_norm not in SENTIMENT_LIST:
                continue
            # Aspect-conditioned input
            texts.append(f"[ASPECT] {cat} [SEP] {text}")
            sents.append(sent_norm)
    return texts, sents


# ── Model building ────────────────────────────────────────────────────────────

def _make_pipelines(task: str) -> dict[str, Pipeline]:
    """Build named sklearn pipelines for the given task."""
    tfidf_params = dict(
        ngram_range=(1, 2),
        max_features=50_000,
        sublinear_tf=True,
        min_df=2,
    )

    if task == "aspect":
        # Multi-label: wrap each classifier in OneVsRestClassifier
        return {
            "LR":    Pipeline([("tfidf", TfidfVectorizer(**tfidf_params)),
                               ("clf", OneVsRestClassifier(
                                   LogisticRegression(max_iter=1000, C=1.0)))]),
            "SVC":   Pipeline([("tfidf", TfidfVectorizer(**tfidf_params)),
                               ("clf", OneVsRestClassifier(
                                   LinearSVC(max_iter=2000, C=1.0)))]),
        }
    else:  # sentiment — single-label
        return {
            "LR":  Pipeline([("tfidf", TfidfVectorizer(**tfidf_params)),
                              ("clf", LogisticRegression(max_iter=1000, C=1.0))]),
            "SVC": Pipeline([("tfidf", TfidfVectorizer(**tfidf_params)),
                              ("clf", LinearSVC(max_iter=2000, C=1.0))]),
            "NB":  Pipeline([("tfidf", TfidfVectorizer(
                                  ngram_range=(1, 2), max_features=50_000,
                                  sublinear_tf=False, min_df=2)),
                              ("clf", MultinomialNB())]),
        }


# ── Aspect detection ──────────────────────────────────────────────────────────

def train_eval_aspect(
    train_df:  pd.DataFrame,
    test_df:   pd.DataFrame,
    train_name:str,
) -> Dict:
    log.info(f"\n[Aspect Detection] train={train_name}  "
             f"n_train={len(train_df)}  n_test={len(test_df)}")

    X_train, y_train_raw = prepare_aspect_data(train_df)
    X_test,  y_test_raw  = prepare_aspect_data(test_df)

    mlb = MultiLabelBinarizer(classes=COARSE_LIST)
    Y_train = mlb.fit_transform(y_train_raw)
    Y_test  = mlb.transform(y_test_raw)

    results = {}
    pipelines = _make_pipelines("aspect")

    for name, pipe in pipelines.items():
        log.info(f"  Training {name}...")
        pipe.fit(X_train, Y_train)
        Y_pred = pipe.predict(X_test)

        macro_f1  = f1_score(Y_test, Y_pred, average="macro",  zero_division=0)
        micro_f1  = f1_score(Y_test, Y_pred, average="micro",  zero_division=0)
        ham_loss  = hamming_loss(Y_test, Y_pred)

        per_class_f1 = f1_score(Y_test, Y_pred, average=None, zero_division=0)
        per_class = {cat: round(float(f), 4)
                     for cat, f in zip(COARSE_LIST, per_class_f1)}

        log.info(f"    macro-F1={macro_f1:.4f}  micro-F1={micro_f1:.4f}  "
                 f"hamming={ham_loss:.4f}")
        log.info(f"    per-class: {per_class}")

        results[name] = {
            "macro_f1":  round(macro_f1, 4),
            "micro_f1":  round(micro_f1, 4),
            "hamming":   round(ham_loss, 4),
            "per_class": per_class,
        }

    return results


# ── Sentiment classification ──────────────────────────────────────────────────

def train_eval_sentiment(
    train_df:  pd.DataFrame,
    test_df:   pd.DataFrame,
    train_name:str,
) -> Dict:
    log.info(f"\n[Sentiment] train={train_name}  "
             f"n_train={len(train_df)}  n_test={len(test_df)}")

    X_train, y_train = prepare_sentiment_data(train_df)
    X_test,  y_test  = prepare_sentiment_data(test_df)

    results = {}
    pipelines = _make_pipelines("sentiment")

    for name, pipe in pipelines.items():
        log.info(f"  Training {name}...")
        try:
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
        except Exception as e:
            log.warning(f"  {name} failed: {e}")
            continue

        macro_f1 = f1_score(y_test, y_pred, average="macro",  zero_division=0,
                            labels=SENTIMENT_LIST)
        acc      = accuracy_score(y_test, y_pred)

        per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0,
                                labels=SENTIMENT_LIST)
        per_class = {s: round(float(f), 4)
                     for s, f in zip(SENTIMENT_LIST, per_class_f1)}

        log.info(f"    macro-F1={macro_f1:.4f}  accuracy={acc:.4f}")
        log.info(f"    per-class: {per_class}")

        results[name] = {
            "macro_f1": round(macro_f1, 4),
            "accuracy": round(acc, 4),
            "per_class":per_class,
        }

    return results


# ── Main function ─────────────────────────────────────────────────────────────

def run_baselines(splits: dict, output_dir: str = "outputs/results") -> dict:
    """
    Run all baseline experiments and save results.

    Experiments:
      B1: SemEval train only
      B2: Full train (SemEval + MAMS + ACOS)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_results = {}

    experiment_sets = {
        "B1_semeval_only": (splits["semeval_train"], splits["semeval_test"]),
        "B2_full_train":   (splits["full_train"],    splits["mams_test"]),   # test on MAMS (hardest)
    }

    # ── Aspect detection ──────────────────────────────────────────────────────
    aspect_results = {}
    for exp_name, (train_df, test_df) in experiment_sets.items():
        if train_df.empty or test_df.empty:
            log.warning(f"Skipping {exp_name} (empty split)")
            continue
        aspect_results[exp_name] = train_eval_aspect(train_df, test_df, exp_name)

    save_json(aspect_results, f"{output_dir}/baseline_aspect_results.json")
    log.info(f"Saved -> {output_dir}/baseline_aspect_results.json")
    all_results["aspect"] = aspect_results

    # ── Sentiment classification ──────────────────────────────────────────────
    sentiment_results = {}
    for exp_name, (train_df, test_df) in experiment_sets.items():
        if train_df.empty or test_df.empty:
            continue
        sentiment_results[exp_name] = train_eval_sentiment(train_df, test_df, exp_name)

    save_json(sentiment_results, f"{output_dir}/baseline_sentiment_results.json")
    log.info(f"Saved -> {output_dir}/baseline_sentiment_results.json")
    all_results["sentiment"] = sentiment_results

    # ── Summary table ─────────────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("BASELINE SUMMARY")
    log.info("="*60)
    for task, task_results in all_results.items():
        log.info(f"\n  Task: {task}")
        for exp, models in task_results.items():
            for model, metrics in models.items():
                log.info(f"    {exp:<25s} {model:<6s}  "
                         f"macro-F1={metrics['macro_f1']:.4f}")

    return all_results
