"""
src/models/llm_classifier.py

LLM-as-Classifier: zero-shot and few-shot ABSA using the Cerebras API.

This module runs the same ABSA task directly through an LLM without any
fine-tuning, using structured prompts.  Results are compared against our
fine-tuned models to establish a "GPT baseline" for the dissertation.

Two modes:
  zero_shot  — prompt only includes task description + taxonomy
  few_shot   — also includes 4–8 training examples (from MAMS/ACOS)

Output:
  outputs/results/llm_classifier_{model}_{mode}_results.json
"""

from __future__ import annotations

import ast
import json
import sys
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from config.taxonomy import (
    COARSE_CATEGORIES, SENTIMENT_LABELS, map_to_coarse, normalise_sentiment
)
from src.augmentation.llm_client import CerebrasClient
from src.augmentation.prompt_templates import TAXONOMY_BLOCK
from src.evaluation.metrics import pair_f1, sentiment_metrics
from src.utils.io import save_json, load_config
from src.utils.logging import get_logger

log = get_logger("llm_classifier")

ZERO_SHOT_SYSTEM = (
    "You are an expert Aspect-Based Sentiment Analysis (ABSA) annotator. "
    "Given a restaurant review, extract all aspect-sentiment pairs using "
    "the provided taxonomy. Output ONLY valid JSON."
)

FEW_SHOT_EXAMPLES_FIXED = [
    {
        "text": "Absolutely fantastic food but the service let us down.",
        "predictions": [
            {"category": "Food",    "sentiment": "positive"},
            {"category": "Service", "sentiment": "negative"},
        ]
    },
    {
        "text": "Great location, reasonable prices, though the ambience could be better.",
        "predictions": [
            {"category": "General", "sentiment": "positive"},
            {"category": "Price",   "sentiment": "positive"},
            {"category": "Ambience","sentiment": "negative"},
        ]
    },
    {
        "text": "The ramen was delicious — rich, complex broth.",
        "predictions": [
            {"category": "Food", "sentiment": "positive"},
        ]
    },
    {
        "text": "Staff were dismissive and we waited over an hour for our mains.",
        "predictions": [
            {"category": "Service", "sentiment": "negative"},
        ]
    },
]


def _build_prompt(
    text:     str,
    mode:     str,
    examples: Optional[List[Dict]] = None,
) -> str:
    examples = examples or []

    if mode == "zero_shot":
        few_shot_block = ""
    else:
        # Build few-shot block from fixed + dynamic examples
        example_lines = []
        for ex in (FEW_SHOT_EXAMPLES_FIXED + examples)[:6]:
            example_lines.append(
                f'Review: "{ex["text"]}"\n'
                f'Output: {json.dumps({"predictions": ex["predictions"]})}'
            )
        few_shot_block = "\n\nExamples:\n" + "\n\n".join(example_lines) + "\n"

    prompt = f"""You are performing Aspect-Based Sentiment Analysis on restaurant reviews.

{TAXONOMY_BLOCK}

Task: Identify all aspect categories mentioned in the review and their sentiments.
Use ONLY the 5 coarse categories: Food, Service, Ambience, Price, General.
Each category should appear at most once per review.
If a category is not mentioned, do not include it.
{few_shot_block}
Review: "{text}"

Output as JSON:
{{"predictions": [{{"category": "...", "sentiment": "..."}}]}}

Output ONLY the JSON object:"""

    return prompt


def _parse_llm_response(raw: Optional[Dict]) -> List[Tuple[str, str]]:
    """Parse LLM JSON response into list of (category, sentiment) tuples."""
    if not isinstance(raw, dict):
        return []
    preds = raw.get("predictions", [])
    if not isinstance(preds, list):
        return []

    result = []
    seen_cats = set()
    for pred in preds:
        if not isinstance(pred, dict):
            continue
        cat  = pred.get("category", "")
        sent = normalise_sentiment(pred.get("sentiment", ""))
        # Map to coarse if needed
        cat_mapped = map_to_coarse(cat) if cat not in COARSE_CATEGORIES else cat
        if cat_mapped in COARSE_CATEGORIES and sent in SENTIMENT_LABELS:
            if cat_mapped not in seen_cats:
                result.append((cat_mapped, sent))
                seen_cats.add(cat_mapped)
    return result


def run_llm_classifier(
    client:   CerebrasClient,
    test_df:  pd.DataFrame,
    mode:     str = "zero_shot",       # "zero_shot" | "few_shot"
    model:    str = "gemma-4-31b",
    n_samples:int = 200,               # evaluate on a subset to stay within API budget
    seed:     int = 42,
) -> Dict:
    """
    Run LLM-as-Classifier on n_samples from test_df.

    Returns evaluation metrics (Pair-F1, sentiment accuracy).
    """
    assert mode in ("zero_shot", "few_shot"), f"Invalid mode: {mode}"

    rng = random.Random(seed)
    sample_df = test_df.sample(min(n_samples, len(test_df)), random_state=seed)

    gold_pairs: List[List[Tuple[str, str]]] = []
    pred_pairs: List[List[Tuple[str, str]]] = []
    gold_sents: List[str] = []
    pred_sents: List[str] = []

    done = 0
    errors = 0

    for _, row in sample_df.iterrows():
        text = str(row["text"])
        try:
            gold_cats  = ast.literal_eval(row["categories_coarse"])
            gold_s     = ast.literal_eval(row["sentiments"])
        except Exception:
            gold_cats, gold_s = [], []

        gold = [(c, s) for c, s in zip(gold_cats, gold_s)]
        gold_pairs.append(gold)
        gold_sents.extend(gold_s)

        prompt = _build_prompt(text, mode)

        try:
            raw = client.generate_json(prompt, system=ZERO_SHOT_SYSTEM, max_tokens=256)
            pred = _parse_llm_response(raw)
        except Exception as e:
            log.debug(f"LLM call failed: {e}")
            pred = []
            errors += 1

        pred_pairs.append(pred)
        pred_sents.extend([s for _, s in pred])

        done += 1
        if done % 20 == 0:
            log.info(f"  [{mode}] {done}/{n_samples} done  errors={errors}")

    # Pad pred_sents if lengths mismatch (for sentiment metrics)
    if len(pred_sents) < len(gold_sents):
        pred_sents += ["neutral"] * (len(gold_sents) - len(pred_sents))
    elif len(pred_sents) > len(gold_sents):
        pred_sents = pred_sents[:len(gold_sents)]

    pipe_metrics = pair_f1(gold_pairs, pred_pairs)
    sent_m       = sentiment_metrics(
        gold_sents[:len(pred_sents)], pred_sents) if gold_sents else {}

    results = {
        "model":    model,
        "mode":     mode,
        "n_samples":done,
        "n_errors": errors,
        "pipeline": pipe_metrics,
        "sentiment":sent_m,
    }

    log.info(f"\n[{mode}] Pair-F1={pipe_metrics['pair_f1']:.4f}  "
             f"Sent-F1={sent_m.get('macro_f1', 0):.4f}")
    return results
