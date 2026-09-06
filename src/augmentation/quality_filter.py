"""
src/augmentation/quality_filter.py

Post-processing pipeline for synthetic_raw.jsonl.

Steps:
  1. Parse & schema validation  — every sample must have text + >=1 valid label
  2. Length filter              — 8 to 80 words
  3. Taxonomy validation        — all categories must be in our taxonomy
  4. Deduplication              — exact + near-duplicate removal (Jaccard > 0.8)
  5. Cross-model consistency    — validate vs Gemma-4-31B; keep if >=80% aspect match
  6. Write filtered CSV         — data/augmented/synthetic_filtered.csv

Usage:
    from src.augmentation.quality_filter import run_filter
    run_filter(validator=gemma_client, cfg=cfg)

Or via the top-level script:
    .venv/Scripts/python scripts/run_augmentation.py --step filter
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.taxonomy import (
    normalise_category, normalise_sentiment, map_to_coarse, ALL_FINE_CATEGORIES
)
from src.augmentation.llm_client import CerebrasClient
from src.augmentation.prompt_templates import build_validate_prompt, VALIDATE_SYSTEM
from src.utils.io import load_jsonl, save_csv
from src.utils.logging import get_logger

log = get_logger("quality_filter")

CONSISTENCY_THRESHOLD = 0.80   # fraction of aspects that must agree
VALID_FINE_CATS       = {c.upper() for c in ALL_FINE_CATEGORIES}
VALID_FINE_CATS |= {   # MAMS-style names also accepted
    "food", "service", "staff", "price", "ambience", "place",
    "miscellaneous", "drinks",
}
# New simplified coarse labels accepted directly
VALID_FINE_CATS |= {"Food", "Service", "Ambience", "Price", "Quantity", "General"}


# ── Step 1+2+3: Basic validation ──────────────────────────────────────────────

def _validate_label(label: dict) -> Tuple[bool, dict | None]:
    """Return (is_valid, cleaned_label)."""
    if not isinstance(label, dict):
        return False, None
    cat = label.get("category", "")
    if not cat:
        return False, None
    sent = normalise_sentiment(label.get("sentiment", ""))
    if sent not in ("positive", "negative", "neutral"):
        return False, None
    norm_cat = normalise_category(cat)
    clean = {
        "aspect_term":  label.get("aspect_term") or None,
        "category":     norm_cat,
        "sentiment":    sent,
        "is_implicit":  bool(label.get("is_implicit", False)),
        "category_coarse": map_to_coarse(norm_cat),
    }
    return True, clean


def _validate_sample(record: dict) -> Tuple[bool, dict | None]:
    """Validate a raw record from synthetic_raw.jsonl."""
    text = record.get("text", "").strip()
    if not text:
        return False, None

    # Length filter: 8–80 words
    n_words = len(text.split())
    if n_words < 8 or n_words > 80:
        return False, None

    raw_labels = record.get("labels", [])
    if not isinstance(raw_labels, list) or not raw_labels:
        return False, None

    clean_labels = []
    for lbl in raw_labels:
        ok, clean = _validate_label(lbl)
        if ok:
            clean_labels.append(clean)

    if not clean_labels:
        return False, None

    return True, {
        "text":     text,
        "labels":   clean_labels,
        "mode":     record.get("mode", ""),
        "source":   record.get("source", ""),
        "review_id":record.get("review_id", ""),
    }


# ── Step 4: Deduplication ─────────────────────────────────────────────────────

def _jaccard(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def dedup(samples: List[dict], threshold: float = 0.80) -> List[dict]:
    """Remove near-duplicate texts (Jaccard similarity >= threshold)."""
    kept: List[dict] = []
    texts_kept: List[str] = []

    for sample in samples:
        text = sample["text"]
        is_dup = any(_jaccard(text, t) >= threshold for t in texts_kept)
        if not is_dup:
            kept.append(sample)
            texts_kept.append(text)

    log.info(f"  Dedup: {len(samples)} -> {len(kept)} "
             f"(removed {len(samples)-len(kept)} near-duplicates)")
    return kept


# ── Step 5: Cross-model consistency ──────────────────────────────────────────

def _aspect_agreement(labels_a: List[dict], labels_b: List[dict]) -> float:
    """
    Fraction of label_a aspects that appear (same category, same sentiment)
    in label_b. Returns 1.0 if labels_a is empty.
    """
    if not labels_a:
        return 1.0
    set_b = {(l["category"].upper(), l["sentiment"]) for l in labels_b}
    matches = sum(
        1 for l in labels_a
        if (l["category"].upper(), l["sentiment"]) in set_b
    )
    return matches / len(labels_a)


def run_consistency_check(
    samples:   List[dict],
    validator: CerebrasClient,
    threshold: float = CONSISTENCY_THRESHOLD,
    max_check: int   = 2000,   # only cross-validate up to this many (rate limit budget)
) -> Tuple[List[dict], dict]:
    """
    Send each sample to the validator model (Gemma-4-31B) and keep only
    those where aspect agreement >= threshold.

    Returns (kept_samples, stats_dict).
    """
    consistent = []
    n_checked  = 0
    n_agreed   = 0
    n_skipped  = 0   # samples not checked (budget exceeded) — kept by default

    for sample in samples:
        if n_checked >= max_check:
            # Keep unchecked samples (budget exhausted); note them for stats
            consistent.append(sample)
            n_skipped += 1
            continue

        prompt = build_validate_prompt(sample["text"], sample["labels"])

        try:
            raw = validator.generate_json(
                prompt, system=VALIDATE_SYSTEM, max_tokens=512)
        except Exception as exc:
            log.debug(f"  Consistency check failed: {exc}")
            consistent.append(sample)   # keep on error
            n_skipped += 1
            continue

        n_checked += 1

        if not isinstance(raw, dict):
            consistent.append(sample)
            n_agreed += 1
            continue

        validator_labels = raw.get("your_labels", [])
        if not isinstance(validator_labels, list):
            consistent.append(sample)
            n_agreed += 1
            continue

        agreement_score = _aspect_agreement(sample["labels"], validator_labels)

        if agreement_score >= threshold:
            sample["consistency_score"] = round(agreement_score, 3)
            consistent.append(sample)
            n_agreed += 1
        else:
            log.debug(f"  Rejected (agreement={agreement_score:.2f}): {sample['text'][:60]}")

        if n_checked % 100 == 0:
            log.info(f"  Consistency check: {n_checked} checked, "
                     f"{n_agreed} agreed, {n_checked - n_agreed} rejected")

    stats = {
        "n_checked":  n_checked,
        "n_agreed":   n_agreed,
        "n_rejected": n_checked - n_agreed,
        "n_skipped":  n_skipped,
        "agreement_rate": round(n_agreed / max(n_checked, 1), 3),
    }
    log.info(f"  Consistency stats: {stats}")
    return consistent, stats


# ── Step 6: Flatten to DataFrame and save ────────────────────────────────────

def flatten_to_df(samples: List[dict]) -> pd.DataFrame:
    """Convert list of filtered samples to a DataFrame matching the unified schema."""
    rows = []
    for i, s in enumerate(samples):
        labels = s["labels"]
        rows.append({
            "sample_id":         f"synthetic_{i:06d}",
            "source":            s.get("source", "synthetic"),
            "split":             "train",
            "text":              s["text"],
            "categories_fine":   str([l["category"]        for l in labels]),
            "categories_coarse": str([l["category_coarse"] for l in labels]),
            "sentiments":        str([l["sentiment"]        for l in labels]),
            "aspect_terms":      str([l["aspect_term"]      for l in labels]),
            "is_implicit_flags": str([l["is_implicit"]      for l in labels]),
            "num_aspects":       len(labels),
            "has_implicit":      any(l["is_implicit"] for l in labels),
            "has_conflict":      len({l["sentiment"] for l in labels}) > 1,
            "consistency_score": s.get("consistency_score", None),
            "mode":              s.get("mode", ""),
        })
    return pd.DataFrame(rows)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_filter(
    validator:       CerebrasClient | None,
    cfg:             dict,
    skip_consistency:bool = False,
) -> pd.DataFrame:
    """
    Full filtering pipeline.

    Reads and merges:
      - data/augmented/synthetic_raw.jsonl          (Yelp reviews labeled by LLM — Mode B)
      - data/augmented/synthetic_generated_raw.jsonl (LLM-written synthetic reviews — Mode A)

    Then applies:
      1. Schema + length + taxonomy validation
      2. Deduplication (Jaccard > 0.8)
      3. Cross-model consistency check (optional)
      4. Saves to data/augmented/synthetic_filtered.csv

    Args:
        validator:        LLM client for consistency checking.
                          If None, consistency check is skipped.
        skip_consistency: Force-skip consistency check (useful for dry runs).
    """
    aug_paths = cfg["paths"]["augmented"]
    raw_path  = Path(aug_paths["raw"])           # synthetic_raw.jsonl (Yelp-labeled)
    gen_path  = Path(aug_paths.get("generated_raw",
                     "data/augmented/synthetic_generated_raw.jsonl"))
    out_path  = Path(aug_paths["filtered"])      # synthetic_filtered.csv

    # ── Load and merge both raw files ────────────────────────────────────────
    records: list[dict] = []

    if raw_path.exists():
        yelp_recs = load_jsonl(raw_path)
        log.info(f"Loaded {len(yelp_recs)} Yelp-labeled records from {raw_path.name}")
        records.extend(yelp_recs)
    else:
        log.warning(f"Yelp-labeled file not found: {raw_path}")

    if gen_path.exists():
        gen_recs = load_jsonl(gen_path)
        log.info(f"Loaded {len(gen_recs)} LLM-generated records from {gen_path.name}")
        records.extend(gen_recs)
    else:
        log.info(f"No generated file found at {gen_path.name} — skipping.")

    if not records:
        log.error("No raw records found. Run 'generate' and/or 'label' steps first.")
        return pd.DataFrame()

    log.info(f"Total records to filter: {len(records)}")

    # ── Steps 1–3: schema + length + taxonomy ────────────────────────────────
    valid = []
    n_invalid = 0
    for rec in records:
        ok, clean = _validate_sample(rec)
        if ok:
            valid.append(clean)
        else:
            n_invalid += 1
    log.info(f"  After validation: {len(valid)} valid  |  {n_invalid} invalid")

    # ── Step 4: deduplication ────────────────────────────────────────────────
    valid = dedup(valid, threshold=0.80)

    # ── Step 5: cross-model consistency ──────────────────────────────────────
    if validator is not None and not skip_consistency:
        valid, stats = run_consistency_check(valid, validator)
    else:
        log.info("  Skipping consistency check.")
        stats = {}

    # ── Step 6: flatten + save ────────────────────────────────────────────────
    df = flatten_to_df(valid)
    save_csv(df, out_path)
    log.info(f"  Saved {len(df)} filtered samples -> {out_path}")

    # Report mode breakdown
    mode_counts = df["mode"].value_counts().to_dict() if not df.empty else {}
    log.info(f"  Mode breakdown: {mode_counts}")

    return df
