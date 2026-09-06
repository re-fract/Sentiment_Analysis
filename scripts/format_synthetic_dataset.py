"""
scripts/format_synthetic_dataset.py

Converts synthetic_raw.jsonl into a wide-format dataset where columns are:
  id, review, <ASPECT_CATEGORY_1>, <ASPECT_CATEGORY_2>, ...

Each category column contains the sentiment label ('positive', 'negative', 'neutral')
or empty/null if the aspect category is not mentioned in the review.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

# Standard fine categories in SemEval / ACOS taxonomy
DEFAULT_CATEGORIES = [
    "FOOD#QUALITY",
    "SERVICE#GENERAL",
    "AMBIENCE#GENERAL",
    "FOOD#PRICES",
    "FOOD#STYLE_OPTIONS",
    "RESTAURANT#GENERAL",
    "DRINKS#QUALITY",
    "RESTAURANT#PRICES",
    "RESTAURANT#MISCELLANEOUS",
    "LOCATION#GENERAL",
    "DRINKS#STYLE_OPTIONS",
    "DRINKS#PRICES",
    "FOOD#PORTION_SIZES",
]

COARSE_MAPPING = {
    "FOOD#QUALITY": "Food",
    "FOOD#PRICES": "Price",
    "FOOD#STYLE_OPTIONS": "Food",
    "SERVICE#GENERAL": "Service",
    "AMBIENCE#GENERAL": "Ambience",
    "RESTAURANT#PRICES": "Price",
    "RESTAURANT#GENERAL": "General",
    "RESTAURANT#MISCELLANEOUS": "General",
    "DRINKS#QUALITY": "Food",
    "DRINKS#PRICES": "Price",
    "DRINKS#STYLE_OPTIONS": "Food",
    "LOCATION#GENERAL": "General",
    "FOOD#PORTION_SIZES": "Food",
}


def normalize_sentiment(sent: Optional[str]) -> Optional[str]:
    if not sent:
        return None
    s = sent.strip().lower()
    if s in ("positive", "pos", "awesome", "great"):
        return "positive"
    if s in ("negative", "neg", "bad", "poor"):
        return "negative"
    if s in ("neutral", "neu", "conflict"):
        return "neutral"
    return s


def transform_jsonl_to_wide(
    input_path: Path,
    use_coarse: bool = False,
    categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Reads synthetic JSONL and returns a wide DataFrame."""
    records = []
    
    with open(input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            sample_id = (
                data.get("id")
                or data.get("review_id")
                or data.get("sample_id")
                or f"synthetic_{idx:06d}"
            )
            review_text = data.get("review") or data.get("text") or ""

            cat_sent_map = defaultdict(list)
            for lbl in data.get("labels", []):
                cat = lbl.get("category")
                raw_sent = lbl.get("sentiment")
                norm_sent = normalize_sentiment(raw_sent)
                if not cat or not norm_sent:
                    continue
                if use_coarse:
                    target_cat = COARSE_MAPPING.get(cat, "General")
                else:
                    target_cat = cat
                cat_sent_map[target_cat].append(norm_sent)

            records.append({
                "id": sample_id,
                "review": review_text,
                "cat_sent_map": cat_sent_map,
            })

    if categories is None:
        if use_coarse:
            categories = ["Food", "Service", "Ambience", "Price", "General"]
        else:
            all_seen_cats = set()
            for r in records:
                all_seen_cats.update(r["cat_sent_map"].keys())
            # Keep order from DEFAULT_CATEGORIES first, then any extra seen categories
            categories = [c for c in DEFAULT_CATEGORIES if c in all_seen_cats]
            for c in sorted(all_seen_cats):
                if c not in categories:
                    categories.append(c)

    # Build row dicts
    rows = []
    for r in records:
        row = {
            "id": r["id"],
            "review": r["review"],
        }
        for cat in categories:
            sents = r["cat_sent_map"].get(cat, [])
            if not sents:
                row[cat] = None
            else:
                unique_sents = list(dict.fromkeys(sents))
                if len(unique_sents) == 1:
                    row[cat] = unique_sents[0]
                else:
                    row[cat] = ", ".join(unique_sents)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def main():
    parser = argparse.ArgumentParser(description="Convert synthetic_raw.jsonl to wide aspect category format.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/augmented/synthetic_raw.jsonl",
        help="Path to input jsonl file (default: data/augmented/synthetic_raw.jsonl)",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="data/augmented/synthetic_aspect_categories.csv",
        help="Path to output CSV file",
    )
    parser.add_argument(
        "--out-jsonl",
        type=str,
        default="data/augmented/synthetic_aspect_categories.jsonl",
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--coarse",
        action="store_true",
        help="Map categories to coarse taxonomy (Food, Service, Ambience, Price, General)",
    )

    args = parser.parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    print(f"Reading from: {in_path}")
    df = transform_jsonl_to_wide(in_path, use_coarse=args.coarse)

    if args.out_csv:
        out_csv_path = Path(args.out_csv)
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv_path, index=False, encoding="utf-8")
        print(f"Saved CSV ({len(df)} rows) to: {out_csv_path}")

    if args.out_jsonl:
        out_jsonl_path = Path(args.out_jsonl)
        out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(out_jsonl_path, orient="records", lines=True, force_ascii=False)
        print(f"Saved JSONL ({len(df)} rows) to: {out_jsonl_path}")

    print("\nDataset Preview:")
    print(df.head(2).to_dict(orient="records"))


if __name__ == "__main__":
    main()
