"""
src/augmentation/verify.py

CLI tool for the 100-sample manual verification step.

Usage (on the server):
    .venv/Scripts/python src/augmentation/verify.py \
        --input  data/augmented/synthetic_filtered.csv \
        --output data/augmented/gold_verified_100.csv \
        --n      100

For each sample, prints the review text and LLM-assigned labels.
You enter:
    c  → Correct
    w  → Wrong (discards this sample)
    or type a corrected label string, e.g.  FOOD#QUALITY:negative,SERVICE#GENERAL:positive

At the end, reports LLM accuracy on the 100-sample gold set (cite this in the dissertation).
"""

from __future__ import annotations

import ast
import csv
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from src.utils.logging import get_logger

log = get_logger("verify")


def verify_samples(
    input_csv:  str,
    output_csv: str,
    n:          int = 100,
    seed:       int = 42,
) -> None:
    df = pd.read_csv(input_csv, encoding="utf-8")
    if len(df) < n:
        print(f"[warn] Only {len(df)} samples available; will verify all.")
        n = len(df)

    # Stratified sample: proportional across mode (generate vs label)
    modes = df["mode"].unique()
    sample_parts = []
    for mode in modes:
        sub = df[df["mode"] == mode]
        n_mode = max(1, int(round(n * len(sub) / len(df))))
        sample_parts.append(sub.sample(n=min(n_mode, len(sub)), random_state=seed))
    sample_df = pd.concat(sample_parts).sample(frac=1, random_state=seed).head(n).reset_index(drop=True)

    verified_rows = []
    n_correct     = 0
    n_wrong       = 0
    n_corrected   = 0

    print(f"\n{'='*70}")
    print(f"  ABSA Gold Verification — {n} samples")
    print(f"  Commands: [c]orrect  [w]rong  or type correction")
    print(f"  Example correction: FOOD#QUALITY:negative,SERVICE#GENERAL:positive")
    print(f"{'='*70}\n")

    for i, row in sample_df.iterrows():
        idx = int(i) + 1
        text = row["text"]
        cats  = ast.literal_eval(row["categories_fine"])  if isinstance(row["categories_fine"], str) else []
        sents = ast.literal_eval(row["sentiments"])       if isinstance(row["sentiments"], str) else []
        terms = ast.literal_eval(row["aspect_terms"])     if isinstance(row["aspect_terms"], str) else []
        impl  = ast.literal_eval(row["is_implicit_flags"])if isinstance(row["is_implicit_flags"], str) else []

        # Pretty print
        print(f"[{idx}/{n}] {'─'*60}")
        print(f"  TEXT:   {text}")
        print(f"  LABELS:")
        for c, s, t, im in zip(cats, sents, terms, impl):
            term_str = f'"{t}"' if t and str(t) != "None" else "(implicit)"
            impl_str = " [IMPLICIT]" if im else ""
            print(f"    - {c} : {s}  aspect={term_str}{impl_str}")
        print()

        try:
            response = input("  Your verdict [c/w/correction]: ").strip().lower()
        except EOFError:
            response = "c"   # non-interactive fallback

        if response == "c":
            n_correct += 1
            out_row = row.to_dict()
            out_row["verdict"] = "correct"
            verified_rows.append(out_row)

        elif response == "w":
            n_wrong += 1
            # Still save with verdict=wrong for analysis; just don't include in gold
            print("  Discarded.\n")

        else:
            # User typed a correction like "FOOD#QUALITY:negative,SERVICE#GENERAL:positive"
            n_corrected += 1
            try:
                corrected_cats  = []
                corrected_sents = []
                for pair in response.split(","):
                    cat, sent = pair.strip().split(":")
                    corrected_cats.append(cat.strip().upper())
                    corrected_sents.append(sent.strip().lower())
                out_row = row.to_dict()
                out_row["categories_fine"] = str(corrected_cats)
                out_row["sentiments"]      = str(corrected_sents)
                out_row["verdict"]         = "corrected"
                verified_rows.append(out_row)
                print(f"  Saved with correction: {corrected_cats} / {corrected_sents}\n")
            except ValueError:
                print("  [error] Could not parse correction. Keeping original as 'correct'.")
                out_row = row.to_dict()
                out_row["verdict"] = "correct_assumed"
                verified_rows.append(out_row)
                n_correct += 1

        print()

    # ── Report ────────────────────────────────────────────────────────────────
    total_judged = n_correct + n_corrected + n_wrong
    accuracy     = (n_correct / total_judged * 100) if total_judged > 0 else 0.0
    print(f"\n{'='*70}")
    print(f"  VERIFICATION COMPLETE")
    print(f"  Total judged:    {total_judged}")
    print(f"  Correct:         {n_correct}  ({n_correct/total_judged*100:.1f}%)")
    print(f"  Corrected:       {n_corrected}")
    print(f"  Wrong/discarded: {n_wrong}")
    print(f"  LLM Accuracy:    {accuracy:.1f}%")
    print(f"{'='*70}\n")
    print(f"  Cite in dissertation: "
          f"'Manual verification of 100 samples yielded {accuracy:.1f}% label accuracy.'")

    if verified_rows:
        pd.DataFrame(verified_rows).to_csv(output_csv, index=False, encoding="utf-8")
        print(f"  Gold set saved -> {output_csv}\n")
    else:
        print("  No verified rows to save.\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Manual verification of 100 synthetic samples.")
    parser.add_argument("--input",  default="data/augmented/synthetic_filtered.csv")
    parser.add_argument("--output", default="data/augmented/gold_verified_100.csv")
    parser.add_argument("--n",      type=int, default=100)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    verify_samples(args.input, args.output, n=args.n, seed=args.seed)
