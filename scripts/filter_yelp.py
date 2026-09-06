"""
scripts/filter_yelp.py — Filter Yelp dataset to restaurant reviews only.

Run ONCE before the augmentation pipeline:
    .venv/Scripts/python scripts/filter_yelp.py

Outputs:
    data/raw/yelp_restaurants_sampled.jsonl   <- ~3,000 sampled restaurant reviews

Strategy:
  1. Stream yelp_academic_dataset_business.json, collect business_ids where
     'categories' contains a restaurant keyword.
  2. Stream yelp_academic_dataset_review.json, keep reviews from those business_ids.
  3. Apply quality filters (length, star diversity, no spam).
  4. Stratified sample: ~600 per star bucket (1,2,3,4,5) => ~3,000 total.
     This ensures sentiment diversity before LLM labeling.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.io import load_config
from src.utils.logging import get_logger

log = get_logger("filter_yelp")

RESTAURANT_KEYWORDS = {
    "restaurant", "restaurants", "food", "pizza", "burger", "burgers",
    "sushi", "steakhouse", "bistro", "diner", "brasserie", "cafe",
    "cafes", "coffee", "bakery", "bakeries", "bar", "bars", "pub",
    "pubs", "taco", "tacos", "noodle", "seafood", "thai", "chinese",
    "italian", "mexican", "indian", "bbq", "barbecue", "buffet",
    "sandwiches", "salad", "brunch", "breakfast", "lunch", "dinner",
}

MIN_WORDS      = 20    # min words to keep a review
MAX_WORDS      = 120   # max words (long reviews are hard to label)
DEFAULT_TARGET = 5000  # total reviews to sample
RANDOM_SEED    = 42


def is_restaurant_business(categories_str: str | None) -> bool:
    if not categories_str:
        return False
    cats_lower = categories_str.lower()
    return any(kw in cats_lower for kw in RESTAURANT_KEYWORDS)


def filter_yelp(
    cfg: dict,
    target_total: int = DEFAULT_TARGET,
    output_path: str | Path = "data/raw/yelp_restaurants_5000.jsonl",
    force: bool = False,
) -> Path:
    raw_dir = Path("data/raw")
    biz_file = raw_dir / "yelp_academic_dataset_business.json"
    rev_file = raw_dir / "yelp_academic_dataset_review.json"
    out_file = Path(output_path)

    if out_file.exists() and not force:
        log.info(f"Output already exists: {out_file}. Delete it or pass force=True to re-run.")
        return out_file

    target_per_star = max(1, target_total // 5)

    # ── Step 1: Collect restaurant business IDs ───────────────────────────────
    log.info("Step 1: Collecting restaurant business IDs...")
    restaurant_biz_ids: set[str] = set()
    with biz_file.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                biz = json.loads(line)
            except json.JSONDecodeError:
                continue
            if is_restaurant_business(biz.get("categories")):
                restaurant_biz_ids.add(biz["business_id"])
            if (i + 1) % 50_000 == 0:
                log.info(f"  Processed {i+1:,} businesses, found {len(restaurant_biz_ids):,} restaurants")

    log.info(f"Found {len(restaurant_biz_ids):,} restaurant businesses")

    # ── Step 2: Stream reviews, filter by restaurant + quality ────────────────
    log.info("Step 2: Filtering reviews...")
    # Bucket by star rating for stratified sampling
    buckets: dict[int, list[dict]] = defaultdict(list)
    total_seen = 0
    total_kept = 0

    with rev_file.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rev = json.loads(line)
            except json.JSONDecodeError:
                continue

            total_seen += 1
            if total_seen % 500_000 == 0:
                log.info(f"  Seen {total_seen:,} reviews, kept {total_kept:,}")

            # Must be from a restaurant
            if rev.get("business_id") not in restaurant_biz_ids:
                continue

            text  = rev.get("text", "").strip().replace("\n", " ")
            words = len(text.split())
            if words < MIN_WORDS or words > MAX_WORDS:
                continue

            # Must have a star rating
            stars = rev.get("stars")
            if stars not in (1.0, 2.0, 3.0, 4.0, 5.0):
                continue
            star_bucket = int(stars)

            # Already have enough in this bucket?  Keep some extras for shuffle
            if len(buckets[star_bucket]) >= target_per_star * 3:
                continue

            buckets[star_bucket].append({
                "review_id":   rev["review_id"],
                "business_id": rev["business_id"],
                "stars":       stars,
                "text":        text,
            })
            total_kept += 1

    log.info(f"Bucket sizes before sampling: { {k: len(v) for k, v in buckets.items()} }")

    # ── Step 3: Stratified sample ─────────────────────────────────────────────
    rng = random.Random(RANDOM_SEED)
    sampled: list[dict] = []
    for star, reviews in buckets.items():
        rng.shuffle(reviews)
        sampled.extend(reviews[:target_per_star])

    rng.shuffle(sampled)
    log.info(f"Sampled {len(sampled):,} reviews total")

    # ── Step 4: Write output ──────────────────────────────────────────────────
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as fh:
        for rev in sampled:
            fh.write(json.dumps(rev, ensure_ascii=False) + "\n")

    log.info(f"Saved -> {out_file}")
    log.info("Done.")
    return out_file


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Filter Yelp dataset to sampled restaurant reviews")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET, help="Total reviews to sample (default: 5000)")
    parser.add_argument("--output", type=str, default="data/raw/yelp_restaurants_5000.jsonl", help="Output file path")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing output file")
    args = parser.parse_args()

    cfg = load_config()
    filter_yelp(cfg, target_total=args.target, output_path=args.output, force=args.force)
