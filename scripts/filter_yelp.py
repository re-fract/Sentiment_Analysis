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

import re

log = get_logger("filter_yelp")

RESTAURANT_KEYWORDS = {
    "restaurant", "restaurants", "food", "pizza", "burger", "burgers",
    "sushi", "steakhouse", "bistro", "diner", "brasserie", "cafe",
    "cafes", "coffee", "bakery", "bakeries", "bar", "bars", "pub",
    "pubs", "taco", "tacos", "noodle", "seafood", "thai", "chinese",
    "italian", "mexican", "indian", "bbq", "barbecue", "buffet",
    "sandwiches", "salad", "brunch", "breakfast", "lunch", "dinner",
}

PRICE_RE = re.compile(
    r"\b(price|prices|pricing|expensive|cheap|cost|costs|costly|overpriced|affordable|value|bill|check|dollar|dollars|\$|worth)\b",
    re.IGNORECASE,
)
AMBIENCE_RE = re.compile(
    r"\b(ambience|ambiance|atmosphere|vibe|vibes|decor|interior|music|loud|quiet|noisy|seating|lighting|cozy|spacious|patio|outdoor|crowded|view)\b",
    re.IGNORECASE,
)
BUT_RE = re.compile(r"\bbut\b", re.IGNORECASE)

MIN_WORDS      = 20    # min words to keep a review
MAX_WORDS      = 120   # max words (long reviews are hard to label)
DEFAULT_TARGET = 5000  # total fresh reviews to sample
DEFAULT_RARE   = 1500  # targeted rare 3-star reviews (price, ambience, or 'but')
RANDOM_SEED    = 42


def is_restaurant_business(categories_str: str | None) -> bool:
    if not categories_str:
        return False
    cats_lower = categories_str.lower()
    return any(kw in cats_lower for kw in RESTAURANT_KEYWORDS)


def is_rare_3star_review(stars: float, text: str) -> bool:
    """Check if review is a high-value 3-star review with price, ambience, or 'but'."""
    if stars != 3.0:
        return False
    return bool(PRICE_RE.search(text) or AMBIENCE_RE.search(text) or BUT_RE.search(text))


def filter_yelp(
    cfg: dict,
    target_total: int = DEFAULT_TARGET,
    rare_target: int = DEFAULT_RARE,
    output_path: str | Path = "data/raw/yelp_restaurants_10000.jsonl",
    exclude_files: list[str | Path] | None = None,
    combine_with: str | Path | None = None,
    force: bool = False,
) -> Path:
    raw_dir = Path("data/raw")
    biz_file = raw_dir / "yelp_academic_dataset_business.json"
    rev_file = raw_dir / "yelp_academic_dataset_review.json"
    out_file = Path(output_path)

    if out_file.exists() and not force:
        log.info(f"Output already exists: {out_file}. Delete it or pass force=True to re-run.")
        return out_file

    # Calculate quotas per stratum
    use_rare = rare_target > 0
    rare_3star_target = min(rare_target, target_total) if use_rare else 0
    rem_target = target_total - rare_3star_target

    target_per_other_star: dict[int, int] = {}
    if use_rare:
        base_other = rem_target // 4
        rem_other = rem_target % 4
        for idx, s in enumerate((1, 2, 4, 5)):
            target_per_other_star[s] = base_other + (1 if idx < rem_other else 0)
        log.info(
            f"Target plan: {rare_3star_target:,} rare 3-star reviews + "
            f"{target_per_other_star} for other stars = {target_total:,} fresh reviews"
        )
    else:
        target_per_star = max(1, target_total // 5)
        log.info(f"Target plan: {target_per_star:,} reviews per star (1-5) = {target_total:,} fresh reviews")

    # ── Step 0: Load exclusion IDs ────────────────────────────────────────────
    exclude_ids: set[str] = set()
    if exclude_files:
        for ef in exclude_files:
            ef_path = Path(ef)
            if ef_path.exists():
                count_before = len(exclude_ids)
                with ef_path.open(encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            r_id = record.get("review_id")
                            if r_id:
                                exclude_ids.add(r_id)
                        except json.JSONDecodeError:
                            continue
                log.info(f"Loaded {len(exclude_ids) - count_before:,} exclusion IDs from {ef_path}")
    log.info(f"Total exclusion IDs loaded: {len(exclude_ids):,}")

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
    rare_bucket: list[dict] = []
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

            # Exclude already-sampled or already-labeled reviews
            if rev.get("review_id") in exclude_ids:
                continue

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

            item = {
                "review_id":   rev["review_id"],
                "business_id": rev["business_id"],
                "stars":       stars,
                "text":        text,
            }

            if use_rare and stars == 3.0:
                if is_rare_3star_review(stars, text):
                    if len(rare_bucket) < rare_3star_target * 3:
                        rare_bucket.append(item)
                        total_kept += 1
            else:
                star_bucket = int(stars)
                limit = (target_per_other_star[star_bucket] * 3) if use_rare else (target_per_star * 3)
                if len(buckets[star_bucket]) < limit:
                    buckets[star_bucket].append(item)
                    total_kept += 1

            # Early stop when all required candidate pools reach 3x target
            if use_rare:
                if len(rare_bucket) >= rare_3star_target * 3 and all(
                    len(buckets[s]) >= target_per_other_star[s] * 3 for s in (1, 2, 4, 5)
                ):
                    log.info(
                        f"All candidate pools reached 3x capacity (rare 3-star: {len(rare_bucket):,}). "
                        f"Stopping stream early at review #{total_seen:,}!"
                    )
                    break
            else:
                if all(len(buckets[s]) >= target_per_star * 3 for s in (1, 2, 3, 4, 5)):
                    log.info(
                        f"All 5 star buckets reached candidate capacity ({target_per_star * 3:,} each). "
                        f"Stopping stream early at review #{total_seen:,}!"
                    )
                    break

    log.info(
        f"Candidate pools before sampling: "
        f"{ {'rare_3star': len(rare_bucket)} if use_rare else {} } | "
        f"{ {k: len(v) for k, v in sorted(buckets.items())} }"
    )

    # ── Step 3: Stratified sample ─────────────────────────────────────────────
    rng = random.Random(RANDOM_SEED)
    sampled: list[dict] = []

    if use_rare:
        rng.shuffle(rare_bucket)
        sampled.extend(rare_bucket[:rare_3star_target])
        log.info(f"Sampled {rare_3star_target:,} targeted rare 3-star reviews")

        for s in (1, 2, 4, 5):
            reviews = buckets[s]
            rng.shuffle(reviews)
            sampled.extend(reviews[:target_per_other_star[s]])
            log.info(f"Sampled {target_per_other_star[s]:,} reviews for Star {s}")
    else:
        for star, reviews in buckets.items():
            rng.shuffle(reviews)
            sampled.extend(reviews[:target_per_star])

    rng.shuffle(sampled)
    log.info(f"Sampled {len(sampled):,} fresh reviews total")

    # ── Step 4: Combine with existing if requested, then write output ─────────
    out_file.parent.mkdir(parents=True, exist_ok=True)
    all_output: list[dict] = []

    if combine_with:
        combine_path = Path(combine_with)
        if combine_path.exists():
            log.info(f"Loading existing reviews from {combine_path} to combine...")
            with combine_path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            all_output.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            log.info(f"Loaded {len(all_output):,} base reviews from {combine_path}")
        else:
            log.warning(f"--combine-with path {combine_path} does not exist. Skipping combine.")

    all_output.extend(sampled)
    log.info(
        f"Total reviews to write: {len(all_output):,} "
        f"({len(all_output) - len(sampled):,} base + {len(sampled):,} fresh)"
    )

    with out_file.open("w", encoding="utf-8") as fh:
        for rev in all_output:
            fh.write(json.dumps(rev, ensure_ascii=False) + "\n")

    log.info(f"Saved -> {out_file} ({len(all_output):,} reviews)")
    log.info("Done.")
    return out_file


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Filter Yelp dataset to sampled restaurant reviews")
    parser.add_argument(
        "--target",
        type=int,
        default=DEFAULT_TARGET,
        help="Total fresh reviews to sample (default: 5000)",
    )
    parser.add_argument(
        "--rare-target",
        type=int,
        default=DEFAULT_RARE,
        help="Target count of rare 3-star reviews mentioning price, ambience, or 'but' (default: 1500; set 0 to disable)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/yelp_restaurants_10000.jsonl",
        help="Output file path (default: data/raw/yelp_restaurants_10000.jsonl)",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=["data/raw/yelp_restaurants_5000.jsonl", "data/augmented/yelp_labeled_5000.jsonl"],
        help="Files containing review IDs to strictly exclude from sampling",
    )
    parser.add_argument(
        "--combine-with",
        type=str,
        default="data/raw/yelp_restaurants_5000.jsonl",
        help="Base JSONL file of existing reviews to combine with newly sampled reviews",
    )
    parser.add_argument("--force", action="store_true", help="Force overwrite existing output file")
    args = parser.parse_args()

    cfg = load_config()
    filter_yelp(
        cfg,
        target_total=args.target,
        rare_target=args.rare_target,
        output_path=args.output,
        exclude_files=args.exclude,
        combine_with=args.combine_with,
        force=args.force,
    )
