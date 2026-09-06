"""
src/augmentation/generator.py

Two-mode augmentation orchestrator. Fully resumable — writes to JSONL
after every API call; skips already-completed buckets on restart.

Mode A — GENERATE:
    Reads augmentation_targets.json (from EDA), sends GPT-OSS-120B prompts
    to generate synthetic reviews targeting underrepresented (cat, sentiment)
    buckets.

Mode B — LABEL:
    Reads yelp_restaurants_sampled.jsonl, sends Yelp reviews to GPT-OSS-120B
    for ABSA labeling. Saves raw LLM output alongside the source text.

Both modes write to data/augmented/synthetic_raw.jsonl.
Quality filtering happens separately in quality_filter.py.
"""

from __future__ import annotations

import json
import sys
import random
from pathlib import Path
from typing import List

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.augmentation.llm_client import CerebrasClient, make_clients
from src.augmentation.prompt_templates import (
    build_generate_prompt,
    build_label_prompt,
    build_batch_label_prompt,
    GENERATE_SYSTEM,
    LABEL_SYSTEM,
    BATCH_LABEL_SYSTEM,
)
from src.utils.io import load_config, load_jsonl, append_jsonl
from src.utils.logging import get_logger

log = get_logger("generator")

RAW_OUT = Path("data/augmented/synthetic_raw_v2.jsonl")
SAMPLES_PER_REQUEST = 5    # how many reviews per LLM call in generate mode


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_already_done(mode: str) -> set[str]:
    """Return set of bucket keys already saved to the raw JSONL."""
    done: set[str] = set()
    for record in load_jsonl(RAW_OUT):
        if record.get("mode") == mode:
            done.add(record.get("bucket_key", ""))
    return done


def save_record(record: dict, out_path: Path | str = RAW_OUT) -> None:
    append_jsonl(record, Path(out_path))


# ── Mode A: GENERATE ──────────────────────────────────────────────────────────

def _parse_generate_response(raw_json: Any, bucket_key: str) -> List[dict]:
    """Parse the JSON array or dict returned by the LLM in generate mode."""
    if isinstance(raw_json, dict):
        if "reviews" in raw_json and isinstance(raw_json["reviews"], list):
            raw_json = raw_json["reviews"]
        elif "samples" in raw_json and isinstance(raw_json["samples"], list):
            raw_json = raw_json["samples"]
        elif "text" in raw_json and "labels" in raw_json:
            raw_json = [raw_json]
    if not isinstance(raw_json, list):
        log.debug(f"  [generate] Non-list response for {bucket_key}")
        return []
    results = []
    for item in raw_json:
        if not isinstance(item, dict):
            continue
        text   = item.get("text", "").strip()
        labels = item.get("labels", [])
        if not text or not isinstance(labels, list) or not labels:
            continue
        results.append({"text": text, "labels": labels})
    return results


def run_generate_mode(
    generator:    CerebrasClient,
    targets_path: str = "data/augmented/augmentation_targets.json",
    out_path:     str = "data/augmented/synthetic_generated_raw.jsonl",
    seed_data:    List[dict] | None = None,
) -> None:
    """
    Generate synthetic reviews to fill class-imbalance buckets.

    Each bucket is a (coarse_cat|sentiment) key from augmentation_targets.json.
    Within each bucket we also vary num_aspects (1, 2, 3+) and implicit flags.
    Saves outputs to `out_path`.
    """
    targets_file = Path(targets_path)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not targets_file.exists():
        log.error(f"Targets file not found: {targets_path}. Run notebooks/eda.py first.")
        return

    with targets_file.open() as fh:
        targets: dict[str, int] = json.load(fh)

    # New coarse taxonomy — categories are used directly (no fine-grained mapping)
    coarse_to_fine = {
        "Food":     ["Food"],
        "Service":  ["Service"],
        "Ambience": ["Ambience"],
        "Price":    ["Price"],
        "Quantity": ["Quantity"],
        "General":  ["General"],
    }

    # Count how many samples are already generated per bucket in the target out_file
    bucket_counts: dict[str, int] = {}
    if out_file.exists():
        for record in load_jsonl(out_file):
            if record.get("mode") == "generate" and record.get("bucket_key"):
                bkey = record["bucket_key"]
                bucket_counts[bkey] = bucket_counts.get(bkey, 0) + 1

    total_target_samples = sum(targets.values())
    total_already_done = sum(bucket_counts.get(k, 0) for k in targets)
    log.info(f"[generate] Targets: {len(targets)} buckets ({total_target_samples} samples)  |  "
             f"Already generated: {total_already_done}  -> Saving to: {out_file}")

    total_generated = 0

    for bucket_key, target_count in targets.items():
        have = bucket_counts.get(bucket_key, 0)
        deficit = max(0, target_count - have)
        if deficit <= 0:
            continue

        coarse_cat, sentiment = bucket_key.split("|")
        fine_cats = coarse_to_fine.get(coarse_cat, ["RESTAURANT#GENERAL"])

        # How many requests needed?
        n_requests = max(1, (deficit + SAMPLES_PER_REQUEST - 1) // SAMPLES_PER_REQUEST)
        log.info(f"[generate] Bucket: {bucket_key}  deficit={deficit}  "
                 f"requests={n_requests}")

        bucket_pbar = tqdm(total=deficit, desc=f"Generating {bucket_key}", unit="rev")

        for req_i in range(n_requests):
            # Vary aspect count across requests
            n_aspects   = [1, 1, 2, 2, 3][req_i % 5]
            inc_implicit= (req_i % 3 == 2)

            # For multi-aspect, pick a partner fine-grained category
            if n_aspects > 1:
                cat_list = [random.choice(fine_cats)]
                partner_coarse = random.choice(
                    [c for c in coarse_to_fine if c != coarse_cat])
                partner_fine = random.choice(coarse_to_fine[partner_coarse])
                cat_list.append(partner_fine)
                sent_list = [sentiment, random.choice(
                    ["positive", "negative", "neutral"])]
                if n_aspects > 2:
                    cat_list.append(random.choice(fine_cats))
                    sent_list.append(sentiment)
            else:
                cat_list  = [random.choice(fine_cats)]
                sent_list = [sentiment]

            prompt = build_generate_prompt(
                target_categories = cat_list,
                target_sentiments = sent_list,
                include_implicit  = inc_implicit,
                n_samples         = SAMPLES_PER_REQUEST,
                seed_examples     = seed_data,
            )

            parsed = []
            try:
                raw = generator.generate_json(
                    prompt, system=GENERATE_SYSTEM, max_tokens=3500)
                parsed = _parse_generate_response(raw, bucket_key)
            except Exception as exc:
                log.warning(f"  Request failed: {exc}")

            for sample in parsed:
                sample["mode"]       = "generate"
                sample["bucket_key"] = bucket_key
                sample["source"]     = "synthetic"
                save_record(sample, out_path=out_file)

            bucket_pbar.update(len(parsed))
            total_generated += len(parsed)

        bucket_pbar.close()

    log.info(f"[generate] Total new synthetic samples generated this run: {total_generated}")


def _parse_batch_response(raw: Any, expected_count: int) -> List[List[dict]]:
    """
    Robustly extract label lists for each review in a batch from LLM output.
    """
    if raw is None:
        return []
    
    if isinstance(raw, dict):
        if "reviews" in raw and isinstance(raw["reviews"], list):
            raw = raw["reviews"]
        elif "labels" in raw and expected_count == 1:
            lbls = raw.get("labels")
            return [lbls if isinstance(lbls, list) else []]
        else:
            # Check for keys like '1', '2' or 'review_1'
            parsed = []
            for k in range(1, expected_count + 1):
                val = raw.get(str(k)) or raw.get(f"review_{k}") or raw.get(f"Review {k}") or raw.get(f"review{k}")
                if isinstance(val, dict):
                    parsed.append(val.get("labels", []))
                elif isinstance(val, list):
                    parsed.append(val)
                else:
                    parsed.append([])
            if any(len(p) > 0 for p in parsed):
                return parsed

    if isinstance(raw, list):
        results = []
        for item in raw:
            if isinstance(item, dict):
                lbls = item.get("labels")
                if isinstance(lbls, list):
                    results.append(lbls)
                elif "category" in item:
                    # Item itself is a label
                    results.append([item])
                else:
                    results.append([])
            elif isinstance(item, list):
                results.append(item)
            else:
                results.append([])
        if len(results) == expected_count:
            return results

    return []


def _parse_single_response(raw: Any) -> List[dict] | None:
    """
    Robustly extract labels for a single review from LLM output.
    Returns None if output was completely unparseable.
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        lbls = raw.get("labels")
        if isinstance(lbls, list):
            return lbls
    elif isinstance(raw, list):
        if len(raw) > 0 and isinstance(raw[0], dict):
            if "labels" in raw[0]:
                return raw[0].get("labels", [])
            elif "category" in raw[0]:
                return raw
        elif len(raw) == 0:
            return []
    return None


# ── Mode B: LABEL (Yelp) ──────────────────────────────────────────────────────

def run_label_mode(
    generator:        CerebrasClient,
    yelp_sampled_path:str = "data/raw/yelp_restaurants_sampled.jsonl",
    max_to_label:     int = 2000,
    batch_size:       int = 2,
) -> None:
    """
    Label up to `max_to_label` Yelp reviews with ABSA annotations using batched
    prompting (default 2 reviews/call) and live tqdm progress.

    Already-labeled review_ids (stored in synthetic_raw_v2.jsonl AND the legacy
    synthetic_raw.jsonl) are skipped, guaranteeing completely new reviews.
    """
    yelp_path = Path(yelp_sampled_path)
    if not yelp_path.exists():
        log.error(f"Yelp sampled file not found. Run scripts/filter_yelp.py first.")
        return

    # Which review_ids are already done IN V2 (counts toward target)?
    v2_done_ids: set[str] = set()
    for record in load_jsonl(RAW_OUT):
        if record.get("mode") == "label" and record.get("labels"):
            v2_done_ids.add(record.get("review_id", ""))

    # Also exclude any IDs from the legacy file to get completely new reviews
    excluded_ids: set[str] = set(v2_done_ids)
    legacy_path = Path("data/augmented/synthetic_raw.jsonl")
    if legacy_path.exists():
        n_before = len(excluded_ids)
        for record in load_jsonl(legacy_path):
            if record.get("mode") == "label":
                excluded_ids.add(record.get("review_id", ""))
        log.info(f"[label] Excluding {len(excluded_ids) - n_before} IDs from legacy synthetic_raw.jsonl")

    log.info(f"[label] Already in v2: {len(v2_done_ids)}  |  Target: {max_to_label}  |  Total excluded: {len(excluded_ids)}")

    # done_ids used for candidate filtering
    done_ids = excluded_ids
    remaining = max_to_label - len(v2_done_ids)
    if remaining <= 0:
        log.info("[label] Already reached target in v2. Nothing to do.")
        return

    # Collect pending candidate reviews
    pending_reviews: List[dict] = []
    with yelp_path.open(encoding="utf-8") as fh:
        for line in fh:
            if len(pending_reviews) >= remaining:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rev = json.loads(line)
            except json.JSONDecodeError:
                continue

            rev_id = rev.get("review_id", "")
            if not rev_id or rev_id in done_ids:
                continue
            text = rev.get("text", "").strip()
            if not text:
                continue

            pending_reviews.append(rev)

    if not pending_reviews:
        log.info("[label] No pending reviews to label.")
        return

    log.info(f"[label] Labeling {len(pending_reviews)} reviews (batch_size={batch_size})...")
    labeled_count = 0

    pbar = tqdm(total=len(pending_reviews), desc="Labeling Yelp Reviews", unit="rev")

    # Process in batches
    for i in range(0, len(pending_reviews), batch_size):
        batch = pending_reviews[i : i + batch_size]
        batch_success = False

        if batch_size > 1 and len(batch) > 1:
            prompt_input = [{"index": idx + 1, "text": r["text"]} for idx, r in enumerate(batch)]
            prompt = build_batch_label_prompt(prompt_input)

            raw = None
            try:
                raw = generator.generate_json(
                    prompt, system=BATCH_LABEL_SYSTEM, max_tokens=3500
                )
            except Exception as exc:
                log.warning(f"  Batch request failed: {exc}")

            parsed_batch = _parse_batch_response(raw, len(batch))
            if parsed_batch and len(parsed_batch) == len(batch):
                batch_success = True
                for idx, rev in enumerate(batch):
                    labels = parsed_batch[idx]
                    if isinstance(labels, list) and len(labels) > 0:
                        record = {
                            "mode":      "label",
                            "review_id": rev["review_id"],
                            "stars":     rev.get("stars"),
                            "text":      rev["text"],
                            "labels":    labels,
                            "source":    "yelp",
                        }
                        save_record(record)
                        done_ids.add(rev["review_id"])
                        labeled_count += 1
                    pbar.update(1)

        if not batch_success:
            # Fallback to single review labeling for items in this batch
            for rev in batch:
                rev_id = rev["review_id"]
                if rev_id in done_ids:
                    continue
                prompt_single = build_label_prompt(rev["text"])
                try:
                    raw_single = generator.generate_json(
                        prompt_single, system=LABEL_SYSTEM, max_tokens=2500
                    )
                except Exception as exc:
                    log.warning(f"  Fallback labeling failed for {rev_id}: {exc}")
                    pbar.update(1)
                    continue

                labels = _parse_single_response(raw_single)
                if labels is None:
                    log.warning(f"  Unparseable response for {rev_id}")
                    pbar.update(1)
                    continue

                if isinstance(labels, list) and len(labels) > 0:
                    record = {
                        "mode":      "label",
                        "review_id": rev_id,
                        "stars":     rev.get("stars"),
                        "text":      rev["text"],
                        "labels":    labels,
                        "source":    "yelp",
                    }
                    save_record(record)
                    done_ids.add(rev_id)
                    labeled_count += 1
                pbar.update(1)

    pbar.close()
    log.info(f"[label] Done. Successfully labeled {labeled_count} new reviews this run.")
