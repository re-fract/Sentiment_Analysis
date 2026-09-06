"""
scripts/run_daily_labeling.py — Daily automated Yelp review labeling with 5 Groq keys.

Features:
- Rotates through a pool of Groq API keys as each hits its daily limit (RPD / TPD).
- Automatically halts when all keys in the pool are exhausted for the day.
- Resumable: skips already-labeled reviews; appends to output JSONL immediately.
- Safe for GitHub Actions: stops cleanly well before any 6-hour job limits.
- Supports continuous daemon mode (--daemon) for always-on VPS setups.

Usage:
    # Single daily run (standard for GitHub Actions / Cron):
    python scripts/run_daily_labeling.py --target-count 5000

    # Continuous daemon mode (for an always-on VPS):
    python scripts/run_daily_labeling.py --daemon
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, List, Optional, Set

# Add workspace root to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(WORKSPACE_ROOT / ".env")
except ImportError:
    pass

from src.augmentation.key_rotator import GroqKeyRotator, AllKeysExhaustedError
from src.augmentation.prompt_templates import (
    build_batch_label_prompt,
    build_label_prompt,
    BATCH_LABEL_SYSTEM,
    LABEL_SYSTEM,
)
from src.utils.io import append_jsonl, load_jsonl
from src.utils.logging import get_logger

log = get_logger("daily_labeling")


def _get_utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _seconds_until_next_utc_reset(hour: int = 0, minute: int = 5) -> float:
    """Calculate seconds until the next 00:05 UTC (Groq quota reset + buffer)."""
    now = _get_utc_now()
    reset_today = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if now >= reset_today:
        reset_time = reset_today + timedelta(days=1)
    else:
        reset_time = reset_today
    return max(60.0, (reset_time - now).total_seconds())


def load_existing_done_ids(output_file: Path) -> Set[str]:
    """Load set of already labeled review_ids from output file."""
    done_ids: Set[str] = set()
    if output_file.exists():
        for record in load_jsonl(output_file):
            rev_id = record.get("review_id")
            if rev_id:
                done_ids.add(rev_id)
    return done_ids


def parse_batch_response(raw: Any, expected_count: int) -> List[List[dict]]:
    """Robustly parse batch response list from LLM output."""
    if raw is None:
        return []

    if isinstance(raw, dict):
        if "reviews" in raw and isinstance(raw["reviews"], list):
            raw = raw["reviews"]
        elif "labels" in raw and expected_count == 1:
            lbls = raw.get("labels")
            return [lbls if isinstance(lbls, list) else []]
        else:
            parsed = []
            for k in range(1, expected_count + 1):
                val = (
                    raw.get(str(k))
                    or raw.get(f"review_{k}")
                    or raw.get(f"Review {k}")
                    or raw.get(f"review{k}")
                )
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


def run_labeling_pass(
    rotator: GroqKeyRotator,
    input_file: Path,
    output_file: Path,
    target_count: int = 5000,
    batch_size: int = 2,
    max_runtime_seconds: float = 4 * 3600,
) -> tuple[int, bool]:
    """
    Execute a labeling session until:
      - target_count is reached, OR
      - all keys are exhausted for today, OR
      - max_runtime_seconds is reached.

    Returns:
      (newly_labeled_count, all_keys_exhausted_flag)
    """
    start_time = time.time()
    done_ids = load_existing_done_ids(output_file)
    initial_done = len(done_ids)

    log.info(
        f"[Labeling Pass] Existing labeled items: {initial_done}/{target_count}. "
        f"Output file: {output_file}"
    )

    if initial_done >= target_count:
        log.info(f"[Labeling Pass] Target of {target_count} items already achieved. Nothing to do.")
        return 0, False

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Collect pending reviews
    pending: List[dict] = []
    with input_file.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rev = json.loads(line)
            except json.JSONDecodeError:
                continue

            rev_id = rev.get("review_id")
            if not rev_id or rev_id in done_ids:
                continue
            if not rev.get("text", "").strip():
                continue

            pending.append(rev)
            if len(pending) + initial_done >= target_count:
                break

    if not pending:
        log.info("[Labeling Pass] No pending un-labeled reviews found in input file.")
        return 0, False

    log.info(
        f"[Labeling Pass] Starting batch labeling for {len(pending)} pending reviews "
        f"(batch_size={batch_size})..."
    )

    newly_labeled = 0
    all_exhausted = False

    for i in range(0, len(pending), batch_size):
        # Watchdog: Exit cleanly if approaching max execution time
        elapsed = time.time() - start_time
        if elapsed >= max_runtime_seconds:
            log.warning(
                f"[Labeling Pass] Runtime safety limit reached ({elapsed / 3600:.1f} hours). "
                f"Halting pass cleanly for now."
            )
            break

        batch = pending[i : i + batch_size]
        batch_success = False

        if batch_size > 1 and len(batch) > 1:
            prompt_input = [{"index": idx + 1, "text": r["text"]} for idx, r in enumerate(batch)]
            prompt = build_batch_label_prompt(prompt_input)

            try:
                raw = rotator.generate_json(prompt, system=BATCH_LABEL_SYSTEM, max_tokens=2500)
                parsed_batch = parse_batch_response(raw, len(batch))
                if parsed_batch and len(parsed_batch) == len(batch):
                    batch_success = True
                    for idx, rev in enumerate(batch):
                        labels = parsed_batch[idx]
                        record = {
                            "mode": "label",
                            "review_id": rev["review_id"],
                            "business_id": rev.get("business_id", ""),
                            "stars": rev.get("stars"),
                            "text": rev["text"],
                            "labels": labels if isinstance(labels, list) else [],
                            "source": "yelp",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                        append_jsonl(record, output_file)
                        done_ids.add(rev["review_id"])
                        newly_labeled += 1
            except AllKeysExhaustedError:
                log.warning("[Labeling Pass] All API keys exhausted during batch request.")
                all_exhausted = True
                break
            except Exception as exc:
                log.warning(f"[Labeling Pass] Batch request error: {exc}")

        # Fallback to single review labeling if batch failed
        if not batch_success:
            for rev in batch:
                rev_id = rev["review_id"]
                if rev_id in done_ids:
                    continue

                prompt_single = build_label_prompt(rev["text"])
                try:
                    raw_single = rotator.generate_json(prompt_single, system=LABEL_SYSTEM, max_tokens=1500)
                    labels = raw_single.get("labels", []) if isinstance(raw_single, dict) else []
                    record = {
                        "mode": "label",
                        "review_id": rev_id,
                        "business_id": rev.get("business_id", ""),
                        "stars": rev.get("stars"),
                        "text": rev["text"],
                        "labels": labels if isinstance(labels, list) else [],
                        "source": "yelp",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    append_jsonl(record, output_file)
                    done_ids.add(rev_id)
                    newly_labeled += 1
                except AllKeysExhaustedError:
                    log.warning("[Labeling Pass] All API keys exhausted during single review request.")
                    all_exhausted = True
                    break
                except Exception as exc:
                    log.warning(f"[Labeling Pass] Single request failed for {rev_id}: {exc}")

            if all_exhausted:
                break

        total_now = initial_done + newly_labeled
        if newly_labeled % 10 == 0 or total_now >= target_count:
            log.info(f"Progress: {total_now}/{target_count} labeled (+{newly_labeled} this session)")

        if total_now >= target_count:
            log.info(f"🎉 Completed target: {total_now}/{target_count} reviews labeled!")
            break

    log.info(
        f"[Labeling Pass Finished] Labeled {newly_labeled} reviews this run. "
        f"Total completed: {initial_done + newly_labeled}/{target_count}."
    )
    return newly_labeled, all_exhausted


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated Daily Yelp Labeler with 5 Groq Keys")
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Input sampled reviews JSONL path (default: data/raw/yelp_restaurants_5000.jsonl or yelp_restaurants_sampled.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/augmented/yelp_labeled_5000.jsonl",
        help="Output labeled JSONL path",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=5000,
        help="Total reviews to label (default: 5000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Reviews per API call (default: 2)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.3-70b-versatile",
        help="Groq model (default: llama-3.3-70b-versatile)",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=3.5,
        help="Maximum hours to run before safe stop (default: 3.5 hrs, well under GitHub 6h limit)",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run continuously in background, sleeping until 00:05 UTC when daily limits are hit",
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default="data/augmented/key_state.json",
        help="State file tracking key exhaustion for today",
    )
    args = parser.parse_args()

    # Determine input file path
    if args.input:
        input_path = Path(args.input)
    else:
        candidate_5000 = Path("data/raw/yelp_restaurants_5000.jsonl")
        candidate_sampled = Path("data/raw/yelp_restaurants_sampled.jsonl")
        if candidate_5000.exists():
            input_path = candidate_5000
        elif candidate_sampled.exists():
            input_path = candidate_sampled
        else:
            input_path = candidate_5000

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rotator = GroqKeyRotator(
        model=args.model,
        state_file=args.state_file,
    )

    if not args.daemon:
        # Standard Single Run (GitHub Actions or Daily Cron)
        _, _ = run_labeling_pass(
            rotator=rotator,
            input_file=input_path,
            output_file=output_path,
            target_count=args.target_count,
            batch_size=args.batch_size,
            max_runtime_seconds=args.max_hours * 3600,
        )
        log.info("Daily run completed successfully. Exiting code 0.")
        sys.exit(0)

    else:
        # Daemon Mode (Always-On VPS)
        log.info("Starting in Continuous DAEMON mode...")
        while True:
            done_count = len(load_existing_done_ids(output_path))
            if done_count >= args.target_count:
                log.info(f"Target of {args.target_count} reviews reached! Daemon exiting.")
                break

            _, all_exhausted = run_labeling_pass(
                rotator=rotator,
                input_file=input_path,
                output_file=output_path,
                target_count=args.target_count,
                batch_size=args.batch_size,
                max_runtime_seconds=args.max_hours * 3600,
            )

            done_count = len(load_existing_done_ids(output_path))
            if done_count >= args.target_count:
                log.info(f"Target of {args.target_count} reached! Daemon exiting.")
                break

            sleep_seconds = _seconds_until_next_utc_reset(hour=0, minute=5)
            log.info(
                f"[Daemon] Pausing for {sleep_seconds / 3600:.2f} hours until Groq daily reset (00:05 UTC)..."
            )
            time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
