"""
src/augmentation/key_rotator.py

Manages a pool of Groq API keys with automatic failover and daily limit rotation.
- Differentiates short-term rate limits (RPM / TPM: sleeps briefly and retries)
  from daily quota exhaustion (RPD / TPD: rotates immediately to the next key).
- Gracefully stops when all keys in the pool have reached their daily limit.
"""

from __future__ import annotations

import os
import re
import time
import json
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Set

from src.utils.logging import get_logger

try:
    from dotenv import load_dotenv
    # Load .env from workspace root
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()
except ImportError:
    pass

log = get_logger("key_rotator")


class AllKeysExhaustedError(Exception):
    """Raised when every API key in the pool has hit its daily limit."""
    pass


def _get_current_utc_date() -> str:
    """Return YYYY-MM-DD in UTC (matching Groq's daily reset timezone)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def load_api_keys_from_env() -> List[str]:
    """
    Extract all available Groq API keys from environment variables.
    Checks:
      1. GROQ_API_KEYS (comma-separated or newline-separated string)
      2. GROQ_API_KEY_1, GROQ_API_KEY_2, ..., GROQ_API_KEY_10
      3. GROQ_API_KEY (single key fallback)
    """
    keys: List[str] = []

    # 1. Comma / newline separated list
    raw_list = os.environ.get("GROQ_API_KEYS", "").strip()
    if raw_list:
        for k in re.split(r"[,\s\n]+", raw_list):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)

    # 2. Numbered keys (GROQ_API_KEY_1, GROQ_API_KEY_2, ...)
    for i in range(1, 20):
        k = os.environ.get(f"GROQ_API_KEY_{i}", "").strip()
        if k and k not in keys:
            keys.append(k)

    # 3. Standard fallback
    single_key = os.environ.get("GROQ_API_KEY", "").strip()
    if single_key and single_key not in keys:
        keys.append(single_key)

    return keys


def is_daily_limit_error(exc_str: str) -> tuple[bool, str]:
    """
    Check whether an exception string corresponds to a daily quota limit (RPD / TPD)
    rather than a short-term per-minute rate limit (RPM / TPM).
    """
    exc_lower = exc_str.lower()

    # Direct Groq quota keywords
    if "requests per day" in exc_lower or "rpd" in exc_lower:
        return True, "Requests Per Day (RPD) limit reached"
    if "tokens per day" in exc_lower or "tpd" in exc_lower:
        return True, "Tokens Per Day (TPD) limit reached"
    if "daily limit" in exc_lower or "quota exceeded" in exc_lower:
        return True, "Daily quota limit reached"

    # Check if Groq tells us to wait an absurd amount of time (> 5 minutes / hours)
    match = re.search(r"try again in (\d+(\.\d+)?)s", exc_lower)
    if match:
        seconds = float(match.group(1))
        if seconds > 300:  # More than 5 minutes implies daily reset wait
            return True, f"Long wait required ({seconds:.0f}s)"

    # Check if wait time is in minutes/hours: e.g. "try again in 14h23m"
    if re.search(r"try again in \d+h", exc_lower) or re.search(r"try again in \d+m", exc_lower):
        return True, "Multi-minute/hour reset wait required"

    return False, ""


def extract_retry_after(exc_str: str, default_wait: float = 3.0) -> float:
    """Extract wait duration from standard Groq / OpenAI rate limit messages."""
    match = re.search(r"try again in (\d+(\.\d+)?)s", exc_str, re.IGNORECASE)
    if match:
        return float(match.group(1)) + 0.5
    return default_wait


class GroqKeyRotator:
    """
    Rotating client for OpenAI-compatible Groq endpoints.
    Switches keys transparently when daily quota limits are encountered.
    """

    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        model: str = "openai/gpt-oss-120b",
        base_url: Optional[str] = None,
        max_transient_retries: int = 5,
        base_backoff: float = 2.0,
        state_file: Optional[Path | str] = None,
    ):
        loaded_keys = api_keys or load_api_keys_from_env()
        if not loaded_keys:
            raise ValueError(
                "No Groq API keys found. Set GROQ_API_KEYS (comma-separated) or "
                "GROQ_API_KEY_1 ... GROQ_API_KEY_5 in your environment or .env file."
            )

        self.api_keys = loaded_keys
        self.model = model
        self.base_url = base_url or os.environ.get("GROQ_BASE_URL", self.BASE_URL)
        self.max_transient_retries = max_transient_retries
        self.base_backoff = base_backoff
        self.state_file = Path(state_file) if state_file else None

        self.current_index: int = 0
        self.exhausted_keys: Set[int] = set()
        self.current_date: str = _get_current_utc_date()

        # Load existing state if available for today
        self._load_state()

        # Ensure active key is valid
        self._select_next_valid_key()

        log.info(
            f"[GroqKeyRotator] Initialized with {len(self.api_keys)} key(s). "
            f"Active: Key #{self.current_index + 1} "
            f"({len(self.exhausted_keys)} exhausted today). Model: {self.model}"
        )

    def _get_client_for_key(self, key: str):
        try:
            from openai import OpenAI
            return OpenAI(api_key=key, base_url=self.base_url)
        except ImportError:
            raise ImportError("Run: pip install openai>=1.0")

    def _check_and_refresh_date(self) -> None:
        """If UTC date has advanced, reset the daily exhaustion set."""
        now_date = _get_current_utc_date()
        if now_date != self.current_date:
            log.info(f"[GroqKeyRotator] UTC Date changed from {self.current_date} to {now_date}. Resetting key quota status.")
            self.current_date = now_date
            self.exhausted_keys.clear()
            self._save_state()

    def _select_next_valid_key(self) -> bool:
        """Find the next key that has not been marked exhausted today."""
        self._check_and_refresh_date()
        if len(self.exhausted_keys) >= len(self.api_keys):
            return False

        for offset in range(len(self.api_keys)):
            idx = (self.current_index + offset) % len(self.api_keys)
            if idx not in self.exhausted_keys:
                self.current_index = idx
                return True
        return False

    def mark_current_key_exhausted(self, reason: str) -> None:
        """Mark the active key as exhausted for the current UTC day and rotate."""
        self.exhausted_keys.add(self.current_index)
        log.warning(
            f"[GroqKeyRotator] Key #{self.current_index + 1} exhausted for {self.current_date}: {reason}. "
            f"({len(self.exhausted_keys)}/{len(self.api_keys)} exhausted)"
        )
        self._save_state()

        has_next = self._select_next_valid_key()
        if has_next:
            log.info(f"[GroqKeyRotator] Switched to Key #{self.current_index + 1}.")
        else:
            log.warning(f"[GroqKeyRotator] ALL {len(self.api_keys)} keys exhausted for today ({self.current_date}).")

    def is_all_exhausted(self) -> bool:
        self._check_and_refresh_date()
        return len(self.exhausted_keys) >= len(self.api_keys)

    def _save_state(self) -> None:
        if not self.state_file:
            return
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "date": self.current_date,
                "exhausted_indices": list(self.exhausted_keys),
                "total_keys": len(self.api_keys),
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            with self.state_file.open("w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            log.debug(f"Failed to save state file: {e}")

    def _load_state(self) -> None:
        if not self.state_file or not self.state_file.exists():
            return
        try:
            with self.state_file.open("r", encoding="utf-8") as f:
                state = json.load(f)
            if state.get("date") == self.current_date:
                self.exhausted_keys = set(state.get("exhausted_indices", []))
                log.info(f"[GroqKeyRotator] Restored state for {self.current_date}: {len(self.exhausted_keys)} keys already exhausted.")
        except Exception as e:
            log.debug(f"Failed to load state file: {e}")

    def generate(
        self,
        prompt: str,
        system: str = "You are a helpful assistant that outputs valid JSON.",
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> str:
        """
        Execute completion with intelligent key rotation and transient retry handling.
        """
        while True:
            self._check_and_refresh_date()
            if self.is_all_exhausted():
                raise AllKeysExhaustedError(
                    f"All {len(self.api_keys)} Groq API keys have reached their daily limit for {self.current_date}."
                )

            active_key = self.api_keys[self.current_index]
            client = self._get_client_for_key(active_key)
            transient_attempt = 0

            while transient_attempt < self.max_transient_retries:
                try:
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=max_tokens,
                        temperature=temperature,
                        response_format={"type": "json_object"},
                    )
                    content = (response.choices[0].message.content or "").strip()
                    if not content:
                        raise ValueError("Empty response received from Groq.")
                    return content

                except Exception as exc:
                    exc_str = str(exc)

                    # 1. Check if this is a Daily Quota limit (RPD / TPD)
                    is_daily, reason = is_daily_limit_error(exc_str)
                    if is_daily:
                        self.mark_current_key_exhausted(reason)
                        # Break out of transient loop to try next key
                        break

                    # 2. Check if this is a transient rate limit (RPM / TPM)
                    transient_attempt += 1
                    if "429" in exc_str or "rate limit" in exc_str.lower():
                        wait_sec = extract_retry_after(exc_str, default_wait=self.base_backoff ** transient_attempt)
                        log.info(
                            f"[GroqKeyRotator] Rate limit on Key #{self.current_index + 1} (RPM/TPM). "
                            f"Sleeping {wait_sec:.1f}s before retry ({transient_attempt}/{self.max_transient_retries})..."
                        )
                        time.sleep(wait_sec)
                    elif transient_attempt < self.max_transient_retries:
                        wait_sec = self.base_backoff ** transient_attempt
                        log.warning(
                            f"[GroqKeyRotator] Request error on Key #{self.current_index + 1}: {exc_str[:120]}. "
                            f"Retrying in {wait_sec:.1f}s..."
                        )
                        time.sleep(wait_sec)
                    else:
                        log.error(f"[GroqKeyRotator] Transient retries exhausted on Key #{self.current_index + 1}: {exc_str}")
                        # Rotate to next key in case of persistent network/key anomaly
                        self.mark_current_key_exhausted(f"Transient retries exhausted: {exc_str[:80]}")
                        break

    def generate_json(
        self,
        prompt: str,
        system: str = "You are a helpful assistant that outputs valid JSON only.",
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> Any:
        """
        Call generate() and parse as JSON, stripping markdown code blocks if present.
        """
        raw = self.generate(prompt, system=system, max_tokens=max_tokens, temperature=temperature)
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            log.warning(f"[GroqKeyRotator] Failed to parse JSON: {exc}. Raw text: {text[:200]}")
            return None
