"""
src/augmentation/llm_client.py

Cerebras API client for GPT-OSS-120B (generator) and Gemma-4-31B (validator).
Uses the OpenAI-compatible endpoint.

Rate limits (per model):
    5 req/min  |  30,000 tokens/min  |  1,000,000 tokens/day
We operate at 4 req/min to leave headroom.
"""

from __future__ import annotations

import os
import time
import json
import threading
from typing import Any
from dataclasses import dataclass, field

from src.utils.logging import get_logger

try:
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:
    pass

log = get_logger("llm_client")

# ── Lazy import so openai is optional until actually used ─────────────────────
def _get_openai():
    try:
        from openai import OpenAI
        return OpenAI
    except ImportError:
        raise ImportError("Run: pip install openai>=1.0")


@dataclass
class RateLimiter:
    """Token-bucket rate limiter shared across threads."""
    max_rpm: int = 4
    max_tpm: int = 28_000   # slightly under 30K hard limit

    _lock:        threading.Lock   = field(default_factory=threading.Lock, repr=False)
    _req_times:   list             = field(default_factory=list, repr=False)
    _tokens_used: int              = field(default=0, repr=False)
    _window_start:float            = field(default_factory=time.time, repr=False)

    def wait_for_request(self, estimated_tokens: int = 1000) -> None:
        with self._lock:
            now = time.time()
            # Slide the 60-second window
            self._req_times = [t for t in self._req_times if now - t < 60]
            if now - self._window_start >= 60:
                self._tokens_used = 0
                self._window_start = now

            # Wait if hitting req/min limit
            while len(self._req_times) >= self.max_rpm:
                oldest = self._req_times[0]
                sleep_for = 60 - (now - oldest) + 0.5
                if sleep_for > 0:
                    log.debug(f"Rate limit: sleeping {sleep_for:.1f}s")
                    time.sleep(sleep_for)
                now = time.time()
                self._req_times = [t for t in self._req_times if now - t < 60]

            # Wait if hitting token/min limit
            if self._tokens_used + estimated_tokens > self.max_tpm:
                sleep_for = 60 - (now - self._window_start) + 1
                if sleep_for > 0:
                    log.debug(f"Token budget: sleeping {sleep_for:.1f}s")
                    time.sleep(sleep_for)
                self._tokens_used = 0
                self._window_start = time.time()

            self._req_times.append(time.time())
            self._tokens_used += estimated_tokens


class CerebrasClient:
    """
    Thin wrapper around the Cerebras OpenAI-compatible API.

    Usage:
        client = CerebrasClient(model="gpt-oss-120b", api_key="...")
        response = client.generate(prompt, max_tokens=512)
    """

    BASE_URL = "https://api.cerebras.ai/v1"

    def __init__(
        self,
        model:        str,
        api_key:      str,
        base_url:     str | None = None,
        rate_limiter: RateLimiter | None = None,
        max_retries:  int = 3,
        retry_backoff: float = 2.0,
    ):
        self.model         = model
        self.max_retries   = max_retries
        self.retry_backoff = retry_backoff
        self._limiter      = rate_limiter or RateLimiter()
        self.base_url      = base_url or os.environ.get("OPENAI_BASE_URL", self.BASE_URL)

        OpenAI = _get_openai()
        self._client = OpenAI(
            api_key  = api_key,
            base_url = self.base_url,
        )

    def generate(
        self,
        prompt:     str,
        system:     str = "You are a helpful assistant that outputs valid JSON.",
        max_tokens: int = 1024,
        temperature:float = 0.7,
    ) -> str:
        """
        Send a prompt and return the response text.
        Retries on failure with exponential backoff and 429 rate limit extraction.
        """
        # Conservative token estimation: ~1.6 tokens per word + system + max output
        est_tokens = int(len(prompt.split()) * 1.6) + int(len(system.split()) * 1.6) + max_tokens
        self._limiter.wait_for_request(estimated_tokens=est_tokens)

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model       = self.model,
                    messages    = [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": prompt},
                    ],
                    max_tokens  = max_tokens,
                    temperature = temperature,
                )
                choice = response.choices[0]
                msg = choice.message
                content = (msg.content or "").strip()
                if not content:
                    raise ValueError("Empty response received from model. Retrying...")
                return content
            except Exception as exc:
                exc_str = str(exc)
                # Check if Groq gave an explicit retry wait time
                wait = self.retry_backoff ** attempt
                if "try again in" in exc_str.lower():
                    import re
                    match = re.search(r"try again in (\d+(\.\d+)?)s", exc_str, re.IGNORECASE)
                    if match:
                        wait = float(match.group(1)) + 1.0
                    else:
                        wait = max(wait, 6.0)
                elif "429" in exc_str or "rate limit" in exc_str.lower():
                    wait = max(wait, 5.0)

                log.warning(f"[{self.model}] Attempt {attempt}/{self.max_retries} "
                            f"failed: {exc}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
                if attempt == self.max_retries:
                    log.error(f"[{self.model}] All retries exhausted.")
                    raise

    def generate_json(
        self,
        prompt:     str,
        system:     str = "You are a helpful assistant that outputs valid JSON only.",
        max_tokens: int = 1024,
        temperature:float = 0.7,
    ) -> Any:
        """
        Like generate() but parses JSON. Strips markdown fences if present.
        Returns None if JSON parsing fails.
        """
        raw = self.generate(prompt, system=system,
                            max_tokens=max_tokens, temperature=temperature)
        # Strip ```json ... ``` fences
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            log.debug(f"JSON parse failed. Raw: {text[:200]}")
            return None


def make_clients(cfg: dict) -> tuple["CerebrasClient", "CerebrasClient"]:
    """
    Return (generator, validator) clients.

    Priority for every setting:
      1. config.yaml  cerebras.*  (always wins if set)
      2. Provider defaults inferred from API key prefix

    API key resolution order:
      GROQ_API_KEY env var  →  CEREBRAS_API_KEY env var  →  config.yaml cerebras.api_key
    """
    cb  = cfg.get("cerebras", {})
    key = (
        os.environ.get("GROQ_API_KEY", "")
        or os.environ.get("CEREBRAS_API_KEY", "")
        or cb.get("api_key", "")
    )
    if not key or key == "YOUR_API_KEY_HERE" or key == "YOUR_CEREBRAS_API_KEY_HERE":
        raise ValueError(
            "API key not found. Set GROQ_API_KEY or CEREBRAS_API_KEY in .env, "
            "or set cerebras.api_key in config/config.yaml."
        )

    is_groq = key.startswith("gsk_") or bool(os.environ.get("GROQ_API_KEY"))

    # ── Resolve settings: config.yaml first, then provider defaults ──────────
    if cb.get("base_url"):
        base_url = cb["base_url"]
    elif is_groq:
        base_url = "https://api.groq.com/openai/v1"
    else:
        base_url = "https://api.cerebras.ai/v1"

    if cb.get("generator_model"):
        gen_model = cb["generator_model"]
    elif is_groq:
        gen_model = "llama-3.1-70b-versatile"
    else:
        gen_model = "gpt-oss-120b"

    if cb.get("validator_model"):
        val_model = cb["validator_model"]
    else:
        val_model = gen_model   # reuse same model for validation

    # Rate limits: config wins, then provider defaults
    if "rate_limit_rpm" in cb:
        rpm = cb["rate_limit_rpm"]
    elif is_groq:
        rpm = 25                # Groq: 30 RPM hard limit; 25 for safety
    else:
        rpm = 4                 # Cerebras: 5 RPM hard limit; 4 for safety

    if "token_budget_tpm" in cb:
        tpm = cb["token_budget_tpm"]
    elif is_groq:
        tpm = 120_000           # Groq: 131,072 TPM
    else:
        tpm = 28_000            # Cerebras: 30,000 TPM

    log.info(
        f"LLM backend: {'Groq' if is_groq else 'Cerebras'}  "
        f"base_url={base_url}  "
        f"generator={gen_model}  validator={val_model}  "
        f"RPM={rpm}  TPM={tpm}"
    )

    limiter_gen = RateLimiter(max_rpm=rpm, max_tpm=tpm)
    limiter_val = RateLimiter(max_rpm=rpm, max_tpm=tpm)

    generator = CerebrasClient(
        model        = gen_model,
        api_key      = key,
        base_url     = base_url,
        rate_limiter = limiter_gen,
        max_retries  = cb.get("max_retries", 3),
        retry_backoff= cb.get("retry_backoff_base", 2.0),
    )
    validator = CerebrasClient(
        model        = val_model,
        api_key      = key,
        base_url     = base_url,
        rate_limiter = limiter_val,
        max_retries  = cb.get("max_retries", 3),
        retry_backoff= cb.get("retry_backoff_base", 2.0),
    )
    return generator, validator
