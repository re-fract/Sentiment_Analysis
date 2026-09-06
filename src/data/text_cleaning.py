"""
src/data/text_cleaning.py

Minimal text cleaning for restaurant review text.

Design philosophy:
  - Keep the text as close to natural language as possible.
  - BERT/RoBERTa tokenizers handle casing, punctuation, and contractions natively
    so we do NOT lowercase, lemmatise, or strip punctuation.
  - We only fix encoding artefacts and collapse whitespace.

Functions
---------
clean_text(text)  : single string → cleaned string
clean_series(s)   : pandas Series → cleaned Series (vectorised)
"""

from __future__ import annotations

import html
import re
import unicodedata

import pandas as pd

# Common encoding artefacts seen in scraped restaurant reviews
_HTML_ENTITIES = re.compile(r"&[a-zA-Z]+;|&#\d+;|&#x[0-9a-fA-F]+;")
_MULTI_SPACE    = re.compile(r"[ \t]+")
_MULTI_NEWLINE  = re.compile(r"\n{2,}")
_CONTROL_CHARS  = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> str:
    """
    Lightly clean a review string.

    Steps (in order):
      1. Unescape HTML entities  (&amp; → &,  &#39; → ')
      2. Strip control characters (non-printable ASCII)
      3. Normalise unicode to NFC  (composited form → consistent)
      4. Collapse multiple spaces/tabs to a single space
      5. Collapse 2+ newlines to one  (preserves paragraph breaks)
      6. Strip leading / trailing whitespace

    NOT done (intentional):
      - Lowercasing      — BERT is case-sensitive; casing carries sentiment signal
      - Punctuation removal — exclamations/question marks matter for sentiment
      - Stopword removal  — transformers handle this in attention
      - Spelling correction — would change the original text domain
    """
    if not isinstance(text, str):
        return ""

    # 1. HTML entities
    text = html.unescape(text)

    # 2. Remaining HTML-like entities (e.g. &nbsp; not caught by html.unescape)
    text = _HTML_ENTITIES.sub(" ", text)

    # 3. Control characters
    text = _CONTROL_CHARS.sub("", text)

    # 4. Unicode normalisation (NFC)
    text = unicodedata.normalize("NFC", text)

    # 5. Collapse whitespace
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n", text)

    # 6. Strip
    return text.strip()


def clean_series(s: pd.Series) -> pd.Series:
    """Vectorised version of clean_text for a pandas Series."""
    return s.astype(str).map(clean_text)
