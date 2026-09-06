"""
Parsers for all four raw dataset formats present in data/raw/.

  PyABSASegParser  — .seg / .raw files (SemEval 14/15/16 via ABSADatasets)
  MAMSParser       — MAMS-ACSA/raw/*.xml
  ACOSParser       — acos_datasets/504.Restaurant16/*.jsonl
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

import sys, os
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.taxonomy import (
    normalise_category,
    normalise_sentiment,
    map_to_coarse,
)
from src.data.unified_schema import AspectAnnotation, UnifiedSample


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_id(source: str, split: str, index: int) -> str:
    return f"{source}_{split}_{index:05d}"


# ── 1. PyABSASegParser ────────────────────────────────────────────────────────

class PyABSASegParser:
    """
    Parses PyABSA-format .seg / .raw files (SemEval 14 / 15 / 16).

    File format — repeating 3-line blocks:
        <sentence with $T$ placeholder>
        <aspect_term>
        <Positive|Negative|Neutral>

    The placeholder $T$ marks where the aspect term appears in the sentence.
    We restore the full sentence by replacing $T$ with the aspect term.

    NOTE: These files are ATSA-level (aspect-term sentiment).  The aspect term
    is a surface phrase (e.g. "food", "waiters"), not a SemEval category code.
    We store it as category_fine = aspect_term.lower() and map to coarse via
    the MAMS taxonomy (which also uses natural-language terms).
    """

    def __init__(self, source: str):
        self.source = source  # e.g. "semeval14", "semeval15", "semeval16"

    def parse_file(self, path: str | Path, split: str) -> List[UnifiedSample]:
        path = Path(path)
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

        samples: List[UnifiedSample] = []
        idx = 0
        sample_idx = 0

        # Group lines into blocks of 3
        while idx + 2 < len(lines):
            masked_sent = lines[idx].strip()
            aspect_term = lines[idx + 1].strip()
            raw_sentiment = lines[idx + 2].strip()
            idx += 3

            if not masked_sent or not aspect_term or not raw_sentiment:
                continue

            # Restore full sentence
            full_text = masked_sent.replace("$T$", aspect_term)

            sentiment = normalise_sentiment(raw_sentiment)
            cat_fine  = normalise_category(aspect_term.lower())
            cat_coarse = map_to_coarse(cat_fine)

            annotation = AspectAnnotation(
                category_fine=cat_fine,
                category_coarse=cat_coarse,
                sentiment=sentiment,
                aspect_term=aspect_term,
                is_implicit=False,   # ATSA format always has explicit terms
            )

            samples.append(UnifiedSample(
                text=full_text,
                aspects=[annotation],
                sample_id=_make_id(self.source, split, sample_idx),
                source=self.source,
                split=split,
            ))
            sample_idx += 1

        return samples


# ── 2. MAMSParser ─────────────────────────────────────────────────────────────

class MAMSParser:
    """
    Parses MAMS-ACSA/raw/{train,val,test}.xml.

    XML structure:
        <sentences>
          <sentence>
            <text>...</text>
            <aspectCategories>
              <aspectCategory category="food" polarity="positive"/>
              ...
            </aspectCategories>
          </sentence>
        </sentences>

    Every sentence is guaranteed to have >=2 aspects with different sentiments
    (that's the whole point of MAMS).
    """

    SOURCE = "mams"

    def parse_file(self, path: str | Path, split: str) -> List[UnifiedSample]:
        path = Path(path)
        tree = ET.parse(str(path))
        root = tree.getroot()

        samples: List[UnifiedSample] = []

        for sample_idx, sent_el in enumerate(root.findall("sentence")):
            text_el = sent_el.find("text")
            if text_el is None or not text_el.text:
                continue
            text = text_el.text.strip()

            aspects: List[AspectAnnotation] = []
            cats_el = sent_el.find("aspectCategories")
            if cats_el is not None:
                for cat_el in cats_el.findall("aspectCategory"):
                    raw_cat  = cat_el.get("category", "").strip()
                    raw_pol  = cat_el.get("polarity", "").strip()
                    cat_fine = normalise_category(raw_cat)
                    aspects.append(AspectAnnotation(
                        category_fine=cat_fine,
                        category_coarse=map_to_coarse(cat_fine),
                        sentiment=normalise_sentiment(raw_pol),
                        aspect_term=None,   # MAMS ACSA has no term-level annotation
                        is_implicit=False,
                    ))

            if not aspects:
                continue

            samples.append(UnifiedSample(
                text=text,
                aspects=aspects,
                sample_id=_make_id(self.SOURCE, split, sample_idx),
                source=self.SOURCE,
                split=split,
            ))

        return samples


# ── 3. ACOSParser ─────────────────────────────────────────────────────────────

class ACOSParser:
    """
    Parses acos_datasets/504.Restaurant16/*.tsv.jsonl.

    JSONL format (one JSON object per line):
        {
          "text": "...",
          "labels": [
            {
              "aspect":   "food" | "NULL",
              "opinion":  "lousy" | "NULL",
              "polarity": "negative",
              "category": "FOOD#QUALITY"
            },
            ...
          ]
        }

    aspect == "NULL" means the aspect is implicit (not explicitly mentioned).
    """

    SOURCE = "acos"

    def parse_file(self, path: str | Path, split: str) -> List[UnifiedSample]:
        path = Path(path)
        samples: List[UnifiedSample] = []

        with path.open(encoding="utf-8") as fh:
            for sample_idx, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = obj.get("text", "").strip()
                if not text:
                    continue

                aspects: List[AspectAnnotation] = []
                for lbl in obj.get("labels", []):
                    raw_aspect   = lbl.get("aspect", "").strip()
                    raw_category = lbl.get("category", "").strip()
                    raw_polarity = lbl.get("polarity", "").strip()

                    is_implicit  = (raw_aspect.upper() == "NULL")
                    aspect_term  = None if is_implicit else raw_aspect

                    cat_fine  = normalise_category(raw_category)
                    aspects.append(AspectAnnotation(
                        category_fine=cat_fine,
                        category_coarse=map_to_coarse(cat_fine),
                        sentiment=normalise_sentiment(raw_polarity),
                        aspect_term=aspect_term,
                        is_implicit=is_implicit,
                    ))

                if not aspects:
                    continue

                samples.append(UnifiedSample(
                    text=text,
                    aspects=aspects,
                    sample_id=_make_id(self.SOURCE, split, sample_idx),
                    source=self.SOURCE,
                    split=split,
                ))

        return samples
