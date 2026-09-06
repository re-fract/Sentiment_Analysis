"""Unified data schema shared by all parsers and dataset classes."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import ast
import pandas as pd


@dataclass
class AspectAnnotation:
    category_fine: str          # e.g. "FOOD#QUALITY" or "food" (source-normalised)
    category_coarse: str        # e.g. "Food" — mapped via taxonomy
    sentiment: str              # "positive" | "negative" | "neutral"
    aspect_term: Optional[str]  # surface word(s), None if implicit
    is_implicit: bool           # True when no explicit aspect term in text


@dataclass
class UnifiedSample:
    text: str
    aspects: List[AspectAnnotation]
    sample_id: str
    source: str   # "semeval14" | "semeval15" | "semeval16" | "mams" | "acos" | "synthetic"
    split: str    # "train" | "val" | "test" | "dev"

    # ── Serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Flatten to a dict suitable for a single CSV row."""
        return {
            "sample_id":         self.sample_id,
            "source":            self.source,
            "split":             self.split,
            "text":              self.text,
            # Store lists as Python-repr strings so pd.read_csv round-trips
            "categories_fine":   str([a.category_fine   for a in self.aspects]),
            "categories_coarse": str([a.category_coarse for a in self.aspects]),
            "sentiments":        str([a.sentiment        for a in self.aspects]),
            "aspect_terms":      str([a.aspect_term      for a in self.aspects]),
            "is_implicit_flags": str([a.is_implicit      for a in self.aspects]),
            "num_aspects":       len(self.aspects),
            "has_implicit":      any(a.is_implicit for a in self.aspects),
            "has_conflict":      len({a.sentiment for a in self.aspects}) > 1,
        }

    @classmethod
    def from_dict(cls, row: dict) -> "UnifiedSample":
        """Reconstruct from a CSV row (inverse of to_dict)."""
        fine     = ast.literal_eval(row["categories_fine"])
        coarse   = ast.literal_eval(row["categories_coarse"])
        sents    = ast.literal_eval(row["sentiments"])
        terms    = ast.literal_eval(row["aspect_terms"])
        implicit = ast.literal_eval(row["is_implicit_flags"])
        aspects  = [
            AspectAnnotation(f, c, s, t if t != "None" else None, i)
            for f, c, s, t, i in zip(fine, coarse, sents, terms, implicit)
        ]
        return cls(
            text=row["text"],
            aspects=aspects,
            sample_id=row["sample_id"],
            source=row["source"],
            split=row["split"],
        )


# ── CSV helpers ───────────────────────────────────────────────────────────────

def samples_to_dataframe(samples: List[UnifiedSample]) -> pd.DataFrame:
    return pd.DataFrame([s.to_dict() for s in samples])


def dataframe_to_samples(df: pd.DataFrame) -> List[UnifiedSample]:
    return [UnifiedSample.from_dict(row) for _, row in df.iterrows()]
