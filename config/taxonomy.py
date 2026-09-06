"""
Aspect taxonomy definitions and mapping utilities.

Two source taxonomies:
  - SemEval/ACOS: 13 fine-grained Entity#Attribute categories
  - MAMS: 8 coarse categories (different naming convention)

Both map to a unified set of 5 coarse categories used at inference time.
"""

from typing import Optional

# ── SemEval / ACOS fine-grained categories ─────────────────────────────────
SEMEVAL_CATEGORIES = [
    "FOOD#QUALITY",
    "FOOD#PRICES",
    "FOOD#STYLE_OPTIONS",
    "SERVICE#GENERAL",
    "AMBIENCE#GENERAL",
    "RESTAURANT#PRICES",
    "RESTAURANT#GENERAL",
    "RESTAURANT#MISCELLANEOUS",
    "DRINKS#QUALITY",
    "DRINKS#PRICES",
    "DRINKS#STYLE_OPTIONS",
    "LOCATION#GENERAL",
    # SemEval-2014 uses these coarser ones; we keep them for completeness
    "ANECDOTES/MISCELLANEOUS",
]

# ── MAMS categories ─────────────────────────────────────────────────────────
MAMS_CATEGORIES = [
    "food",
    "service",
    "staff",
    "price",
    "ambience",
    "place",
    "miscellaneous",
    "drinks",
]

# ── Unified coarse categories (used at inference / in the prototype) ────────
COARSE_CATEGORIES = [
    "Food",
    "Service",
    "Ambience",
    "Price",
    "General",
]

# ── Sentiment labels (lowercase for consistency across datasets) ────────────
SENTIMENT_LABELS = ["positive", "negative", "neutral"]
SENTIMENT_TO_ID  = {s: i for i, s in enumerate(SENTIMENT_LABELS)}
ID_TO_SENTIMENT  = {i: s for i, s in enumerate(SENTIMENT_LABELS)}

# ── Fine → Coarse mappings ──────────────────────────────────────────────────

_SEMEVAL_TO_COARSE: dict[str, str] = {
    "FOOD#QUALITY":            "Food",
    "FOOD#PRICES":             "Price",
    "FOOD#STYLE_OPTIONS":      "Food",
    "SERVICE#GENERAL":         "Service",
    "AMBIENCE#GENERAL":        "Ambience",
    "RESTAURANT#PRICES":       "Price",
    "RESTAURANT#GENERAL":      "General",
    "RESTAURANT#MISCELLANEOUS":"General",
    "DRINKS#QUALITY":          "Food",
    "DRINKS#PRICES":           "Price",
    "DRINKS#STYLE_OPTIONS":    "Food",
    "LOCATION#GENERAL":        "General",
    "ANECDOTES/MISCELLANEOUS": "General",
}

_MAMS_TO_COARSE: dict[str, str] = {
    "food":          "Food",
    "service":       "Service",
    "staff":         "Service",
    "price":         "Price",
    "ambience":      "Ambience",
    "place":         "General",
    "miscellaneous": "General",
    "drinks":        "Food",
}

# Direct coarse label pass-through (canonical 5-category taxonomy)
_COARSE_DIRECT: dict[str, str] = {
    "Food":     "Food",
    "Service":  "Service",
    "Ambience": "Ambience",
    "Price":    "Price",
    "Quantity": "Price",   # Redirect legacy Quantity annotations to Price (value for money)
    "General":  "General",
}

# Combined lookup (normalised to upper for case-insensitive matching)
_ALL_TO_COARSE: dict[str, str] = {
    **{k.upper(): v for k, v in _SEMEVAL_TO_COARSE.items()},
    **{k.upper(): v for k, v in _MAMS_TO_COARSE.items()},
    **{k.upper(): v for k, v in _COARSE_DIRECT.items()},
}

# ── Normalisation helpers ───────────────────────────────────────────────────

def normalise_category(raw: str) -> str:
    """
    Normalise a raw category string to a canonical form.
    SemEval categories → upper-case (FOOD#QUALITY).
    MAMS categories → lower-case (food).
    New coarse categories → title-case (Food, Service, …).
    Unknown categories are returned as-is after stripping whitespace.
    """
    stripped = raw.strip()
    # Check direct coarse label first (case-insensitive)
    title = stripped.title()
    if title in _COARSE_DIRECT:
        return _COARSE_DIRECT[title]
    upper = stripped.upper()
    if upper in _SEMEVAL_TO_COARSE:
        return upper
    lower = stripped.lower()
    if lower in _MAMS_TO_COARSE:
        return lower
    return stripped


def map_to_coarse(category: str) -> str:
    """
    Map any fine-grained category (SemEval or MAMS) to one of the 5
    coarse categories. Returns 'General' for unknown categories.
    """
    return _ALL_TO_COARSE.get(category.upper(), "General")


def normalise_sentiment(raw: str) -> str:
    """Normalise sentiment string to lowercase label."""
    mapping = {
        "positive": "positive",
        "pos":      "positive",
        "Positive": "positive",
        "negative": "negative",
        "neg":      "negative",
        "Negative": "negative",
        "neutral":  "neutral",
        "neu":      "neutral",
        "Neutral":  "neutral",
        "conflict": "neutral",   # SemEval-2014 has 'conflict'; treat as neutral
    }
    return mapping.get(raw.strip(), raw.strip().lower())


# ── Category ↔ ID (for multi-label classification) ─────────────────────────
# We use the full combined set of categories for the classifier.
ALL_FINE_CATEGORIES = sorted(set(SEMEVAL_CATEGORIES) | {c.upper() for c in MAMS_CATEGORIES})

CATEGORY_TO_ID: dict[str, int] = {c: i for i, c in enumerate(ALL_FINE_CATEGORIES)}
ID_TO_CATEGORY: dict[int, str] = {i: c for i, c in enumerate(ALL_FINE_CATEGORIES)}
NUM_CATEGORIES = len(ALL_FINE_CATEGORIES)

COARSE_TO_ID: dict[str, int] = {c: i for i, c in enumerate(COARSE_CATEGORIES)}
ID_TO_COARSE: dict[int, str] = {i: c for i, c in enumerate(COARSE_CATEGORIES)}
NUM_COARSE = len(COARSE_CATEGORIES)


def get_category_id(category: str) -> Optional[int]:
    norm = normalise_category(category)
    return CATEGORY_TO_ID.get(norm)


def get_sentiment_id(sentiment: str) -> int:
    return SENTIMENT_TO_ID.get(normalise_sentiment(sentiment), 2)  # default neutral
