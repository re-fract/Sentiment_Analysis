"""
src/augmentation/prompt_templates.py

Prompt templates for two augmentation modes:

  Mode A — GENERATE: LLM writes a new review + labels from scratch.
            Used to fill specific (category, sentiment, num_aspects) buckets.

  Mode B — LABEL:    LLM assigns ABSA labels to an existing Yelp review.
            Used to create silver-standard annotations on real text.
"""

from __future__ import annotations

from typing import List
import random


# ── Taxonomy context (injected into every prompt) ─────────────────────────────

TAXONOMY_BLOCK = """
ASPECT CATEGORIES (use EXACTLY these 5 names, nothing else):
  Food      — taste, quality, freshness, ingredients, culinary style, beverages, recipe balance (e.g. sauce ratio, toppings)
  Service   — staff friendliness, wait time, attentiveness, hospitality, order accuracy, delivery
  Ambience  — atmosphere, décor, noise level, cleanliness, seating comfort, views, physical space
  Price     — cost, affordability, bill, deals/discounts, AND portion sizes / value for money (amount received relative to cost)
  General   — overall dining experience, location, accessibility, recommendation, would/would not return

SENTIMENTS: positive, negative, neutral

CRITICAL TAXONOMY BOUNDARIES:
- Portions & Value -> Price: Mentions of portion size, serving size, stingy/generous amounts, or feeling ripped-off/full relative to price belong to Price (e.g. "tiny portion for $15", "generous serving", "portions are huge" -> category: Price).
- Ingredients & Recipe Balance -> Food: Specific food items, ingredients, or recipe balance (e.g. "not too much cheese", "plenty of pepperoni", "bacon was undercooked", "sauce was bland") belong to Food, NEVER Price.
- Implicit Aspects: Set "aspect_term": null and "is_implicit": true ONLY when an opinion is clearly expressed about a category without naming the entity (e.g. "took 45 minutes to get a table" implies Service without saying "service" or "waiter"). If the entity IS named (e.g. "service was slow", "waiter was rude"), set "aspect_term" to that exact noun and "is_implicit": false.
""".strip()


# ── Mode A: GENERATE ──────────────────────────────────────────────────────────

GENERATE_SYSTEM = (
    "You are a dataset construction assistant for NLP research. "
    "Your task is to write realistic, natural-sounding restaurant reviews "
    "and annotate them with aspect-based sentiment labels using the 5-category taxonomy. "
    "Output ONLY valid JSON, no explanation."
)

GENERATE_FEW_SHOT_EXAMPLES = [
    # single aspect, explicit, negative
    {
        "text": "The pasta was completely overcooked and tasteless.",
        "labels": [
            {"aspect_term": "pasta", "category": "Food",
             "sentiment": "negative", "is_implicit": False}
        ]
    },
    # dual aspect, mixed sentiment
    {
        "text": "Our waiter was incredibly attentive, but the prices are outrageous for what you get.",
        "labels": [
            {"aspect_term": "waiter", "category": "Service",
             "sentiment": "positive", "is_implicit": False},
            {"aspect_term": "prices", "category": "Price",
             "sentiment": "negative", "is_implicit": False}
        ]
    },
    # implicit aspect
    {
        "text": "We waited nearly 45 minutes just to get our appetisers.",
        "labels": [
            {"aspect_term": None, "category": "Service",
             "sentiment": "negative", "is_implicit": True}
        ]
    },
    # triple aspect
    {
        "text": "Lovely ambience with dim lighting and soft music. "
                "The cocktail menu is creative and reasonably priced, "
                "though the food took a while to arrive.",
        "labels": [
            {"aspect_term": "ambience",     "category": "Ambience",
             "sentiment": "positive", "is_implicit": False},
            {"aspect_term": "cocktail menu","category": "Food",
             "sentiment": "positive", "is_implicit": False},
            {"aspect_term": None,           "category": "Service",
             "sentiment": "negative", "is_implicit": True}
        ]
    },
    # neutral
    {
        "text": "The location is convenient, right in the middle of downtown.",
        "labels": [
            {"aspect_term": "location", "category": "General",
             "sentiment": "neutral", "is_implicit": False}
        ]
    },
    # Portion / value example (mapped to Price)
    {
        "text": "For the price I expected more on the plate — the portions are tiny.",
        "labels": [
            {"aspect_term": "portions", "category": "Price",
             "sentiment": "negative", "is_implicit": False},
            {"aspect_term": "price", "category": "Price",
             "sentiment": "negative", "is_implicit": False}
        ]
    },
]


def build_generate_prompt(
    target_categories: List[str],
    target_sentiments: List[str],
    include_implicit:  bool = False,
    n_samples:         int  = 5,
    seed_examples:     List[dict] | None = None,
) -> str:
    """
    Build a prompt asking the LLM to generate `n_samples` reviews
    that cover the given (category, sentiment) targets.

    Args:
        target_categories: e.g. ["Ambience", "Service"]
        target_sentiments:  e.g. ["positive",         "negative"]
                            Must be same length as target_categories.
        include_implicit:  Whether to ask for at least one implicit aspect.
        n_samples:         How many distinct reviews to generate.
        seed_examples:     Real training examples to inject as additional few-shots.
    """
    assert len(target_categories) == len(target_sentiments)

    targets_str = "\n".join(
        f"  - category: {cat}, sentiment: {sent}"
        for cat, sent in zip(target_categories, target_sentiments)
    )
    implicit_note = (
        "At least one aspect in each review should be IMPLICIT "
        "(aspect_term = null). " if include_implicit else ""
    )

    # Build few-shot block from fixed examples + optional real examples
    examples = list(GENERATE_FEW_SHOT_EXAMPLES)
    if seed_examples:
        examples = seed_examples[:3] + examples[:2]   # mix real + fixed
    random.shuffle(examples)

    import json
    examples_str = "\n".join(
        f"Example {i+1}:\n{json.dumps(ex, indent=2)}"
        for i, ex in enumerate(examples[:4])
    )

    prompt = f"""You are writing training data for an Aspect-Based Sentiment Analysis (ABSA) system.

{TAXONOMY_BLOCK}

TASK: Generate {n_samples} different, realistic restaurant review sentences. Each review MUST express opinions about these specific aspect-sentiment pairs:
{targets_str}

Requirements:
- Each sentence must be between 10 and 60 words.
- Use natural, varied language (avoid starting all sentences with "The").
- {implicit_note}Reviews should be realistic — they could appear on Yelp or TripAdvisor.
- Include diverse scenarios: date nights, family meals, business lunches, takeaway, etc.

Here are a few annotated examples for reference:
{examples_str}

Now generate {n_samples} NEW reviews (different from the examples above) as a JSON array:
[
  {{
    "text": "...",
    "labels": [
      {{"aspect_term": "...", "category": "...", "sentiment": "...", "is_implicit": false}},
      ...
    ]
  }},
  ...
]

Output ONLY the JSON array, no other text."""

    return prompt


# ── Mode B: LABEL (Yelp reviews) ──────────────────────────────────────────────

LABEL_SYSTEM = (
    "You are an expert annotator for Aspect-Based Sentiment Analysis (ABSA). "
    "Given a restaurant review, extract all mentioned aspect-sentiment pairs "
    "using the provided 5-category taxonomy. "
    "Extract minimal head-noun aspect terms without sentiment adjectives. "
    "Evaluate pragmatic author intent (account for sarcasm, hyperbole, and rhetorical praise). "
    "Output ONLY valid JSON."
)

LABEL_FEW_SHOT = """Examples:

Review: "The lamb chops were outstanding, but the service was painfully slow and the dining room felt dated."
Output:
{"labels": [
  {"aspect_term": "lamb chops",  "category": "Food",     "sentiment": "positive", "is_implicit": false},
  {"aspect_term": "service",     "category": "Service",  "sentiment": "negative", "is_implicit": false},
  {"aspect_term": "dining room", "category": "Ambience", "sentiment": "negative", "is_implicit": false}
]}

Review: "The carbonara was delicious, but $22 for such a tiny portion is ridiculous."
Output:
{"labels": [
  {"aspect_term": "carbonara", "category": "Food",  "sentiment": "positive", "is_implicit": false},
  {"aspect_term": "portion",   "category": "Price", "sentiment": "negative", "is_implicit": false},
  {"aspect_term": "$22",       "category": "Price", "sentiment": "negative", "is_implicit": false}
]}

Review: "We sat for over 40 minutes before anyone even brought us water, though the patio views were gorgeous."
Output:
{"labels": [
  {"aspect_term": null,        "category": "Service",  "sentiment": "negative", "is_implicit": true},
  {"aspect_term": "views",     "category": "Ambience", "sentiment": "positive", "is_implicit": false}
]}

Review: "Do not miss this spot! You will eat until you explode and their homemade ravioli is to die for."
Output:
{"labels": [
  {"aspect_term": "spot",    "category": "General", "sentiment": "positive", "is_implicit": false},
  {"aspect_term": "ravioli", "category": "Food",    "sentiment": "positive", "is_implicit": false}
]}"""


def build_label_prompt(review_text: str) -> str:
    """
    Build a prompt to label an existing (Yelp) review with ABSA annotations.
    """
    prompt = f"""You are annotating a restaurant review for Aspect-Based Sentiment Analysis.

{TAXONOMY_BLOCK}

{LABEL_FEW_SHOT}

Rules:
- Minimal Aspect Term: Extract ONLY the concise head noun or minimal compound noun (e.g. "portion" NOT "such a tiny portion", "ravioli" NOT "their homemade ravioli", "waiter" NOT "our rude waiter"). NEVER include opinion modifiers or quantifiers in aspect_term.
- Portions & Value belong to Price: Mentions of portion sizes, serving sizes, and cost complaints are categorized under Price (value for money).
- Ingredients & Recipe Balance belong to Food: Specific ingredients, toppings, or recipe balance (e.g. "not too much cheese", "extra garlic") belong to Food.
- Do NOT over-segment dishes: Do not split a single ordered dish into constituent ingredients unless the reviewer expresses contrasting sentiments about individual parts.
- Implicit Aspects: Use "is_implicit": true and "aspect_term": null ONLY when an aspect is completely unstated (e.g. wait times without saying "wait" or "service"). If the entity IS named, set "aspect_term" to that word and "is_implicit": false.
- Sarcasm & Hyperbole: Discern true pragmatic meaning (e.g. "eat until you explode" or "so good it should be illegal" is positive praise).
- If the review does not discuss any of the taxonomy aspects, output: {{"labels": []}}

Review to annotate:
"{review_text}"

Output ONLY the JSON object:"""

    return prompt


BATCH_LABEL_SYSTEM = (
    "You are an expert annotator for Aspect-Based Sentiment Analysis (ABSA). "
    "Given multiple restaurant reviews, extract all mentioned aspect-sentiment pairs for each review "
    "using the provided 5-category taxonomy. "
    "Extract minimal head-noun aspect terms without sentiment adjectives. "
    "Evaluate pragmatic author intent (account for sarcasm, hyperbole, and rhetorical praise). "
    "Output ONLY a valid JSON array."
)

BATCH_LABEL_FEW_SHOT = """Example input:
Review 1: "The lamb chops were outstanding, but the service was painfully slow and the dining room felt dated."
Review 2: "Great place, huge portions for the price and yummy burgers!"
Review 3: "We waited 45 minutes just for water, but the patio views were lovely."

Example output:
[
  {
    "review_index": 1,
    "labels": [
      {"aspect_term": "lamb chops",  "category": "Food",     "sentiment": "positive", "is_implicit": false},
      {"aspect_term": "service",     "category": "Service",  "sentiment": "negative", "is_implicit": false},
      {"aspect_term": "dining room", "category": "Ambience", "sentiment": "negative", "is_implicit": false}
    ]
  },
  {
    "review_index": 2,
    "labels": [
      {"aspect_term": "place",    "category": "General", "sentiment": "positive", "is_implicit": false},
      {"aspect_term": "portions", "category": "Price",   "sentiment": "positive", "is_implicit": false},
      {"aspect_term": "burgers",  "category": "Food",    "sentiment": "positive", "is_implicit": false}
    ]
  },
  {
    "review_index": 3,
    "labels": [
      {"aspect_term": null,    "category": "Service",  "sentiment": "negative", "is_implicit": true},
      {"aspect_term": "views", "category": "Ambience", "sentiment": "positive", "is_implicit": false}
    ]
  }
]"""


def build_batch_label_prompt(reviews: List[dict]) -> str:
    """
    Build a prompt to label multiple (e.g. 2) reviews in one LLM call.
    Each item in reviews is {'index': 1, 'text': '...'}.
    """
    reviews_formatted = "\n\n".join(
        [f'Review {r["index"]}:\n"{r["text"]}"' for r in reviews]
    )

    prompt = f"""You are annotating restaurant reviews for Aspect-Based Sentiment Analysis.

{TAXONOMY_BLOCK}

{BATCH_LABEL_FEW_SHOT}

Rules:
- Annotate each review independently in the JSON list maintaining the review_index.
- Minimal Aspect Term: Extract ONLY the concise head noun or minimal compound noun (e.g. "portions" NOT "huge portions", "burgers" NOT "yummy burgers"). NEVER include opinion modifiers or quantifiers in aspect_term.
- Portions & Value belong to Price: Mentions of portion sizes, serving sizes, and cost complaints are categorized under Price (value for money).
- Ingredients & Recipe Balance belong to Food: Specific ingredients, toppings, or recipe balance (e.g. "not too much cheese", "extra garlic") belong to Food.
- Do NOT over-segment dishes: Do not split a single ordered dish into constituent ingredients unless the reviewer expresses contrasting sentiments about individual parts.
- Implicit Aspects: Use "is_implicit": true and "aspect_term": null ONLY when an aspect is completely unstated (e.g. wait times without saying "wait" or "service"). If the entity IS named, set "aspect_term" to that word and "is_implicit": false.
- Sarcasm & Hyperbole: Discern true pragmatic meaning (e.g. "eat until you explode" or "so good it should be illegal" is positive praise).
- If a review does not discuss any of the taxonomy aspects, output: {{"review_index": ..., "labels": []}}

Reviews to annotate:
{reviews_formatted}

Output ONLY the JSON list of objects:"""

    return prompt


# ── Mode C: VALIDATE ──────────────────────────────────────────────────────────

VALIDATE_SYSTEM = (
    "You are a quality-control reviewer for ABSA annotations. "
    "Given a restaurant review and its existing labels, verify whether "
    "each label is correct. Output ONLY valid JSON."
)


def build_validate_prompt(review_text: str, labels: List[dict]) -> str:
    """
    Build a prompt asking the validator model (Gemma-4-31B) to re-predict
    the labels for a review. Used for cross-model consistency checking.
    """
    import json
    prompt = f"""You are checking annotations for a restaurant review.

{TAXONOMY_BLOCK}

Review: "{review_text}"

Proposed labels:
{json.dumps(labels, indent=2)}

Task: Re-annotate this review independently. List the correct aspect-sentiment pairs.
Then compare with the proposed labels and flag any disagreements.

Output as JSON:
{{
  "your_labels": [
    {{"aspect_term": "...", "category": "...", "sentiment": "...", "is_implicit": false}}
  ],
  "agreement": true or false,
  "issues": ["describe any disagreements here, or empty list if all agree"]
}}

Output ONLY the JSON object:"""

    return prompt
