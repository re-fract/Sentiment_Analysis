"""
notebooks/eda.py — Exploratory Data Analysis for the ABSA pipeline.

Analyses all parsed CSVs in data/processed/ and saves figures to outputs/figures/.
Run with:
    .venv/Scripts/python notebooks/eda.py

Sections:
    1.  Dataset overview & row counts
    2.  Aspect category distribution (fine-grained) per dataset
    3.  Coarse category distribution
    4.  Sentiment distribution overall and per coarse category
    5.  Number of aspects per sentence
    6.  Sentence length distribution
    7.  Aspect co-occurrence matrix (MAMS train)
    8.  Explicit vs implicit aspects (ACOS)
    9.  Class imbalance summary & augmentation targets
    10. Multi-sentiment sentence analysis
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from collections import Counter
from itertools import combinations

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for server runs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.io import load_config
from src.utils.logging import get_logger

log = get_logger("eda")

# ── Aesthetics ────────────────────────────────────────────────────────────────
PALETTE    = "Set2"
BAR_COLOR  = "#4C72B0"
NEG_COLOR  = "#DD4444"
POS_COLOR  = "#44AA77"
NEU_COLOR  = "#CCAA22"
SENT_COLORS = {"positive": POS_COLOR, "negative": NEG_COLOR, "neutral": NEU_COLOR}
plt.rcParams.update({
    "figure.dpi":       150,
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
})

FIG_DIR = Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

def savefig(name: str) -> None:
    path = FIG_DIR / name
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log.info(f"Saved -> {path}")


# ── Data loading ─────────────────────────────────────────────────────────────

def load_all(cfg: dict) -> dict[str, pd.DataFrame]:
    proc = cfg["paths"]["processed"]
    dfs: dict[str, pd.DataFrame] = {}
    for key, path in proc.items():
        p = Path(path)
        if p.exists():
            dfs[key] = pd.read_csv(p, encoding="utf-8")
            log.info(f"Loaded {key:<30s} {len(dfs[key]):>5d} rows")
        else:
            log.warning(f"Missing: {path}")
    return dfs


def parse_list_col(series: pd.Series) -> list[list]:
    """Convert stringified list column back to Python lists."""
    return [ast.literal_eval(v) if isinstance(v, str) else [] for v in series]


def explode_aspects(df: pd.DataFrame) -> pd.DataFrame:
    """Return a flat DataFrame with one row per (sentence, aspect) pair."""
    rows = []
    for _, row in df.iterrows():
        cats_fine   = ast.literal_eval(row["categories_fine"])   if isinstance(row["categories_fine"],   str) else []
        cats_coarse = ast.literal_eval(row["categories_coarse"]) if isinstance(row["categories_coarse"], str) else []
        sents       = ast.literal_eval(row["sentiments"])        if isinstance(row["sentiments"],        str) else []
        impl_flags  = ast.literal_eval(row["is_implicit_flags"]) if isinstance(row["is_implicit_flags"], str) else []
        for i, (cf, cc, s) in enumerate(zip(cats_fine, cats_coarse, sents)):
            impl = impl_flags[i] if i < len(impl_flags) else False
            rows.append({
                "sample_id":      row["sample_id"],
                "source":         row["source"],
                "split":          row.get("split", "?"),
                "text":           row["text"],
                "category_fine":  cf,
                "category_coarse":cc,
                "sentiment":      s,
                "is_implicit":    impl,
                "num_aspects":    row.get("num_aspects", len(cats_fine)),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Dataset Overview
# ─────────────────────────────────────────────────────────────────────────────

def section_1_overview(dfs: dict[str, pd.DataFrame]) -> None:
    log.info("\n" + "="*60)
    log.info("SECTION 1 — DATASET OVERVIEW")
    log.info("="*60)

    rows = []
    for name, df in dfs.items():
        n_sents   = len(df)
        n_aspects = df["num_aspects"].sum() if "num_aspects" in df.columns else 0
        implicit  = int(df["has_implicit"].sum()) if "has_implicit" in df.columns else 0
        conflict  = int(df["has_conflict"].sum()) if "has_conflict" in df.columns else 0
        avg_asp   = df["num_aspects"].mean() if "num_aspects" in df.columns else 0
        rows.append({
            "Dataset":         name,
            "Sentences":       n_sents,
            "Aspect instances":int(n_aspects),
            "Avg asp/sent":    round(avg_asp, 2),
            "Has implicit":    implicit,
            "Multi-sentiment": conflict,
        })

    summary = pd.DataFrame(rows)
    print("\n" + summary.to_string(index=False))

    # Bar chart: sentence counts per dataset
    labels = [r["Dataset"] for r in rows]
    counts = [r["Sentences"] for r in rows]

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = [BAR_COLOR if "train" in l else "#88AACC" for l in labels]
    bars = ax.bar(range(len(labels)), counts, color=colors, edgecolor="white")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Sentence count")
    ax.set_title("Dataset sizes (dark = train splits)", fontweight="bold")
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                str(c), ha="center", va="bottom", fontsize=7)
    savefig("01_dataset_sizes.png")


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Fine-grained aspect distribution
# ─────────────────────────────────────────────────────────────────────────────

def section_2_fine_category_dist(dfs: dict[str, pd.DataFrame]) -> None:
    log.info("\nSECTION 2 — Fine-grained category distribution")

    # combined_train uses raw aspect terms (1500+ unique) — show coarse instead
    # MAMS and ACOS have proper category codes — show those fine-grained
    for name, use_coarse in [("combined_train", True), ("mams_train", False), ("acos_train", False)]:
        df = dfs.get(name)
        if df is None:
            continue
        flat = explode_aspects(df)
        col = "category_coarse" if use_coarse else "category_fine"
        counts = flat[col].value_counts()

        fig, ax = plt.subplots(figsize=(12, max(4, len(counts) * 0.5)))
        counts.sort_values().plot.barh(ax=ax, color=BAR_COLOR, edgecolor="white")
        ax.set_xlabel("Aspect instances")
        level = "coarse (raw terms — see note)" if use_coarse else "fine-grained"
        ax.set_title(f"Category distribution [{level}] — {name}", fontweight="bold")
        for i, v in enumerate(counts.sort_values()):
            ax.text(v + 5, i, str(v), va="center", fontsize=8)
        savefig(f"02_fine_cat_{name}.png")
        log.info(f"  {name}: {len(counts)} unique {col} values")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Coarse category distribution
# ─────────────────────────────────────────────────────────────────────────────

def section_3_coarse_dist(dfs: dict[str, pd.DataFrame]) -> None:
    log.info("\nSECTION 3 — Coarse category distribution")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, key in zip(axes, ["combined_train", "mams_train", "acos_train"]):
        df = dfs.get(key)
        if df is None:
            ax.set_visible(False)
            continue
        flat = explode_aspects(df)
        counts = flat["category_coarse"].value_counts()
        counts.plot.bar(ax=ax, color=sns.color_palette(PALETTE, len(counts)),
                        edgecolor="white")
        ax.set_title(key, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Instances")
        ax.tick_params(axis="x", rotation=30)
        for p in ax.patches:
            ax.text(p.get_x() + p.get_width()/2, p.get_height() + 5,
                    str(int(p.get_height())), ha="center", va="bottom", fontsize=8)
    plt.suptitle("Coarse category distribution", fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig("03_coarse_dist.png")


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Sentiment distribution
# ─────────────────────────────────────────────────────────────────────────────

def section_4_sentiment_dist(dfs: dict[str, pd.DataFrame]) -> None:
    log.info("\nSECTION 4 — Sentiment distribution")

    # Overall sentiment across all training data
    all_train = pd.concat([dfs[k] for k in ["combined_train", "mams_train", "acos_train"]
                           if k in dfs], ignore_index=True)
    flat = explode_aspects(all_train)

    # Overall pie
    sent_counts = flat["sentiment"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = [SENT_COLORS.get(s, "#999999") for s in sent_counts.index]
    axes[0].pie(sent_counts, labels=sent_counts.index, autopct="%1.1f%%",
                colors=colors, startangle=90, wedgeprops=dict(edgecolor="white"))
    axes[0].set_title("Overall sentiment distribution (all training)", fontweight="bold")

    # Per coarse category stacked bar
    pivot = flat.groupby(["category_coarse", "sentiment"]).size().unstack(fill_value=0)
    # Normalise to %
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pivot_pct = pivot_pct.reindex(columns=["positive", "negative", "neutral"],
                                  fill_value=0)
    bar_colors = [POS_COLOR, NEG_COLOR, NEU_COLOR]
    pivot_pct.plot.bar(ax=axes[1], stacked=True, color=bar_colors,
                       edgecolor="white", rot=30)
    axes[1].set_ylabel("% of aspect instances")
    axes[1].set_title("Sentiment % per coarse category (all train)", fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    savefig("04_sentiment_dist.png")
    log.info(f"  Sentiment counts: {dict(sent_counts)}")

    # Per-dataset sentiment breakdown
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, key in zip(axes, ["combined_train", "mams_train", "acos_train"]):
        df = dfs.get(key)
        if df is None:
            ax.set_visible(False)
            continue
        flat_ds = explode_aspects(df)
        counts  = flat_ds["sentiment"].value_counts().reindex(
            ["positive", "negative", "neutral"], fill_value=0)
        ax.bar(counts.index, counts.values,
               color=[SENT_COLORS[s] for s in counts.index], edgecolor="white")
        ax.set_title(key, fontweight="bold")
        ax.set_ylabel("Count")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 5, str(v), ha="center", va="bottom", fontsize=9)
    plt.suptitle("Sentiment distribution per dataset", fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig("04b_sentiment_per_dataset.png")


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Aspects per sentence
# ─────────────────────────────────────────────────────────────────────────────

def section_5_aspects_per_sentence(dfs: dict[str, pd.DataFrame]) -> None:
    log.info("\nSECTION 5 — Aspects per sentence")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, key in zip(axes, ["combined_train", "mams_train", "acos_train"]):
        df = dfs.get(key)
        if df is None:
            ax.set_visible(False)
            continue
        n_asp = df["num_aspects"].clip(upper=6)   # cap at 6+ for display
        counts = n_asp.value_counts().sort_index()
        labels = [str(int(i)) if i < 6 else "6+" for i in counts.index]
        ax.bar(labels, counts.values, color=BAR_COLOR, edgecolor="white")
        ax.set_title(key, fontweight="bold")
        ax.set_xlabel("Number of aspects")
        ax.set_ylabel("Sentences")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 0.5, str(v), ha="center", va="bottom", fontsize=8)
        # Print stats
        log.info(f"  {key}: mean={df['num_aspects'].mean():.2f}  "
                 f"max={df['num_aspects'].max()}  "
                 f"multi={int((df['num_aspects']>1).sum())} "
                 f"({(df['num_aspects']>1).mean()*100:.1f}%)")
    plt.suptitle("Number of aspects per sentence", fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig("05_aspects_per_sentence.png")


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Sentence length distribution
# ─────────────────────────────────────────────────────────────────────────────

def section_6_sentence_length(dfs: dict[str, pd.DataFrame]) -> None:
    log.info("\nSECTION 6 — Sentence length")

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = sns.color_palette(PALETTE, 3)
    for color, key in zip(colors, ["combined_train", "mams_train", "acos_train"]):
        df = dfs.get(key)
        if df is None:
            continue
        lengths = df["text"].str.split().str.len()
        ax.hist(lengths, bins=40, alpha=0.6, label=key, color=color,
                edgecolor="white", density=True)
        log.info(f"  {key}: mean={lengths.mean():.1f} words  "
                 f"median={lengths.median():.1f}  max={lengths.max()}")
    ax.set_xlabel("Words per sentence")
    ax.set_ylabel("Density")
    ax.set_title("Sentence length distribution (all training sets)", fontweight="bold")
    ax.legend()
    ax.axvline(128, color="red", linestyle="--", alpha=0.6, label="BERT max (128 tokens)")
    savefig("06_sentence_length.png")


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — Aspect co-occurrence matrix (MAMS train — richest multi-aspect)
# ─────────────────────────────────────────────────────────────────────────────

def section_7_cooccurrence(dfs: dict[str, pd.DataFrame]) -> None:
    log.info("\nSECTION 7 — Aspect co-occurrence (MAMS train)")

    df = dfs.get("mams_train")
    if df is None:
        return

    # Build co-occurrence matrix over coarse categories
    cats_per_sent = parse_list_col(df["categories_coarse"])
    coarse_cats   = ["Food", "Service", "Ambience", "Price", "General"]
    matrix = pd.DataFrame(0, index=coarse_cats, columns=coarse_cats)

    for cats in cats_per_sent:
        unique = list(set(cats))
        for a, b in combinations(unique, 2):
            if a in coarse_cats and b in coarse_cats:
                matrix.loc[a, b] += 1
                matrix.loc[b, a] += 1
        for a in unique:
            if a in coarse_cats:
                matrix.loc[a, a] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Co-occurrences"})
    ax.set_title("Coarse-category co-occurrence matrix\n(MAMS train)", fontweight="bold")
    plt.tight_layout()
    savefig("07_cooccurrence.png")

    log.info("  Top co-occurring pairs:")
    pairs = [(matrix.loc[a, b], a, b)
             for i, a in enumerate(coarse_cats)
             for j, b in enumerate(coarse_cats) if i < j]
    for cnt, a, b in sorted(pairs, reverse=True)[:5]:
        log.info(f"    {a} & {b}: {cnt}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 8 — Explicit vs Implicit (ACOS)
# ─────────────────────────────────────────────────────────────────────────────

def section_8_implicit(dfs: dict[str, pd.DataFrame]) -> None:
    log.info("\nSECTION 8 — Explicit vs Implicit (ACOS)")

    for key in ["acos_train", "acos_test"]:
        df = dfs.get(key)
        if df is None:
            continue
        flat = explode_aspects(df)
        impl_counts = flat["is_implicit"].value_counts()
        n_explicit = int(impl_counts.get(False, 0))
        n_implicit = int(impl_counts.get(True, 0))
        total = n_explicit + n_implicit
        log.info(f"  {key}: explicit={n_explicit} ({n_explicit/total*100:.1f}%)  "
                 f"implicit={n_implicit} ({n_implicit/total*100:.1f}%)")

    # Plot implicit fraction per coarse category
    df_all = pd.concat([dfs[k] for k in ["acos_train", "acos_test"] if k in dfs])
    flat   = explode_aspects(df_all)

    impl_by_cat = flat.groupby("category_coarse")["is_implicit"].agg(["sum", "count"])
    impl_by_cat["pct_implicit"] = impl_by_cat["sum"] / impl_by_cat["count"] * 100
    impl_by_cat = impl_by_cat.sort_values("pct_implicit", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(impl_by_cat.index, impl_by_cat["pct_implicit"],
                  color=NEG_COLOR, alpha=0.75, edgecolor="white")
    ax.set_ylabel("% implicit")
    ax.set_title("Implicit aspect fraction per coarse category (ACOS train+test)",
                 fontweight="bold")
    ax.set_ylim(0, 100)
    for bar, pct in zip(bars, impl_by_cat["pct_implicit"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    savefig("08_implicit_fraction.png")


# ─────────────────────────────────────────────────────────────────────────────
# Section 9 — Class imbalance & augmentation targets
# ─────────────────────────────────────────────────────────────────────────────

def section_9_imbalance(dfs: dict[str, pd.DataFrame]) -> None:
    log.info("\nSECTION 9 — Class imbalance & augmentation targets")

    # Use combined training data
    train_sources = ["combined_train", "mams_train", "acos_train"]
    all_train = pd.concat([dfs[k] for k in train_sources if k in dfs], ignore_index=True)
    flat = explode_aspects(all_train)

    # Fine-grained category × sentiment pivot
    pivot = flat.groupby(["category_coarse", "sentiment"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(columns=["positive", "negative", "neutral"], fill_value=0)

    log.info("\n  Category x Sentiment counts (all training):")
    print(pivot.to_string())

    # Identify underrepresented (category, sentiment) buckets
    target_per_bucket = 300   # rough target per (coarse_cat, sentiment) cell
    deficit = {}
    for cat in pivot.index:
        for sent in ["positive", "negative", "neutral"]:
            have = int(pivot.loc[cat, sent])
            need = max(0, target_per_bucket - have)
            if need > 0:
                deficit[(cat, sent)] = need

    log.info(f"\n  Augmentation targets (per bucket, target={target_per_bucket}):")
    total_needed = 0
    for (cat, sent), n in sorted(deficit.items(), key=lambda x: -x[1]):
        log.info(f"    {cat:<12s} {sent:<10s}: need {n:>4d} more")
        total_needed += n
    log.info(f"\n  Total synthetic samples needed: {total_needed}")

    # Heatmap of current counts
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Instances"})
    ax.set_title("Aspect-Sentiment instance counts (all training data)", fontweight="bold")
    plt.tight_layout()
    savefig("09_imbalance_heatmap.png")

    # Save augmentation targets to JSON for the generator to use
    import json
    targets = {f"{cat}|{sent}": n for (cat, sent), n in deficit.items()}
    out = Path("data/augmented/augmentation_targets.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        json.dump(targets, fh, indent=2)
    log.info(f"\n  Saved augmentation targets -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 10 — Multi-sentiment analysis
# ─────────────────────────────────────────────────────────────────────────────

def section_10_multi_sentiment(dfs: dict[str, pd.DataFrame]) -> None:
    log.info("\nSECTION 10 — Multi-sentiment sentences")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Left: % multi-sentiment per dataset
    keys = ["combined_train", "mams_train", "acos_train", "mams_test", "acos_test"]
    pcts = []
    for key in keys:
        df = dfs.get(key)
        if df is None:
            pcts.append(0)
            continue
        pct = (df["has_conflict"].sum() / len(df) * 100) if "has_conflict" in df.columns else 0
        pcts.append(pct)
        log.info(f"  {key}: {pct:.1f}% have conflicting sentiments across aspects")

    colors = [BAR_COLOR if "train" in k else "#88AACC" for k in keys]
    axes[0].bar(keys, pcts, color=colors, edgecolor="white")
    axes[0].set_ylabel("% of sentences")
    axes[0].set_title("Sentences with conflicting aspect sentiments", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=30)
    for i, v in enumerate(pcts):
        axes[0].text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    # Right: sentence-level sentiment conflict in MAMS — num_aspects breakdown
    df_mams = dfs.get("mams_train")
    if df_mams is not None:
        n_asp = df_mams["num_aspects"].clip(upper=6)
        counts = n_asp.value_counts().sort_index()
        labels = [str(int(i)) if i < 6 else "6+" for i in counts.index]
        axes[1].bar(labels, counts.values, color="#E88844", edgecolor="white")
        axes[1].set_title("MAMS train: aspects per sentence\n(all multi-sentiment by design)",
                          fontweight="bold")
        axes[1].set_xlabel("Number of aspects")
        axes[1].set_ylabel("Sentences")

    plt.tight_layout()
    savefig("10_multi_sentiment.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg  = load_config("config/config.yaml")
    dfs  = load_all(cfg)

    section_1_overview(dfs)
    section_2_fine_category_dist(dfs)
    section_3_coarse_dist(dfs)
    section_4_sentiment_dist(dfs)
    section_5_aspects_per_sentence(dfs)
    section_6_sentence_length(dfs)
    section_7_cooccurrence(dfs)
    section_8_implicit(dfs)
    section_9_imbalance(dfs)
    section_10_multi_sentiment(dfs)

    log.info(f"\nAll figures saved to {FIG_DIR.resolve()}")
    log.info("EDA complete.")


if __name__ == "__main__":
    main()
