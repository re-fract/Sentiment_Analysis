"""
app/app.py — Streamlit ABSA Dashboard

A research prototype that demonstrates the full pipeline on restaurant reviews.

Features:
  1. Single review analysis (type or paste any review)
  2. Batch analysis (upload CSV of reviews)
  3. Model comparison (run any trained experiment side-by-side)
  4. Dataset explorer (browse EDA figures and statistics)

Run with:
    .venv/Scripts/streamlit run app/app.py
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config.taxonomy import COARSE_CATEGORIES, SENTIMENT_LABELS
from src.utils.io import load_config, load_json
from src.utils.logging import get_logger

log = get_logger("app")

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title    = "ABSA Research Dashboard",
    page_icon     = "🍽️",
    layout        = "wide",
    initial_sidebar_state = "expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-title {
        font-size: 2.4rem; font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle { color: #6b7280; font-size: 1rem; margin-bottom: 2rem; }

    .aspect-card {
        background: #1e1e2e; border-radius: 12px;
        padding: 1rem 1.2rem; margin: 0.4rem 0;
        border-left: 4px solid; box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .aspect-cat  { font-weight: 600; font-size: 0.95rem; color: #e2e8f0; }
    .aspect-sent { font-size: 0.85rem; margin-top: 0.15rem; }

    .sent-positive { color: #4ade80; border-color: #4ade80; }
    .sent-negative { color: #f87171; border-color: #f87171; }
    .sent-neutral  { color: #facc15; border-color: #facc15; }

    .metric-box {
        background: #1e1e2e; border-radius: 10px;
        padding: 1rem; text-align: center;
    }
    .metric-val  { font-size: 2rem; font-weight: 700; color: #a78bfa; }
    .metric-label{ font-size: 0.8rem; color: #9ca3af; margin-top: 0.2rem; }

    .stTextArea textarea { background: #1e1e2e !important; color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sentiment colours / helpers
# ─────────────────────────────────────────────────────────────────────────────
SENT_COLORS = {
    "positive": "#4ade80",
    "negative": "#f87171",
    "neutral":  "#facc15",
}
SENT_EMOJI = {"positive": "😊", "negative": "😞", "neutral": "😐"}

CAT_ICONS = {
    "Food":     "🍔",
    "Service":  "🧑‍🍳",
    "Ambience": "✨",
    "Price":    "💰",
    "General":  "🏠",
}


def sentiment_badge(sent: str) -> str:
    color = SENT_COLORS.get(sent, "#9ca3af")
    emoji = SENT_EMOJI.get(sent, "")
    return (f'<span style="background:{color}22;color:{color};'
            f'border:1px solid {color};border-radius:6px;'
            f'padding:2px 8px;font-size:0.8rem;font-weight:600;">'
            f'{emoji} {sent.capitalize()}</span>')


def render_aspect_card(cat: str, sent: str, prob: float | None = None) -> None:
    color    = SENT_COLORS.get(sent, "#9ca3af")
    icon     = CAT_ICONS.get(cat, "📌")
    prob_str = f" &nbsp;·&nbsp; <span style='color:#9ca3af;font-size:0.78rem'>{prob:.0%} confidence</span>" if prob is not None else ""
    st.markdown(
        f'<div class="aspect-card" style="border-color:{color}">'
        f'<div class="aspect-cat">{icon} {cat}{prob_str}</div>'
        f'<div class="aspect-sent">{sentiment_badge(sent)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model...")
def load_pipeline(exp_id: str, model_name: str):
    """Load AspectDetector + SentimentClassifier and return ABSAPipeline."""
    import torch
    from src.models.aspect_detector import AspectDetector
    from src.models.sentiment_classifier import SentimentClassifier, ABSAPipeline

    cfg      = load_config()
    model_dir= Path(cfg["paths"]["outputs"]["models"])
    asp_ckpt = model_dir / f"{exp_id}_aspect_best.pt"
    sent_ckpt= model_dir / f"{exp_id}_sentiment_best.pt"
    thr_path = model_dir / f"{exp_id}_thresholds.json"

    if not asp_ckpt.exists() or not sent_ckpt.exists():
        return None, f"Checkpoints not found for {exp_id}. Train the model first."

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thresholds = load_json(thr_path) if thr_path.exists() else None

    detector   = AspectDetector.from_checkpoint(str(asp_ckpt), model_name,
                                                thresholds=thresholds, device=device)
    classifier = SentimentClassifier.from_checkpoint(str(sent_ckpt), model_name, device=device)
    pipeline   = ABSAPipeline(detector, classifier)
    return pipeline, None


def list_available_experiments() -> list[str]:
    """Return experiment IDs for which both checkpoints exist."""
    try:
        cfg = load_config()
        model_dir = Path(cfg["paths"]["outputs"]["models"])
        available = []
        from src.training.hyperparams import EXPERIMENTS
        for exp_id in EXPERIMENTS:
            if ((model_dir / f"{exp_id}_aspect_best.pt").exists() and
                    (model_dir / f"{exp_id}_sentiment_best.pt").exists()):
                available.append(exp_id)
        return available
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🍽️ ABSA Dashboard")
    st.markdown("*Aspect-Based Sentiment Analysis*")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🔍 Analyze Review", "📊 Model Comparison", "📈 EDA Explorer",
         "📋 Results Table"],
        label_visibility="collapsed",
    )
    st.divider()

    # Experiment selector
    available = list_available_experiments()
    if available:
        selected_exp = st.selectbox("Model (experiment)", available)
        from src.training.hyperparams import EXPERIMENTS
        model_name = EXPERIMENTS.get(selected_exp, {}).get("model_name", "")
        st.caption(f"Model: `{model_name}`")
    else:
        selected_exp = None
        st.warning("No trained models found.\nTrain at least one experiment first.")

    st.divider()
    st.caption("BTP Research Project · 2025–26")


# ─────────────────────────────────────────────────────────────────────────────
# Page 1: Analyze Review
# ─────────────────────────────────────────────────────────────────────────────

if page == "🔍 Analyze Review":
    st.markdown('<div class="main-title">Aspect-Based Sentiment Analysis</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Enter a restaurant review to extract aspect-sentiment pairs</div>',
                unsafe_allow_html=True)

    example_reviews = [
        "The pasta was incredible — perfectly al dente, rich sauce. Service was a bit slow but staff were friendly. Prices are reasonable for the area.",
        "Absolutely terrible experience. Waited 40 minutes, food arrived cold, and the waiter was rude. Never coming back.",
        "Nice cozy atmosphere. The cocktail menu is creative. Food was okay but nothing special.",
        "Best sushi I've had outside Japan. The omakase menu is worth every penny.",
    ]

    col1, col2 = st.columns([3, 1])
    with col2:
        example_pick = st.selectbox("Load example", ["(type your own)"] + [f"Example {i+1}" for i in range(len(example_reviews))])

    default_text = ""
    if example_pick != "(type your own)":
        idx = int(example_pick.split()[-1]) - 1
        default_text = example_reviews[idx]

    review_text = st.text_area(
        "Review text",
        value=default_text,
        height=120,
        placeholder="Paste or type a restaurant review...",
    )

    analyze_btn = st.button("✨ Analyze", type="primary", use_container_width=False)

    if analyze_btn and review_text.strip():
        if selected_exp is None:
            st.error("No trained model available. Train a model first.")
        else:
            pipeline, err = load_pipeline(selected_exp, model_name)
            if err:
                st.error(err)
            else:
                with st.spinner("Analyzing..."):
                    results = pipeline.analyze_with_proba(review_text.strip())

                if not results:
                    st.info("No aspect categories detected in this review.")
                else:
                    st.markdown(f"**{len(results)} aspect(s) detected** &nbsp;·&nbsp; Model: `{selected_exp}`",
                                unsafe_allow_html=True)
                    st.divider()

                    cols = st.columns(min(len(results), 3))
                    for i, r in enumerate(results):
                        with cols[i % len(cols)]:
                            render_aspect_card(r["category"], r["sentiment"],
                                               r.get("aspect_probability"))

                    # Sentiment pie chart
                    st.divider()
                    st.markdown("#### Sentiment distribution")
                    from collections import Counter
                    sent_counts = Counter(r["sentiment"] for r in results)
                    fig, ax = plt.subplots(figsize=(4, 3), facecolor="none")
                    ax.pie(
                        sent_counts.values(),
                        labels=sent_counts.keys(),
                        colors=[SENT_COLORS.get(s, "#888") for s in sent_counts.keys()],
                        autopct="%1.0f%%", startangle=90,
                        wedgeprops={"edgecolor": "#0e1117", "linewidth": 2},
                        textprops={"color": "#e2e8f0"},
                    )
                    ax.set_facecolor("none")
                    st.pyplot(fig, transparent=True)
                    plt.close()

    elif analyze_btn:
        st.warning("Please enter a review first.")


# ─────────────────────────────────────────────────────────────────────────────
# Page 2: Model Comparison
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📊 Model Comparison":
    st.markdown('<div class="main-title">Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Compare predictions across experiments</div>',
                unsafe_allow_html=True)

    if not available:
        st.warning("No trained models found. Train at least one experiment first.")
    else:
        compare_exps = st.multiselect("Select experiments to compare", available,
                                      default=available[:min(2, len(available))])
        review_text = st.text_area("Review text for comparison", height=100,
                                   placeholder="Enter a review to compare models...")
        compare_btn = st.button("Compare Models", type="primary")

        if compare_btn and review_text.strip() and compare_exps:
            cols = st.columns(len(compare_exps))
            for i, exp_id in enumerate(compare_exps):
                from src.training.hyperparams import EXPERIMENTS
                mn = EXPERIMENTS.get(exp_id, {}).get("model_name", "")
                pipe, err = load_pipeline(exp_id, mn)

                with cols[i]:
                    st.markdown(f"**{exp_id}** — `{mn.split('-')[0]}`")
                    if err:
                        st.error(err)
                    else:
                        results = pipe.analyze_with_proba(review_text.strip())
                        if not results:
                            st.info("No aspects detected.")
                        for r in results:
                            render_aspect_card(r["category"], r["sentiment"],
                                               r.get("aspect_probability"))


# ─────────────────────────────────────────────────────────────────────────────
# Page 3: EDA Explorer
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📈 EDA Explorer":
    st.markdown('<div class="main-title">EDA Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Explore dataset statistics and distributions</div>',
                unsafe_allow_html=True)

    fig_dir = Path("outputs/figures")
    if not fig_dir.exists():
        st.warning("No EDA figures found. Run `notebooks/eda.py` first.")
    else:
        figs = sorted(fig_dir.glob("*.png"))
        fig_labels = {f.name: f.name.replace(".png", "").replace("_", " ").title()
                      for f in figs}

        selected_figs = st.multiselect(
            "Select figures to display",
            options=[f.name for f in figs],
            default=[f.name for f in figs[:4]],
            format_func=lambda x: fig_labels.get(x, x),
        )

        if selected_figs:
            n_cols = 2
            cols = st.columns(n_cols)
            for i, fig_name in enumerate(selected_figs):
                with cols[i % n_cols]:
                    st.markdown(f"**{fig_labels[fig_name]}**")
                    st.image(str(fig_dir / fig_name), use_column_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page 4: Results Table
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📋 Results Table":
    st.markdown('<div class="main-title">Results Table</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Full metrics across experiments and test sets</div>',
                unsafe_allow_html=True)

    results_dir = Path("outputs/results")
    summary_csv = results_dir / "all_results_summary.csv"

    if summary_csv.exists():
        df = pd.read_csv(summary_csv)
        st.dataframe(df.style.highlight_max(
            subset=["Aspect macro-F1", "Sent macro-F1", "Pair-F1"],
            color="#2d4a3e",
        ), use_container_width=True, height=400)

        st.download_button(
            "⬇️ Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="absa_results.csv",
            mime="text/csv",
        )

        # Highlight best per metric
        st.divider()
        st.markdown("#### Best models per metric")
        for col in ["Aspect macro-F1", "Sent macro-F1", "Pair-F1"]:
            if col in df.columns:
                idx = pd.to_numeric(df[col], errors="coerce").idxmax()
                if pd.notna(idx):
                    row = df.loc[idx]
                    st.markdown(f"**{col}**: `{row['Experiment']}` on `{row['Test set']}` — **{row[col]}**")

        # Baseline comparison
        baseline_asp  = results_dir / "baseline_aspect_results.json"
        baseline_sent = results_dir / "baseline_sentiment_results.json"
        if baseline_asp.exists():
            with st.expander("📉 TF-IDF Baseline Results"):
                asp_data  = load_json(baseline_asp)
                sent_data = load_json(baseline_sent) if baseline_sent.exists() else {}
                rows = []
                for exp, models in asp_data.items():
                    for model, m in models.items():
                        rows.append({"Experiment": f"Baseline-{exp}", "Classifier": model,
                                     "Aspect macro-F1": m["macro_f1"]})
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    else:
        st.info("No results summary found. Run `scripts/evaluate_all.py` after training.")

        # Show raw eval JSONs if they exist
        eval_jsons = sorted(results_dir.glob("T*_eval.json")) if results_dir.exists() else []
        if eval_jsons:
            st.markdown("#### Available evaluation files:")
            for p in eval_jsons:
                with st.expander(p.name):
                    data = load_json(p)
                    st.json(data)
