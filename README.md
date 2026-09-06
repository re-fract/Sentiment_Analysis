# ABSA Research Pipeline

**Aspect-Based Sentiment Analysis** for restaurant reviews — BTP (Bachelor's Thesis Project) 2025–26.

## Project Overview

This project builds a two-stage ABSA pipeline:
1. **Aspect Detection** — multi-label classification of which of 5 coarse categories are discussed (Food, Service, Ambience, Price, General)
2. **Sentiment Classification** — aspect-conditioned prediction of sentiment (positive / negative / neutral) for each detected category

### Datasets
| Dataset | Split | Samples | Notes |
|---|---|---|---|
| SemEval 14/15/16 | train+test | 8,830 | Fine-grained, single-aspect per row |
| MAMS | train/val/test | 3,949 | 100% multi-sentiment sentences |
| ACOS | train/dev/test | 2,284 | Includes implicit aspects |
| Yelp (silver) | augmentation | ~2,000 | LLM-labeled real reviews |
| Synthetic | augmentation | ~742 | LLM-generated for imbalanced buckets |

## Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set API key (for LLM augmentation)
# Add to .env: CEREBRAS_API_KEY=your_key_here
```

## Pipeline

### Step 1 — Parse raw data
```bash
python scripts/parse_all.py
```

### Step 2 — EDA
```bash
python notebooks/eda.py
# Figures saved to outputs/figures/
```

### Step 3 — LLM Augmentation (requires Cerebras API key)
```bash
# Filter Yelp to restaurant reviews (one-time, ~5 min)
python scripts/run_augmentation.py --step filter-yelp

# Generate synthetic samples for imbalanced buckets
python scripts/run_augmentation.py --step generate

# Label real Yelp reviews
python scripts/run_augmentation.py --step label

# Quality filter + cross-model consistency
python scripts/run_augmentation.py --step filter

# Manual verification (100 samples, interactive)
python scripts/run_augmentation.py --step verify
```

### Step 4 — Baseline models
```bash
python scripts/train_baselines.py
# Results: outputs/results/baseline_*.json
```

### Step 5 — Transformer training (run on server with P100 GPU)
```bash
# Single experiment:
python scripts/train_transformer.py --exp T1

# All 6 experiments:
python scripts/train_transformer.py --exp all

# Experiments:
# T1/T2: BERT-base       (without/with augmentation)
# T3/T4: RoBERTa-base    (without/with augmentation)
# T5/T6: DistilBERT-base (without/with augmentation)
```

### Step 6 — Evaluate
```bash
python scripts/evaluate_all.py --exp all
# Summary: outputs/results/all_results_summary.csv
```

### Step 7 — LLM-as-Classifier baseline (requires API)
```bash
python scripts/run_llm_classifier.py --test mams_test --mode both
```

### Step 8 — Streamlit dashboard
```bash
streamlit run app/app.py
```

## Project Structure

```
btp/
├── config/
│   ├── config.yaml          # All paths and hyperparameters
│   └── taxonomy.py          # Category mappings and utilities
├── data/
│   ├── raw/                 # Raw dataset files
│   ├── processed/           # Parsed CSVs (unified schema)
│   └── augmented/           # Synthetic/silver-labeled data
├── notebooks/
│   └── eda.py               # EDA script (saves figures)
├── scripts/                 # All runnable scripts
├── src/
│   ├── augmentation/        # LLM augmentation pipeline
│   ├── data/                # Parsers, datasets, splits
│   ├── evaluation/          # Metrics, evaluator, error analysis
│   ├── models/              # Baseline, aspect detector, sentiment classifier
│   ├── training/            # Trainer, hyperparameters
│   └── utils/               # IO, logging
├── app/
│   └── app.py               # Streamlit dashboard
└── outputs/
    ├── figures/             # EDA plots
    ├── models/              # Saved checkpoints
    └── results/             # Evaluation JSONs and CSVs
```

## Key Design Decisions

- **Fine-grained training → coarse inference**: Models are trained with detailed categories but predictions are mapped to 5 coarse buckets at inference time for better cross-dataset generalisation.
- **Aspect-conditioned sentiment input**: `[CLS] {text} [SEP] {category} [SEP]` allows the transformer to focus on the relevant aspect, critical for multi-aspect sentences.
- **Cross-model consistency**: Generated/labeled samples are validated by a second LLM (Gemma-4-31B) before being added to training data.
- **Implicit aspects**: ACOS data (~24% implicit aspects) teaches the model to detect opinion targets not explicitly named in text.
