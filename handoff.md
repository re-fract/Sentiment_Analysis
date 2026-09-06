# ABSA Research Pipeline — Handoff Document

> **Last updated**: 2026-08-20 (T7–T10 trained + evaluated)
> **Server**: Tesla P100-PCIE-16GB (15GB)
> **Python env**: .venv/ (activate: source .venv/bin/activate)

---

## 1. Project Goal

Fine-grained Aspect-Based Sentiment Analysis (ABSA) for restaurant reviews.
Detects which aspects are mentioned (Food / Service / Ambience / Price / General)
and classifies sentiment per aspect (positive / negative / neutral).

**Key metric**: Pair-F1 — aspect detection AND correct sentiment must both be right.

---

## 2. Current Status

### Completed
- Data parsing: SemEval 14/15/16, MAMS, ACOS → scripts/parse_all.py
- Training pipeline: trainer, datasets, evaluator → src/training/, src/models/
- **Experiments T1–T10 trained and evaluated** (BERT, RoBERTa, DistilBERT, RoBERTa-large, Joint MTL)
- Text cleaning → src/data/text_cleaning.py (applied in splits.py)
- Label smoothing 0.1 → src/training/trainer.py
- T7/T8 (roberta-large): trained 2026-08-20 02:37–11:11
- T9/T10 (joint MTL): trained 2026-08-20 14:01–15:50
- Full evaluation (T2–T10): run 2026-08-20 19:24–19:50, summary CSV saved
- Groq API integration: openai/gpt-oss-120b auto-detected from GROQ_API_KEY

### Pending (priority order)
1. LLM-as-Classifier baseline:  python scripts/run_llm_classifier.py --test mams_test --mode both
                                python scripts/run_llm_classifier.py --test semeval_test --mode both
2. Streamlit dashboard:          streamlit run app/app.py

---

## 3. Experiment Matrix

| Exp | Model                  | Data         | Status    | MAMS Pair-F1 |
|-----|------------------------|--------------|-----------|--------------|
| T1  | bert-base-uncased      | semeval_only | Done      | 0.722        |
| T2  | bert-base-uncased      | augmented†   | Done      | 0.710        |
| T3  | roberta-base           | semeval_only | Done      | **0.760**    |
| T4  | roberta-base           | augmented†   | Done      | 0.745        |
| T5  | distilbert-base-uncased| semeval_only | Done      | 0.704        |
| T6  | distilbert-base-uncased| augmented†   | Done      | 0.698        |
| T7  | roberta-large          | semeval_only | ✅ Done   | 0.758        |
| T8  | roberta-large          | augmented†   | ✅ Done   | 0.745        |
| T9  | roberta-base (joint)   | semeval_only | ✅ Done   | 0.242 ⚠️    |
| T10 | roberta-base (joint+imp)| full_train  | ✅ Done   | 0.752        |

† T2/T4/T6/T8/T10 trained on augmented split = 10,578 base + **1,098 synthetic** (synthetic_filtered.csv from run_augmentation). Augmentation pipeline is complete.

---

## 4. Full Results (T1–T10)

| Exp | Model      | SemEval Pair-F1 | MAMS Pair-F1 | ACOS Pair-F1 |
|-----|------------|-----------------|--------------|--------------|
| T1  | BERT-base  | 0.636           | 0.722        | 0.448        |
| T2  | BERT+aug†  | 0.612           | 0.710        | 0.493        |
| T3  | RoBERTa    | 0.736           | **0.760**    | 0.532        |
| T4  | RoBERTa+aug| 0.750           | 0.745        | 0.506        |
| T5  | DistilBERT | 0.634           | 0.704        | 0.438        |
| T6  | DistilBERT+| 0.640           | 0.698        | 0.461        |
| T7  | RoBERTa-L  | **0.759**       | 0.758        | 0.543        |
| T8  | RoBERTa-L+aug| 0.729         | 0.745        | **0.580**    |
| T9  | RoBERTa-joint| 0.746         | 0.242 ⚠️   | 0.383        |
| T10 | RoBERTa-joint+impl| 0.750    | 0.752        | 0.523        |

**Best overall: T7** (RoBERTa-large, semeval_only). Best ACOS: T8. T9 failed on MAMS (trained on SemEval only, minority aspect collapse).

---

## 5. Key Files

src/data/
  parsers.py           - SemEval / MAMS / ACOS parsers
  splits.py            - build_splits() — loads + cleans all data
  dataset.py           - AspectDetectionDataset, SentimentDataset
  joint_dataset.py     - ImplicitDataset (for T10 implicit head)
  text_cleaning.py     - clean_text() — HTML unescape, whitespace, control chars

src/models/
  aspect_detector.py   - AspectDetector (multi-label, per-category thresholds)
  sentiment_classifier.py - SentimentClassifier (aspect-conditioned, 3-class)
  joint_model.py       - JointABSAModel (shared encoder + 2-3 heads)

src/training/
  trainer.py           - ABSATrainer (single-task)
  joint_trainer.py     - JointTrainer (multi-task, alternating batches)
  hyperparams.py       - TrainingConfig + EXPERIMENTS dict (T1–T10)

src/evaluation/
  evaluator.py         - evaluate_experiment() → Pair-F1, Aspect macro-F1, Sent macro-F1

src/augmentation/
  llm_client.py        - CerebrasClient (works with Groq too, auto-detects from key)
  generator.py         - Synthetic review generation (Mode A)
  labeller.py          - Yelp review labelling (Mode B)

scripts/
  train_transformer.py - Train T1–T8   (--exp T3 T7, supports multiple)
  train_joint.py       - Train T9–T10  (--exp T9 T10)
  evaluate_all.py      - Evaluate any  (--exp T7 T8 or --summary-only)
  run_augmentation.py  - Full augmentation pipeline
  run_llm_classifier.py - LLM-as-Classifier baseline
  smoke_test.py        - End-to-end sanity check

outputs/
  models/              - {exp_id}_aspect_best.pt, _sentiment_best.pt, _thresholds.json
  results/             - {exp_id}_eval.json, all_results_summary.csv
  
data/
  raw/                 - Original datasets (SemEval ABSADatasets/, MAMS-for-ABSA/, ACOS/, Yelp)
  processed/           - Parsed CSVs
  augmented/           - synthetic_filtered.csv (created by augmentation pipeline)

---

## 6. Environment

### API Keys (.env)
  CEREBRAS_API_KEY=csk-...   # Cerebras — billing issue, not currently usable
  GROQ_API_KEY=gsk-...       # Groq — ACTIVE, use this

### Groq quota for openai/gpt-oss-120b
  RPM: 30  |  RPD: 1,000  |  TPM: 8,000  |  TPD: 200,000

### config.yaml Groq settings (already configured)
  cerebras:
    base_url: "https://api.groq.com/openai/v1"
    generator_model: "openai/gpt-oss-120b"
    validator_model: "openai/gpt-oss-120b"
    rate_limit_rpm: 8
    token_budget_tpm: 6000

  TIP: With 1K RPD at 5 samples/request → ~5000 samples/day.
  To finish in 1 day: set target_total: 800 in config.yaml (160 requests).

---

## 7. Server Commands (copy-paste)

# Activate env
source .venv/bin/activate

# Train T7/T8 (roberta-large, ~3h each)
python scripts/train_transformer.py --exp T7 T8

# Train T9/T10 (joint MTL, ~2.5h / 3.5h)
python scripts/train_joint.py --exp T9 T10

# Evaluate
python scripts/evaluate_all.py --exp T7 T8
python scripts/evaluate_all.py --exp T9 T10
python scripts/evaluate_all.py --summary-only   # full table

# Augmentation (separate terminal, no GPU needed)
python scripts/run_augmentation.py

# After augmentation, retrain augmented variants
python scripts/train_transformer.py --exp T2 T4 T6 T8
python scripts/train_joint.py --exp T10

# LLM baseline
python scripts/run_llm_classifier.py --test mams_test --mode both
python scripts/run_llm_classifier.py --test semeval_test --mode both

# Dashboard
streamlit run app/app.py --server.port 8501

---

## 8. Known Issues & Fixes

DistilBERT rejects token_type_ids
  Fix: Filtered in trainer.py, evaluator.py, joint_trainer.py
  Check: hasattr(model.config, "type_vocab_size")

torch.load FutureWarning
  Fix: weights_only=True added to all torch.load calls

torch.cuda.amp deprecated API
  Fix: torch.amp.GradScaler(device="cuda") + torch.amp.autocast(device_type=...)

tune_thresholds returns dict, doesn't save itself
  Fix: json.dump(tuned_thresholds, open(path, "w"), indent=2)

Joint model checkpoint key mismatch (old design)
  Fix: JointABSAModel now wraps AutoModelForSequenceClassification internally
       → keys match exactly what AspectDetector/SentimentClassifier expect

Groq model needs "openai/" prefix
  Fix: Already in config.yaml as "openai/gpt-oss-120b"

---

## 9. Dissertation Talking Points

1. RoBERTa-base (T3) outperforms BERT by ~4 Pair-F1 points on MAMS — BPE tokenizer + no NSP.
2. DistilBERT (T5) is only 5% behind BERT with 40% fewer params — good efficiency story.
3. ACOS Pair-F1 (~0.53) vs MAMS (~0.76) gap = ~24% implicit aspects + domain shift.
4. Sentiment accuracy (0.82–0.90) >> macro-F1 (0.62–0.78) reveals class imbalance (majority positive).
5. T9 (joint MTL) vs T3 (same model/data) → isolates benefit of shared encoder training.
6. T10 vs T9 → isolates benefit of implicit aspect head on ACOS.
7. LLM augmentation comparison (T2/T4/T6 vs T1/T3/T5) pending real augmented data.
