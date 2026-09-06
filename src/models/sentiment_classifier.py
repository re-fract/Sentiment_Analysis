"""
src/models/sentiment_classifier.py

Transformer-based aspect-conditioned sentiment classifier.

Input:  review text + coarse category
        (tokenized as: [CLS] text [SEP] category [SEP])
Output: softmax over {positive, negative, neutral}

At inference time this is chained after the aspect detector:
  text -> aspect_detector -> [category_1, category_2, ...]
       -> sentiment_classifier(text, cat_i) -> sentiment_i
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from config.taxonomy import SENTIMENT_LABELS, SENTIMENT_TO_ID, ID_TO_SENTIMENT
from src.data.dataset import SentimentDataset
from src.training.trainer import ABSATrainer, build_sentiment_model
from src.training.hyperparams import TrainingConfig
from src.utils.logging import get_logger

log = get_logger("sentiment_classifier")

NUM_SENTIMENT = len(SENTIMENT_LABELS)


class SentimentClassifier:
    """
    Wraps a fine-tuned aspect-conditioned sentiment classifier.

    Usage:
        clf = SentimentClassifier.from_checkpoint(
            "outputs/models/T1_sentiment_best.pt",
            model_name="bert-base-uncased"
        )
        sentiment = clf.predict("The pasta was cold.", "Food")
        # -> "negative"
    """

    def __init__(
        self,
        model_name: str,
        device:     Optional[torch.device] = None,
    ):
        self.device    = device or (torch.device("cuda") if torch.cuda.is_available()
                                    else torch.device("cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = build_sentiment_model(model_name)
        self.model.to(self.device)
        self.model.eval()

    def load_checkpoint(self, path: str) -> "SentimentClassifier":
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        log.info(f"Loaded checkpoint: {path}")
        return self

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_name:      str,
        device:          Optional[torch.device] = None,
    ) -> "SentimentClassifier":
        clf = cls(model_name, device=device)
        clf.load_checkpoint(checkpoint_path)
        return clf

    def _encode(self, text: str, category: str) -> dict:
        enc = self.tokenizer(
            text,
            category,
            max_length=128, truncation=True, padding="max_length",
            return_tensors="pt",
        )
        # DistilBERT has no token_type_ids
        supports_tti = hasattr(self.model.config, "type_vocab_size")
        return {k: v.to(self.device) for k, v in enc.items()
                if k != "token_type_ids" or supports_tti}

    def predict(self, text: str, category: str) -> str:
        """Return predicted sentiment label for (text, category)."""
        enc = self._encode(text, category)
        with torch.no_grad():
            logits = self.model(**enc).logits
        idx = torch.argmax(logits, dim=-1).item()
        return ID_TO_SENTIMENT[idx]

    def predict_proba(self, text: str, category: str) -> Dict[str, float]:
        """Return sentiment probability distribution for (text, category)."""
        enc = self._encode(text, category)
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return {label: float(probs[i]) for i, label in enumerate(SENTIMENT_LABELS)}

    def predict_batch(
        self,
        text:       str,
        categories: List[str],
    ) -> List[str]:
        """
        Predict sentiments for multiple aspect categories in a single review.
        More efficient than calling predict() N times as it batches the encoding.
        """
        if not categories:
            return []

        inputs = self.tokenizer(
            [text] * len(categories),
            categories,
            max_length=128, truncation=True, padding="max_length",
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
        idxs = torch.argmax(logits, dim=-1).cpu().numpy()
        return [ID_TO_SENTIMENT[int(i)] for i in idxs]


# ── Full inference pipeline ───────────────────────────────────────────────────

class ABSAPipeline:
    """
    Chains aspect detection + sentiment classification into a single pipeline.

    Usage:
        pipeline = ABSAPipeline(detector, classifier)
        result = pipeline.analyze("The pasta was cold but the ambience was lovely.")
        # -> [{"category": "Food",    "sentiment": "negative", "probability": 0.91},
        #     {"category": "Ambience","sentiment": "positive", "probability": 0.87}]
    """

    def __init__(
        self,
        aspect_detector:       "AspectDetector",    # type: ignore
        sentiment_classifier:  "SentimentClassifier",
    ):
        self.detector   = aspect_detector
        self.classifier = sentiment_classifier

    def analyze(self, text: str) -> List[Dict]:
        categories = self.detector.predict(text)
        if not categories:
            return []

        sentiments = self.classifier.predict_batch(text, categories)
        probs_det  = self.detector.predict_proba(text)

        return [
            {
                "category":          cat,
                "sentiment":         sent,
                "aspect_probability":round(probs_det.get(cat, 0.0), 4),
            }
            for cat, sent in zip(categories, sentiments)
        ]

    def analyze_with_proba(self, text: str) -> List[Dict]:
        """Full analysis including sentiment probability distributions."""
        categories = self.detector.predict(text)
        if not categories:
            return []

        results = []
        probs_det = self.detector.predict_proba(text)
        for cat in categories:
            sent_probs = self.classifier.predict_proba(text, cat)
            best_sent  = max(sent_probs, key=sent_probs.get)
            results.append({
                "category":              cat,
                "sentiment":             best_sent,
                "sentiment_proba":       sent_probs,
                "aspect_probability":    round(probs_det.get(cat, 0.0), 4),
            })
        return results


# ── Training entry point ──────────────────────────────────────────────────────

def train_sentiment_classifier(
    cfg:      TrainingConfig,
    train_df: "pd.DataFrame",    # type: ignore
    val_df:   "pd.DataFrame",    # type: ignore
    device:   Optional[torch.device] = None,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train_ds  = SentimentDataset(train_df, tokenizer, cfg.max_seq_length)
    val_ds    = SentimentDataset(val_df,   tokenizer, cfg.max_seq_length)

    log.info(f"[{cfg.experiment_id}] Sentiment classifier  "
             f"train={len(train_ds)}  val={len(val_ds)}")

    model   = build_sentiment_model(cfg.model_name)
    trainer = ABSATrainer(model, "sentiment", cfg, train_ds, val_ds, device=device)
    results = trainer.train()
    return results
