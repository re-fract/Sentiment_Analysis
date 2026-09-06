"""
src/models/aspect_detector.py

Transformer-based multi-label aspect category detector.

Input:  review text
Output: probability for each of the 5 coarse categories
        (Food, Service, Ambience, Price, General)

At inference time, per-aspect thresholds (tuned on val set) are applied
instead of a fixed 0.5 cut-off to handle class imbalance.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config.taxonomy import COARSE_CATEGORIES, COARSE_TO_ID, ID_TO_COARSE
from src.data.dataset import AspectDetectionDataset
from src.training.trainer import ABSATrainer, build_aspect_model
from src.training.hyperparams import TrainingConfig
from src.utils.logging import get_logger

log = get_logger("aspect_detector")

NUM_COARSE = len(COARSE_CATEGORIES)


class AspectDetector:
    """
    Wraps a fine-tuned multi-label classifier for aspect detection.

    Usage:
        detector = AspectDetector.from_checkpoint("outputs/models/T1_aspect_best.pt",
                                                   model_name="bert-base-uncased")
        aspects = detector.predict("The food was great but service was slow.")
        # -> ["Food", "Service"]
    """

    def __init__(
        self,
        model_name: str,
        device:     Optional[torch.device] = None,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        self.device    = device or (torch.device("cuda") if torch.cuda.is_available()
                                    else torch.device("cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = build_aspect_model(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Per-aspect thresholds (can be tuned post-training)
        self.thresholds = thresholds or {cat: 0.5 for cat in COARSE_CATEGORIES}

    def load_checkpoint(self, path: str) -> "AspectDetector":
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        log.info(f"Loaded checkpoint: {path}")
        return self

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_name:      str,
        thresholds:      Optional[Dict[str, float]] = None,
        device:          Optional[torch.device]     = None,
    ) -> "AspectDetector":
        detector = cls(model_name, device=device, thresholds=thresholds)
        detector.load_checkpoint(checkpoint_path)
        return detector

    def predict(self, text: str) -> List[str]:
        """Return list of detected coarse categories for a single review."""
        enc = self.tokenizer(
            text,
            max_length=128, truncation=True, padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            logits = self.model(**enc).logits
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        detected = []
        for i, cat in enumerate(COARSE_CATEGORIES):
            if probs[i] >= self.thresholds.get(cat, 0.5):
                detected.append(cat)
        return detected

    def predict_proba(self, text: str) -> Dict[str, float]:
        """Return raw probabilities for all categories."""
        enc = self.tokenizer(
            text,
            max_length=128, truncation=True, padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        return {cat: float(probs[i]) for i, cat in enumerate(COARSE_CATEGORIES)}

    def tune_thresholds(
        self,
        val_df:      "pd.DataFrame",    # type: ignore
        max_length:  int = 128,
    ) -> Dict[str, float]:
        """
        Find per-category F1-optimal threshold on a validation set.
        Sweeps 0.1–0.9 in steps of 0.05 per category.
        Returns the best thresholds dict and saves it alongside the model.
        """
        import pandas as pd
        from sklearn.metrics import f1_score
        import ast

        dataset = AspectDetectionDataset(val_df, self.tokenizer, max_length)
        loader  = DataLoader(dataset, batch_size=32, shuffle=False)

        all_probs  = []
        all_labels = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                labels = batch.pop("labels")
                batch  = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.model(**batch).logits
                probs  = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(labels.numpy())

        all_probs  = np.concatenate(all_probs,  axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        best_thresholds = {}
        for i, cat in enumerate(COARSE_CATEGORIES):
            best_f1, best_thr = 0.0, 0.5
            for thr in np.arange(0.1, 0.95, 0.05):
                preds = (all_probs[:, i] >= thr).astype(int)
                f1    = f1_score(all_labels[:, i], preds, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_thr = f1, float(thr)
            best_thresholds[cat] = round(best_thr, 2)
            log.info(f"  {cat}: best_thr={best_thr:.2f}  best_f1={best_f1:.4f}")

        self.thresholds = best_thresholds
        log.info(f"Tuned thresholds: {best_thresholds}")
        return best_thresholds


# ── Training entry point ──────────────────────────────────────────────────────

def train_aspect_detector(
    cfg:        TrainingConfig,
    train_df:   "pd.DataFrame",    # type: ignore
    val_df:     "pd.DataFrame",    # type: ignore
    device:     Optional[torch.device] = None,
) -> dict:
    """Train an aspect detector and return best val metrics + checkpoint path."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train_ds  = AspectDetectionDataset(train_df, tokenizer, cfg.max_seq_length)
    val_ds    = AspectDetectionDataset(val_df,   tokenizer, cfg.max_seq_length)

    log.info(f"[{cfg.experiment_id}] Aspect detector  "
             f"train={len(train_ds)}  val={len(val_ds)}")

    model   = build_aspect_model(cfg.model_name)
    trainer = ABSATrainer(model, "aspect", cfg, train_ds, val_ds, device=device)
    results = trainer.train()
    return results
