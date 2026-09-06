"""
src/training/joint_trainer.py

Multi-task trainer for the JointABSAModel (T9 / T10).

Training loop
-------------
  For each epoch:
    1. Interleave batches from aspect_loader and sentiment_loader.
    2. For T10, also interleave batches from implicit_loader (ACOS only).
    3. Accumulate gradients; step optimizer every `grad_accum_steps` steps.
    4. Evaluate combined val F1 = 0.5 × aspect_val_F1 + 0.5 × sent_val_F1.
    5. Save full model checkpoint when combined F1 improves.
    6. After training, save separate compatible checkpoints for AspectDetector
       and SentimentClassifier (same key names they expect).

Checkpoint compatibility
------------------------
  The existing AspectDetector loads:
      model.load_state_dict(torch.load(path))
  where the state dict keys are those of AutoModelForSequenceClassification
  (keys like "bert.encoder...", "classifier.weight", etc.).

  JointABSAModel uses:
      self.encoder  → keys like "encoder.bert.encoder..."
      self.aspect_head / self.sentiment_head  → "aspect_head.weight" etc.

  So we remap keys when saving:
    "encoder.*"       → "bert.*"  (or "roberta.*" depending on model)
    "aspect_head.*"   → "classifier.*"
    "sentiment_head.*"→ "classifier.*"
"""

from __future__ import annotations

import json
import sys
import time
from itertools import cycle
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

from config.taxonomy import COARSE_CATEGORIES, COARSE_TO_ID, ID_TO_COARSE, ID_TO_SENTIMENT
from src.models.joint_model import JointABSAModel
from src.training.hyperparams import TrainingConfig
from src.utils.logging import get_logger

import transformers
transformers.logging.set_verbosity_error()

log = get_logger("joint_trainer")

SENTIMENT_LABELS = ["positive", "negative", "neutral"]


# ── JointTrainer ─────────────────────────────────────────────────────────────


class JointTrainer:
    """
    Multi-task trainer for JointABSAModel.

    Parameters
    ----------
    model           : JointABSAModel (shared encoder)
    cfg             : TrainingConfig
    aspect_loader   : DataLoader for aspect detection (text only)
    sentiment_loader: DataLoader for sentiment classification (text + aspect)
    asp_val_loader  : validation DataLoader for aspect (multi-label)
    sent_val_loader : validation DataLoader for sentiment
    device          : torch.device
    implicit_loader : optional DataLoader for T10 implicit head (ACOS only)
    """

    def __init__(
        self,
        model:            JointABSAModel,
        cfg:              TrainingConfig,
        aspect_loader:    DataLoader,
        sentiment_loader: DataLoader,
        asp_val_loader:   DataLoader,
        sent_val_loader:  DataLoader,
        device:           torch.device,
        implicit_loader:  Optional[DataLoader] = None,
    ):
        self.model            = model.to(device)
        self.cfg              = cfg
        self.device           = device
        self.aspect_loader    = aspect_loader
        self.sentiment_loader = sentiment_loader
        self.asp_val_loader   = asp_val_loader
        self.sent_val_loader  = sent_val_loader
        self.implicit_loader  = implicit_loader

        # Loss functions
        smooth = getattr(cfg, "label_smoothing", 0.1)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss  = nn.CrossEntropyLoss(label_smoothing=smooth)
        self.label_smooth = smooth

        # Alpha: weight for aspect vs sentiment loss
        self.alpha = getattr(cfg, "mtl_alpha", 0.5)

        # Optimiser + scheduler
        total_steps = (
            max(len(aspect_loader), len(sentiment_loader))
            // cfg.grad_accum_steps
            * cfg.num_epochs
        )
        warmup_steps = int(total_steps * cfg.warmup_ratio)

        self.optimizer = AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps)

        self.scaler = torch.amp.GradScaler(
            device="cuda",
            enabled=(cfg.fp16 and device.type == "cuda"),
        )

        self.checkpoint_dir = Path(cfg.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.exp_id = cfg.experiment_id

        # Per-training-run bookkeeping
        self.best_combined_f1  = -1.0
        self.best_asp_f1       = 0.0
        self.best_sent_f1      = 0.0

    # ── Batch helpers ─────────────────────────────────────────────────────────

    def _to_device(self, batch: dict) -> dict:
        supports_tti = hasattr(self.model.asp_model.config, "type_vocab_size")
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
            if k != "token_type_ids" or supports_tti
        }

    def _aspect_step(self, batch: dict) -> torch.Tensor:
        labels  = batch.pop("labels").float()
        if self.label_smooth > 0:
            labels = labels * (1 - self.label_smooth) + 0.5 * self.label_smooth
        logits = self.model.forward(batch, task="aspect")
        return self.bce_loss(logits, labels)

    def _sentiment_step(self, batch: dict) -> torch.Tensor:
        labels = batch.pop("labels").long()
        logits = self.model.forward(batch, task="sentiment")
        return self.ce_loss(logits, labels)

    def _implicit_step(self, batch: dict) -> torch.Tensor:
        labels  = batch.pop("implicit_labels").float()
        batch.pop("labels", None)
        if self.label_smooth > 0:
            labels = labels * (1 - self.label_smooth) + 0.5 * self.label_smooth
        logits = self.model.forward(batch, task="implicit")
        return self.bce_loss(logits, labels)

    # ── Validation ────────────────────────────────────────────────────────────

    def _eval_aspect(self) -> float:
        """Return macro-F1 on aspect val set (with default threshold=0.5)."""
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in self.asp_val_loader:
                batch  = self._to_device(batch)
                labels = batch.pop("labels").cpu().numpy()
                batch.pop("implicit_labels", None)
                logits = self.model.forward(batch, task="aspect")
                preds = (torch.sigmoid(logits).cpu().numpy() >= 0.5).astype(int)
                all_preds.append(preds)
                all_labels.append(labels)

        preds_cat  = np.concatenate(all_preds,  axis=0)
        labels_cat = np.concatenate(all_labels, axis=0)
        f1s = []
        for i in range(labels_cat.shape[1]):
            f1s.append(f1_score(labels_cat[:, i], preds_cat[:, i], zero_division=0))
        return float(np.mean(f1s))

    def _eval_sentiment(self) -> float:
        """Return macro-F1 on sentiment val set."""
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in self.sent_val_loader:
                batch  = self._to_device(batch)
                labels = batch.pop("labels").cpu().numpy()
                logits = self.model.forward(batch, task="sentiment")
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())

        return float(f1_score(all_labels, all_preds, average="macro", zero_division=0))

    # ── Save checkpoints ──────────────────────────────────────────────────────

    def _save_checkpoints(self) -> None:
        """Save aspect and sentiment checkpoints compatible with the existing evaluator."""
        asp_path  = str(self.checkpoint_dir / f"{self.exp_id}_aspect_best.pt")
        sent_path = str(self.checkpoint_dir / f"{self.exp_id}_sentiment_best.pt")
        self.model.save_aspect_checkpoint(asp_path)
        self.model.save_sentiment_checkpoint(sent_path)

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self) -> dict:
        cfg = self.cfg
        log.info(
            f"[{self.exp_id}] Joint training start  "
            f"epochs={cfg.num_epochs}  "
            f"asp_batches={len(self.aspect_loader)}  "
            f"sent_batches={len(self.sentiment_loader)}  "
            f"alpha={self.alpha}"
        )
        if self.implicit_loader:
            log.info(f"  Implicit head enabled  imp_batches={len(self.implicit_loader)}")

        n_steps_per_epoch = max(len(self.aspect_loader), len(self.sentiment_loader))
        patience_counter  = 0
        t0 = time.time()

        for epoch in range(1, cfg.num_epochs + 1):
            self.model.train()

            asp_iter  = cycle(self.aspect_loader)
            sent_iter = cycle(self.sentiment_loader)
            imp_iter  = cycle(self.implicit_loader) if self.implicit_loader else None

            total_loss = 0.0
            self.optimizer.zero_grad()

            for step in range(n_steps_per_epoch):
                asp_batch  = self._to_device(next(asp_iter))
                sent_batch = self._to_device(next(sent_iter))

                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=(cfg.fp16 and self.device.type == "cuda"),
                ):
                    asp_loss  = self._aspect_step({**asp_batch})
                    sent_loss = self._sentiment_step({**sent_batch})

                    if imp_iter is not None:
                        imp_batch = self._to_device(next(imp_iter))
                        imp_loss  = self._implicit_step({**imp_batch})
                        # 40% aspect + 40% sentiment + 20% implicit
                        loss = 0.4 * asp_loss + 0.4 * sent_loss + 0.2 * imp_loss
                    else:
                        loss = self.alpha * asp_loss + (1 - self.alpha) * sent_loss

                    loss = loss / cfg.grad_accum_steps

                self.scaler.scale(loss).backward()
                total_loss += loss.item() * cfg.grad_accum_steps

                if (step + 1) % cfg.grad_accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

            # ── Validation ───────────────────────────────────────────────────
            asp_f1  = self._eval_aspect()
            sent_f1 = self._eval_sentiment()
            combined_f1 = 0.5 * asp_f1 + 0.5 * sent_f1
            avg_loss    = total_loss / n_steps_per_epoch
            elapsed     = (time.time() - t0) / 60

            log.info(
                f"  Epoch {epoch}/{cfg.num_epochs}  "
                f"loss={avg_loss:.4f}  "
                f"asp_f1={asp_f1:.4f}  sent_f1={sent_f1:.4f}  "
                f"combined={combined_f1:.4f}  ({elapsed:.1f}min)"
            )

            if combined_f1 > self.best_combined_f1:
                self.best_combined_f1 = combined_f1
                self.best_asp_f1      = asp_f1
                self.best_sent_f1     = sent_f1
                patience_counter = 0
                self._save_checkpoints()
                log.info(f"    ** New best combined: {combined_f1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    log.info(f"  Early stopping at epoch {epoch} "
                             f"(no improvement for {cfg.patience} epochs)")
                    break

        log.info(
            f"[{self.exp_id}] Training done  "
            f"best_asp_f1={self.best_asp_f1:.4f}  "
            f"best_sent_f1={self.best_sent_f1:.4f}  "
            f"best_combined={self.best_combined_f1:.4f}"
        )
        return {
            "best_asp_f1":      self.best_asp_f1,
            "best_sent_f1":     self.best_sent_f1,
            "best_combined_f1": self.best_combined_f1,
        }
