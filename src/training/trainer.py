"""
src/training/trainer.py

Training loop for transformer-based ABSA models.

Handles:
  - Aspect detection   (multi-label BCE loss)
  - Sentiment classification (cross-entropy loss)
  - FP16 (for P100 GPU)
  - Gradient accumulation (effective batch size = 32)
  - Early stopping + best-checkpoint saving
  - Per-step logging to outputs/results/{experiment_id}_{task}_log.jsonl
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import transformers
transformers.logging.set_verbosity_error()   # suppress pretrained weight load warnings

from config.taxonomy import COARSE_CATEGORIES, SENTIMENT_LABELS
from src.training.hyperparams import TrainingConfig
from src.utils.logging import get_logger

log = get_logger("trainer")

NUM_COARSE    = len(COARSE_CATEGORIES)
NUM_SENTIMENT = len(SENTIMENT_LABELS)


# ── Metric helpers (avoid sklearn dependency during training) ─────────────────

def _binary_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Macro-averaged F1 for multi-label binary predictions."""
    eps  = 1e-8
    f1s  = []
    for i in range(preds.shape[1]):
        tp = ((preds[:, i] == 1) & (targets[:, i] == 1)).float().sum()
        fp = ((preds[:, i] == 1) & (targets[:, i] == 0)).float().sum()
        fn = ((preds[:, i] == 0) & (targets[:, i] == 1)).float().sum()
        prec  = tp / (tp + fp + eps)
        rec   = tp / (tp + fn + eps)
        f1    = 2 * prec * rec / (prec + rec + eps)
        f1s.append(f1.item())
    return sum(f1s) / len(f1s)


def _macro_f1_multiclass(preds: torch.Tensor, targets: torch.Tensor,
                          num_classes: int) -> float:
    """Macro-averaged F1 for multiclass predictions."""
    eps = 1e-8
    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).float().sum()
        fp = ((preds == c) & (targets != c)).float().sum()
        fn = ((preds != c) & (targets == c)).float().sum()
        prec = tp / (tp + fp + eps)
        rec  = tp / (tp + fn + eps)
        f1   = 2 * prec * rec / (prec + rec + eps)
        f1s.append(f1.item())
    return sum(f1s) / len(f1s)


# ── Model builders ────────────────────────────────────────────────────────────

def build_aspect_model(model_name: str) -> nn.Module:
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels     = NUM_COARSE,
        problem_type   = "multi_label_classification",
        ignore_mismatched_sizes=True,
    )


def build_sentiment_model(model_name: str) -> nn.Module:
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels   = NUM_SENTIMENT,
        problem_type = "single_label_classification",
        ignore_mismatched_sizes=True,
    )


# ── Training loop ─────────────────────────────────────────────────────────────

class ABSATrainer:
    """
    General-purpose trainer for both aspect detection and sentiment classification.
    """

    def __init__(
        self,
        model:      nn.Module,
        task:       str,           # "aspect" | "sentiment"
        cfg:        TrainingConfig,
        train_ds:   Dataset,
        val_ds:     Dataset,
        device:     Optional[torch.device] = None,
    ):
        self.model  = model
        self.task   = task
        self.cfg    = cfg
        self.device = device or (torch.device("cuda") if torch.cuda.is_available()
                                 else torch.device("cpu"))
        log.info(f"Using device: {self.device}")
        self.model.to(self.device)

        # Data loaders
        self.train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
        self.val_loader   = DataLoader(
            val_ds,   batch_size=cfg.batch_size * 2, shuffle=False, num_workers=0)

        # Optimiser
        no_decay  = ["bias", "LayerNorm.weight"]
        params    = [
            {"params": [p for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             "weight_decay": cfg.weight_decay},
            {"params": [p for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        self.optimizer = AdamW(params, lr=cfg.learning_rate)

        total_steps = (len(self.train_loader) // cfg.grad_accum_steps) * cfg.num_epochs
        warmup_steps= int(total_steps * cfg.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps)

        # AMP scaler — use new API (torch 2.x); falls back gracefully on CPU
        self.scaler = torch.amp.GradScaler(
            device="cuda",
            enabled=(cfg.fp16 and self.device.type == "cuda"),
        )

        # Loss functions
        if task == "aspect":
            # BCEWithLogitsLoss doesn't have label_smoothing, so we apply
            # soft targets manually: pos_target = 1 - smooth, neg_target = smooth
            smooth = getattr(cfg, "label_smoothing", 0.1)
            self.label_smooth = smooth
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            smooth = getattr(cfg, "label_smoothing", 0.1)
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=smooth)

        # Logging
        self.log_path = (Path(cfg.checkpoint_dir) /
                         f"{cfg.experiment_id}_{task}_log.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _to_device(self, batch: dict) -> dict:
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
                if k != "token_type_ids" or hasattr(self.model.config, "type_vocab_size")}

    def _forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        labels  = batch.pop("labels")
        # DistilBERT has no token_type_ids — drop if model doesn't support it
        if "token_type_ids" in batch and not hasattr(self.model.config, "type_vocab_size"):
            batch.pop("token_type_ids")
        outputs = self.model(**batch)
        logits  = outputs.logits

        if self.task == "aspect":
            # Apply manual label smoothing for BCEWithLogitsLoss
            smooth  = getattr(self, "label_smooth", 0.0)
            targets = labels.float()
            if smooth > 0:
                targets = targets * (1 - smooth) + 0.5 * smooth
            loss = self.loss_fn(logits, targets)
        else:
            loss = self.loss_fn(logits, labels.long())

        return loss, logits

    def _eval(self) -> dict:
        self.model.eval()
        all_logits  = []
        all_labels  = []
        total_loss  = 0.0
        n_batches   = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch  = self._to_device(batch)
                labels = batch["labels"].clone()
                loss, logits = self._forward(batch)
                total_loss  += loss.item()
                n_batches   += 1
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        logits_cat = torch.cat(all_logits, dim=0)
        labels_cat = torch.cat(all_labels, dim=0)

        if self.task == "aspect":
            preds  = (torch.sigmoid(logits_cat) > self.cfg.default_threshold).long()
            f1     = _binary_f1(preds, labels_cat.long())
        else:
            preds  = torch.argmax(logits_cat, dim=-1)
            f1     = _macro_f1_multiclass(preds, labels_cat.long(), NUM_SENTIMENT)

        self.model.train()
        return {"val_loss": round(total_loss / max(n_batches, 1), 4), "val_f1": round(f1, 4)}

    def train(self) -> dict:
        best_val_f1  = 0.0
        patience_cnt = 0
        ckpt_path    = (Path(self.cfg.checkpoint_dir) /
                        f"{self.cfg.experiment_id}_{self.task}_best.pt")

        log.info(f"[{self.cfg.experiment_id}|{self.task}] Training start  "
                 f"epochs={self.cfg.num_epochs}  "
                 f"train_batches={len(self.train_loader)}")

        global_step = 0
        t0 = time.time()

        for epoch in range(1, self.cfg.num_epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            self.optimizer.zero_grad()

            for step, batch in enumerate(self.train_loader):
                batch = self._to_device(batch)

                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=(self.cfg.fp16 and self.device.type == "cuda"),
                ):
                    loss, _ = self._forward(batch)
                    loss    = loss / self.cfg.grad_accum_steps

                self.scaler.scale(loss).backward()
                epoch_loss += loss.item() * self.cfg.grad_accum_steps

                if (step + 1) % self.cfg.grad_accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

            # Epoch-level eval
            metrics = self._eval()
            avg_loss = epoch_loss / len(self.train_loader)
            elapsed  = (time.time() - t0) / 60

            log_entry = {
                "epoch":    epoch,
                "step":     global_step,
                "train_loss": round(avg_loss, 4),
                **metrics,
                "elapsed_min": round(elapsed, 1),
            }
            with self.log_path.open("a") as fh:
                fh.write(json.dumps(log_entry) + "\n")

            log.info(f"  Epoch {epoch}/{self.cfg.num_epochs}  "
                     f"train_loss={avg_loss:.4f}  "
                     f"val_f1={metrics['val_f1']:.4f}  "
                     f"val_loss={metrics['val_loss']:.4f}  "
                     f"({elapsed:.1f}min)")

            # Best checkpoint
            if metrics["val_f1"] > best_val_f1:
                best_val_f1  = metrics["val_f1"]
                patience_cnt = 0
                torch.save(self.model.state_dict(), ckpt_path)
                log.info(f"    ** New best: {best_val_f1:.4f} — saved to {ckpt_path}")
            else:
                patience_cnt += 1
                if patience_cnt >= self.cfg.patience:
                    log.info(f"  Early stopping at epoch {epoch} "
                             f"(no improvement for {self.cfg.patience} epochs)")
                    break

        return {"best_val_f1": best_val_f1, "checkpoint": str(ckpt_path)}
