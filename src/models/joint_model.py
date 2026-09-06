"""
src/models/joint_model.py

Joint multi-task ABSA model with a shared encoder and task-specific heads.

Design
------
  Rather than wrapping AutoModel and remapping keys at save time, we use two
  separate AutoModelForSequenceClassification instances that share their
  encoder weights via tied parameters. This means:

    - asp_model  : AutoModelForSequenceClassification(num_labels=5,  problem_type="multi_label_classification")
    - sent_model : AutoModelForSequenceClassification(num_labels=3)
    - imp_model  : AutoModelForSequenceClassification(num_labels=5,  problem_type="multi_label_classification") [T10]

  After building, we tie all encoder layers so a single backward pass through
  either head updates the shared backbone.

  Checkpointing is trivial: save asp_model.state_dict() → aspect checkpoint,
  save sent_model.state_dict() → sentiment checkpoint. These are exactly the
  formats that AspectDetector and SentimentClassifier already load.

Forward call
------------
  forward(..., task="aspect")    → asp_model logits
  forward(..., task="sentiment") → sent_model logits
  forward(..., task="implicit")  → imp_model logits (T10)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from src.utils.logging import get_logger

log = get_logger("joint_model")

NUM_ASPECTS    = 5   # coarse categories
NUM_SENTIMENTS = 3   # positive / negative / neutral


def _share_encoder(source_model, target_model) -> None:
    """
    Point every encoder parameter in `target_model` to the same tensor as
    `source_model`. After this, a gradient update to source also updates target.

    Works for BERT-family models where the backbone attribute is named
    `bert`, `roberta`, or `distilbert`.
    """
    src_enc = _get_backbone(source_model)
    tgt_enc = _get_backbone(target_model)

    src_params = dict(src_enc.named_parameters())
    for name, param in tgt_enc.named_parameters(recurse=False):
        if name in src_params:
            # Replace in-place so the computation graph is shared
            ...

    # Simpler: replace whole modules
    for attr in ["embeddings", "encoder", "pooler"]:
        if hasattr(src_enc, attr) and hasattr(tgt_enc, attr):
            setattr(tgt_enc, attr, getattr(src_enc, attr))


def _get_backbone(model) -> nn.Module:
    """Return the transformer backbone (bert / roberta / distilbert sub-module)."""
    for attr in ("roberta", "bert", "distilbert"):
        if hasattr(model, attr):
            return getattr(model, attr)
    raise AttributeError(f"Cannot find backbone in {type(model).__name__}")


def _supports_token_type_ids(model) -> bool:
    return hasattr(model.config, "type_vocab_size")


class JointABSAModel(nn.Module):
    """
    Joint multi-task model: two (or three) classification heads over a
    single shared transformer encoder.

    Parameters
    ----------
    model_name     : HuggingFace model identifier (e.g. "roberta-base")
    with_implicit  : enable implicit-aspect head (T10)
    """

    def __init__(self, model_name: str, with_implicit: bool = False):
        super().__init__()
        self.model_name    = model_name
        self.with_implicit = with_implicit

        # ── Aspect head model ─────────────────────────────────────────────────
        self.asp_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=NUM_ASPECTS,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True,
        )

        # ── Sentiment head model (shares encoder with asp_model) ──────────────
        self.sent_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=NUM_SENTIMENTS,
            ignore_mismatched_sizes=True,
        )
        _share_encoder(self.asp_model, self.sent_model)

        # ── Implicit head model (T10, shares encoder) ─────────────────────────
        self.imp_model: Optional[nn.Module] = None
        if with_implicit:
            self.imp_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=NUM_ASPECTS,
                problem_type="multi_label_classification",
                ignore_mismatched_sizes=True,
            )
            _share_encoder(self.asp_model, self.imp_model)

        log.info(
            f"JointABSAModel: {model_name}  "
            f"with_implicit={with_implicit}  "
            f"asp_labels={NUM_ASPECTS}  sent_labels={NUM_SENTIMENTS}"
        )

    # ── Forward ──────────────────────────────────────────────────────────────

    def _encode_kwargs(self, model, batch: dict) -> dict:
        kwargs = {
            "input_ids":      batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
        if "token_type_ids" in batch and _supports_token_type_ids(model):
            kwargs["token_type_ids"] = batch["token_type_ids"]
        return kwargs

    def forward(self, batch: dict, task: str = "aspect") -> torch.Tensor:
        """
        Parameters
        ----------
        batch : dict with input_ids, attention_mask, [token_type_ids]
        task  : "aspect" | "sentiment" | "implicit"

        Returns raw logits (no activation).
        """
        if task == "aspect":
            return self.asp_model(**self._encode_kwargs(self.asp_model, batch)).logits
        elif task == "sentiment":
            return self.sent_model(**self._encode_kwargs(self.sent_model, batch)).logits
        elif task == "implicit":
            if self.imp_model is None:
                raise RuntimeError("Implicit head not enabled (with_implicit=False)")
            return self.imp_model(**self._encode_kwargs(self.imp_model, batch)).logits
        else:
            raise ValueError(f"Unknown task: {task!r}")

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def save_aspect_checkpoint(self, path: str) -> None:
        """Save asp_model state dict — identical format to AspectDetector checkpoints."""
        torch.save(self.asp_model.state_dict(), path)
        log.info(f"Aspect checkpoint → {path}")

    def save_sentiment_checkpoint(self, path: str) -> None:
        """Save sent_model state dict — identical format to SentimentClassifier checkpoints."""
        torch.save(self.sent_model.state_dict(), path)
        log.info(f"Sentiment checkpoint → {path}")


def build_joint_model(model_name: str, with_implicit: bool = False) -> JointABSAModel:
    return JointABSAModel(model_name, with_implicit=with_implicit)
