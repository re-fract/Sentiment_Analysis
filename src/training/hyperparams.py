"""
src/training/hyperparams.py

Hyperparameter configs for all transformer training experiments.

Experiment matrix:
  T1: BERT-base       + SemEval only
  T2: BERT-base       + SemEval + Augmented
  T3: RoBERTa-base    + SemEval only
  T4: RoBERTa-base    + SemEval + Augmented
  T5: DistilBERT-base + SemEval only
  T6: DistilBERT-base + SemEval + Augmented
  T7: RoBERTa-large   + SemEval only       (new — improved)
  T8: RoBERTa-large   + SemEval + Augmented (new — improved)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    # Experiment identity
    experiment_id:  str = "T1"
    model_name:     str = "bert-base-uncased"
    data_config:    str = "semeval_only"   # "semeval_only" | "augmented"

    # Paths
    checkpoint_dir: str = "outputs/models"

    # Tokenizer / model
    max_seq_length: int = 128

    # Optimiser
    learning_rate:  float = 2e-5
    weight_decay:   float = 0.01
    warmup_ratio:   float = 0.10

    # Training loop
    num_epochs:     int   = 10
    batch_size:     int   = 16
    grad_accum_steps:int  = 2    # effective batch = 32
    max_grad_norm:  float = 1.0
    fp16:           bool  = True   # use fp16 on P100

    # Regularisation
    label_smoothing: float = 0.1   # applied to both aspect BCE and sentiment CE

    # Multi-task learning (T9/T10)
    mtl_alpha:      float = 0.5    # weight for aspect loss; (1-alpha) for sentiment
    with_implicit:  bool  = False  # enable implicit-aspect head (T10)

    # Early stopping
    patience:       int   = 3     # epochs without val improvement

    # Evaluation
    eval_every_n_steps: int = 100
    default_threshold: float = 0.50  # for multi-label aspect detection


# Experiment configurations
EXPERIMENTS: dict[str, dict] = {
    "T1": dict(experiment_id="T1", model_name="bert-base-uncased",       data_config="semeval_only"),
    "T2": dict(experiment_id="T2", model_name="bert-base-uncased",       data_config="augmented"),
    "T3": dict(experiment_id="T3", model_name="roberta-base",            data_config="semeval_only"),
    "T4": dict(experiment_id="T4", model_name="roberta-base",            data_config="augmented"),
    "T5": dict(experiment_id="T5", model_name="distilbert-base-uncased", data_config="semeval_only"),
    "T6": dict(experiment_id="T6", model_name="distilbert-base-uncased", data_config="augmented"),
    # roberta-large: smaller batch + more grad accumulation to fit P100 16GB
    "T7": dict(experiment_id="T7", model_name="roberta-large",           data_config="semeval_only",
               batch_size=8, grad_accum_steps=4, learning_rate=1e-5),
    "T8": dict(experiment_id="T8", model_name="roberta-large",           data_config="augmented",
               batch_size=8, grad_accum_steps=4, learning_rate=1e-5),
    # Joint multi-task models
    "T9":  dict(experiment_id="T9",  model_name="roberta-base",  data_config="semeval_only"),
    "T10": dict(experiment_id="T10", model_name="roberta-base",  data_config="full_train",
                with_implicit=True),
}


def get_config(experiment_id: str) -> TrainingConfig:
    """Return a TrainingConfig for the given experiment ID."""
    if experiment_id not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_id}. "
                         f"Valid: {list(EXPERIMENTS.keys())}")
    return TrainingConfig(**EXPERIMENTS[experiment_id])
