"""I/O utilities: config loading, CSV read/write, checkpoint helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str | Path = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ── CSV ───────────────────────────────────────────────────────────────────────

def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8")


# ── JSONL (for incremental augmentation writes) ───────────────────────────────

def append_jsonl(record: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


# ── JSON ──────────────────────────────────────────────────────────────────────

def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
