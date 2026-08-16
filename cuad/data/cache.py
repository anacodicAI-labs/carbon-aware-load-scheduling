from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def cache_path(raw_dir: Path, name: str) -> Path:
    return raw_dir / name


def load_cached_csv(path: Path) -> pd.DataFrame | None:
    if path.is_file():
        return pd.read_csv(path)
    return None


def save_cached_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def config_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(raw.encode()).hexdigest()[:16]
