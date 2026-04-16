from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np


def resolve_dtype(name: str):
    normalized = str(name).strip().lower()
    if normalized in {"float32", "fp32", "32"}:
        return jnp.float32
    if normalized in {"float64", "fp64", "64"}:
        return jnp.float64
    raise ValueError(f"Unsupported dtype '{name}'.")


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def json_safe(value: Any):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(json_safe(payload), fh, indent=2, sort_keys=True)
