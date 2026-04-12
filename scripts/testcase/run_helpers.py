from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


UNIFIED_CFG_KEYS = {"model"}


@dataclass(frozen=True)
class RunArtifacts:
    framework: str
    dataset_dir: Path
    run_dir: Path
    history_path: Path
    metrics_path: Path
    plot_path: Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _write_run_configs(run_dir: Path, data_cfg: dict, cfg_dict: dict, proj_cfg: dict) -> None:
    _write_json(run_dir / "data.json", data_cfg)
    _write_json(run_dir / "config.json", cfg_dict)
    _write_json(run_dir / "proj.json", proj_cfg)


def _append_family_metadata(
    dataset_dir: Path,
    *,
    mode: str,
    output_dir: Path,
    data_cfg: dict,
    cfg_dict: dict,
    proj_cfg: dict,
    framework: str | None = None,
    frameworks: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    extra: dict | None = None,
) -> Path:
    family_dir = dataset_dir.parent
    metadata_path = family_dir / "metadata.json"
    payload = {"entries": []}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        if isinstance(loaded, dict) and isinstance(loaded.get("entries"), list):
            payload = loaded

    next_entry_id = 1 + max((int(entry.get("entry_id", 0)) for entry in payload["entries"]), default=0)
    entry = {
        "entry_id": next_entry_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": mode,
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "problem_type": str(data_cfg["type"]).lower(),
        "framework": framework,
        "frameworks": list(frameworks) if frameworks is not None else None,
        "seeds": [int(seed) for seed in seeds] if seeds is not None else None,
        "data_config": copy.deepcopy(data_cfg),
        "config": copy.deepcopy(cfg_dict),
        "proj_config": copy.deepcopy(proj_cfg),
    }
    if extra:
        entry["extra"] = copy.deepcopy(extra)
    payload["entries"].append(entry)
    _write_json(metadata_path, payload)
    return metadata_path


def _str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in {"false", "f", "0", "no", "n"}:
        return False
    if lowered in {"true", "t", "1", "yes", "y"}:
        return True
    raise ValueError(f"{value!r} is not a valid boolean value")


def _normalize_model_name(model_name: str) -> str:
    normalized = str(model_name).strip().lower()
    if normalized != "nlpopt":
        raise ValueError("Only model='nlpopt' is supported in the run-only codespace.")
    return "nlpopt"


def _framework_label(framework: str) -> str:
    if framework != "nlpopt":
        raise ValueError("Only the NLPOpt framework is supported.")
    return "NLPOpt"


def _framework_dir(dataset_dir: Path, framework: str) -> Path:
    _normalize_model_name(framework)
    return dataset_dir / "nlpopt"


def _framework_multi_dir(dataset_dir: Path, framework: str) -> Path:
    return _framework_dir(dataset_dir, framework) / "multi"


def _framework_seed_dir(dataset_dir: Path, framework: str, seed: int) -> Path:
    return _framework_multi_dir(dataset_dir, framework) / str(int(seed))


def _directory_size_mb(path: Path) -> float:
    total_bytes = 0
    if path.exists():
        for child in path.iterdir():
            if child.is_file():
                total_bytes += int(child.stat().st_size)
    return float(total_bytes) / (1024.0 * 1024.0)


def _training_cfg_only(cfg_dict: dict) -> dict:
    return {key: value for key, value in cfg_dict.items() if key not in UNIFIED_CFG_KEYS}


def _num_batches(num_items: int, batch_size: int) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    return max(1, (int(num_items) + int(batch_size) - 1) // int(batch_size))
