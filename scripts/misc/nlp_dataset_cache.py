from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np

from scripts.misc.optimizer_profile import enrich_optimizer_generation_metadata
from scripts.misc.solver_config import resolve_solver_name

_SUPPORTED_PROBLEM_TYPES = {"nlp"}


def normalize_problem_type(problem_type: str) -> str:
    normalized = str(problem_type).strip().lower()
    if normalized not in _SUPPORTED_PROBLEM_TYPES:
        raise ValueError(
            f"Unsupported problem type '{problem_type}'. "
            f"Supported types: {', '.join(sorted(_SUPPORTED_PROBLEM_TYPES))}."
        )
    return normalized


SCHEMA_VERSION = 10


@dataclass(frozen=True)
class DatasetBundle:
    dataset_dir: Path
    dataset_id: str
    generated: bool
    X: np.ndarray
    Y: np.ndarray
    Mu: np.ndarray
    metadata: Dict[str, Any]


def _canonical_data_cfg(data_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    canonical = dict(data_cfg)
    canonical["type"] = normalize_problem_type(str(data_cfg["type"]))
    canonical["schema_version"] = SCHEMA_VERSION
    canonical["n_x"] = int(data_cfg["n_x"])
    canonical["n_y"] = int(data_cfg["n_y"])
    canonical["n_eq"] = int(data_cfg["n_eq"])
    canonical["n_ineq"] = int(data_cfg["n_ineq"])
    canonical["N_samples"] = int(data_cfg["N_samples"])
    canonical["N_points"] = int(data_cfg["N_points"])
    canonical["seed"] = int(data_cfg["seed"])
    canonical["x_L"] = [float(v) for v in data_cfg["x_L"]]
    canonical["x_U"] = [float(v) for v in data_cfg["x_U"]]
    canonical["solver"] = resolve_solver_name(data_cfg, default="SCS")
    canonical["is_diag_Q"] = bool(data_cfg.get("is_diag_Q", False))
    canonical["q_diag_shift"] = float(data_cfg.get("q_diag_shift", 0.5))
    canonical["nl_margin"] = float(data_cfg.get("nl_margin", 1.0))
    canonical["bound_margin"] = float(data_cfg.get("bound_margin", 1.0))
    canonical["bound_scale"] = float(data_cfg.get("bound_scale", 0.2))
    canonical["param_scale"] = float(data_cfg.get("param_scale", 0.4))
    return canonical


def _dataset_hash(data_cfg: Mapping[str, Any]) -> str:
    payload = json.dumps(_canonical_data_cfg(data_cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def build_dataset_id(data_cfg: Mapping[str, Any]) -> str:
    canonical = _canonical_data_cfg(data_cfg)
    stem = (
        f"{canonical['type']}_nx{canonical['n_x']}_ny{canonical['n_y']}"
        f"_neq{canonical['n_eq']}_nineq{canonical['n_ineq']}"
        f"_np{canonical['N_points']}_seed{canonical['seed']}"
    )
    return f"{stem}_{_dataset_hash(canonical)}"


def dataset_dir(case_dir: Path, data_cfg: Mapping[str, Any]) -> Path:
    canonical = _canonical_data_cfg(data_cfg)
    problem_data_root = os.environ.get("NLP_OPT_PROBLEM_DATA_ROOT")
    base_root = Path(problem_data_root) if problem_data_root else (case_dir / "problem_data")
    return base_root / canonical["type"] / build_dataset_id(canonical)


def _paths(base: Path) -> Dict[str, Path]:
    return {
        "arrays": base / "dataset.npz",
        "parameters_csv": base / "parameters.csv",
        "variables_csv": base / "variables.csv",
        "ineq_multipliers_csv": base / "ineq_multipliers.csv",
        "metadata": base / "metadata.json",
        "data_config": base / "data_config.json",
        "problem_data": base / "problem_data.npz",
    }


def dataset_exists(base: Path) -> bool:
    paths = _paths(base)
    return all(path.exists() for path in paths.values())


def _write_csv(path: Path, arr: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(arr.tolist())


def save_dataset(
    base: Path,
    *,
    data_cfg: Mapping[str, Any],
    X: np.ndarray,
    Y: np.ndarray,
    Mu: np.ndarray,
    metadata: Mapping[str, Any],
    problem_data: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    base.mkdir(parents=True, exist_ok=True)
    paths = _paths(base)
    np.savez(paths["arrays"], X=X, Y=Y, Mu=Mu)
    _write_csv(paths["parameters_csv"], X)
    _write_csv(paths["variables_csv"], Y)
    _write_csv(paths["ineq_multipliers_csv"], Mu)
    with open(paths["data_config"], "w", encoding="utf-8") as fh:
        json.dump(_canonical_data_cfg(data_cfg), fh, indent=2, sort_keys=True)
    if problem_data is not None:
        np.savez(paths["problem_data"], **{k: np.asarray(v) for k, v in dict(problem_data).items()})
    artifact_paths = [
        paths["arrays"],
        paths["parameters_csv"],
        paths["variables_csv"],
        paths["ineq_multipliers_csv"],
        paths["data_config"],
    ]
    if problem_data is not None:
        artifact_paths.append(paths["problem_data"])
    enriched_metadata = enrich_optimizer_generation_metadata(
        metadata,
        num_points=int(np.asarray(X).shape[0]),
        artifact_paths=artifact_paths,
    )
    with open(paths["metadata"], "w", encoding="utf-8") as fh:
        json.dump(enriched_metadata, fh, indent=2, sort_keys=True)
    return enriched_metadata


def load_dataset(base: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    paths = _paths(base)
    arrays = np.load(paths["arrays"])
    with open(paths["metadata"], "r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    return np.asarray(arrays["X"]), np.asarray(arrays["Y"]), np.asarray(arrays["Mu"]), metadata


def ensure_cached_dataset(
    case_dir: Path,
    data_cfg: Mapping[str, Any],
    generate_fn: Callable[[], Tuple],
    *,
    force: bool = False,
) -> DatasetBundle:
    base = dataset_dir(case_dir, data_cfg)
    dataset_id = build_dataset_id(data_cfg)
    if dataset_exists(base) and not force:
        X, Y, Mu, metadata = load_dataset(base)
        return DatasetBundle(dataset_dir=base, dataset_id=dataset_id, generated=False, X=X, Y=Y, Mu=Mu, metadata=metadata)

    generated = generate_fn()
    if len(generated) == 4:
        X, Y, metadata, problem_data = generated
        Mu = np.zeros((int(np.asarray(X).shape[0]), 0), dtype=np.float64)
    elif len(generated) == 5:
        X, Y, Mu, metadata, problem_data = generated
    else:
        raise ValueError(
            "Dataset generator must return (X, Y, metadata, problem_data) "
            "or (X, Y, Mu, metadata, problem_data)."
        )
    metadata = save_dataset(base, data_cfg=data_cfg, X=X, Y=Y, Mu=Mu, metadata=metadata, problem_data=problem_data)
    return DatasetBundle(dataset_dir=base, dataset_id=dataset_id, generated=True, X=X, Y=Y, Mu=Mu, metadata=dict(metadata))
