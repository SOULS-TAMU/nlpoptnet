#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import hashlib
import sys
import time
from typing import Any, Callable, Mapping, Optional

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "nlpopt" / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.testcase import run_helpers as unified  # noqa: E402
from scripts.factory.nonconvx_factory import (  # noqa: E402
    DATASET_KIND,
    NonconvexGenerator,
    build_problem_generator,
    build_problem_model,
    build_problem_model_from_data,
    normalize_problem_type,
)
from scripts.misc.optimizer_profile import enrich_optimizer_generation_metadata  # noqa: E402
from scripts.misc.inequality_multipliers import coerce_ineq_multipliers  # noqa: E402
from scripts.testcase import poly_run as poly_nlpopt  # noqa: E402

jax.config.update("jax_enable_x64", True)

SCHEMA_VERSION = 2
_LOCAL_REQUIRED_KEYS = ("type", "n_y", "n_eq", "n_ineq", "num_samples", "seed", "is_diag_Q")


@dataclass(frozen=True)
class DatasetBundle:
    dataset_dir: Path
    dataset_id: str
    generated: bool
    X: np.ndarray
    Y: np.ndarray
    Mu: np.ndarray
    metadata: dict[str, Any]


def _case_workspace() -> Path:
    case_dir = ROOT / "case" / "nonconvx"
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def _local_paths() -> tuple[Path, Path, Path]:
    case_dir = _case_workspace()
    return case_dir / "data.json", case_dir / "config.json", case_dir / "proj.json"


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _load_local_configs() -> tuple[dict, dict, dict]:
    data_path, cfg_path, proj_path = _local_paths()
    if not data_path.exists() or not cfg_path.exists() or not proj_path.exists():
        raise FileNotFoundError(
            "Expected case/nonconvx/data.json, case/nonconvx/config.json, and case/nonconvx/proj.json."
        )
    return _load_json(data_path), _load_json(cfg_path), _load_json(proj_path)


def _json_hash(payload: dict) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:12]


def _normalize_local_data_cfg(data_cfg: Mapping[str, Any]) -> dict[str, Any]:
    if all(key in data_cfg for key in _LOCAL_REQUIRED_KEYS):
        raw_type = str(data_cfg["type"])
        n_y = int(data_cfg["n_y"])
        n_eq = int(data_cfg["n_eq"])
        n_ineq = int(data_cfg["n_ineq"])
        num_samples = int(data_cfg["num_samples"])
        seed = int(data_cfg["seed"])
        is_diag_q = bool(data_cfg.get("is_diag_Q", True))
        force_regenerate = bool(data_cfg.get("force_regenerate", False))
        schema_version = int(data_cfg.get("schema_version", SCHEMA_VERSION))
    elif all(key in data_cfg for key in ("type", "n", "me", "mi", "num_samples", "seed")):
        raw_type = str(data_cfg["type"])
        n_y = int(data_cfg["n"])
        n_eq = int(data_cfg["me"])
        n_ineq = int(data_cfg["mi"])
        num_samples = int(data_cfg["num_samples"])
        seed = int(data_cfg["seed"])
        is_diag_q = bool(data_cfg.get("is_diag_Q", True))
        force_regenerate = bool(data_cfg.get("force_regenerate", False))
        schema_version = int(data_cfg.get("schema_version", SCHEMA_VERSION))
    else:
        missing = [key for key in _LOCAL_REQUIRED_KEYS if key not in data_cfg]
        raise ValueError(f"case/nonconvx/data.json is missing required keys: {', '.join(sorted(missing))}")

    normalized = {
        "type": normalize_problem_type(raw_type),
        "n_y": n_y,
        "n_eq": n_eq,
        "n_ineq": n_ineq,
        "num_samples": num_samples,
        "seed": seed,
        "is_diag_Q": is_diag_q,
        "force_regenerate": force_regenerate,
        "schema_version": schema_version,
    }
    if normalized["n_y"] <= 0:
        raise ValueError("n_y must be positive.")
    if normalized["n_eq"] < 0 or normalized["n_ineq"] < 0:
        raise ValueError("n_eq and n_ineq must be nonnegative.")
    if normalized["n_eq"] > normalized["n_y"]:
        raise ValueError("n_eq cannot exceed n_y.")
    if normalized["num_samples"] <= 0:
        raise ValueError("num_samples must be positive.")
    return normalized


def _translated_poly_cfg(local_cfg: Mapping[str, Any]) -> dict[str, Any]:
    local = _normalize_local_data_cfg(local_cfg)
    p = int(local["n_eq"])
    return {
        "type": DATASET_KIND,
        "p": p,
        "n": int(local["n_y"]),
        "me": int(local["n_eq"]),
        "mi": int(local["n_ineq"]),
        "num_samples": int(local["num_samples"]),
        "seed": int(local["seed"]),
        "x_L": [-1.0] * p,
        "x_U": [1.0] * p,
        "is_diag_Q": bool(local["is_diag_Q"]),
        "force_regenerate": bool(local.get("force_regenerate", False)),
    }


def _canonical_data_cfg(data_cfg: Mapping[str, Any]) -> dict[str, Any]:
    local = _normalize_local_data_cfg(data_cfg)
    return {
        "type": local["type"],
        "n_y": local["n_y"],
        "n_eq": local["n_eq"],
        "n_ineq": local["n_ineq"],
        "num_samples": local["num_samples"],
        "seed": local["seed"],
        "is_diag_Q": local["is_diag_Q"],
        "schema_version": local["schema_version"],
    }


def build_dataset_id(data_cfg: Mapping[str, Any]) -> str:
    canonical = _canonical_data_cfg(data_cfg)
    stem = (
        f"{canonical['type']}_ny{canonical['n_y']}"
        f"_neq{canonical['n_eq']}_nineq{canonical['n_ineq']}"
        f"_ns{canonical['num_samples']}_seed{canonical['seed']}"
        f"_{'diag' if canonical['is_diag_Q'] else 'dense'}"
    )
    return f"{stem}_{_json_hash(canonical)}"


def dataset_dir(case_dir: Path, data_cfg: Mapping[str, Any]) -> Path:
    return case_dir / "problem_data" / DATASET_KIND / build_dataset_id(data_cfg)


def _paths(base: Path) -> dict[str, Path]:
    return {
        "arrays": base / "dataset.npz",
        "parameters_csv": base / "parameters.csv",
        "variables_csv": base / "variables.csv",
        "ineq_multipliers_csv": base / "ineq_multipliers.csv",
        "metadata": base / "metadata.json",
        "data": base / "data.json",
        "problem_data": base / "problem_data.npz",
    }


def _write_csv(path: Path, arr: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(np.asarray(arr).tolist())


def _dataset_exists(base: Path) -> bool:
    paths = _paths(base)
    return all(
        path.exists()
        for path in (
            paths["arrays"],
            paths["parameters_csv"],
            paths["variables_csv"],
            paths["ineq_multipliers_csv"],
            paths["metadata"],
            paths["data"],
        )
    )


def _save_dataset(
    base: Path,
    *,
    data_cfg: Mapping[str, Any],
    X: np.ndarray,
    Y: np.ndarray,
    Mu: np.ndarray,
    metadata: Mapping[str, Any],
    problem_data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    base.mkdir(parents=True, exist_ok=True)
    paths = _paths(base)
    np.savez(paths["arrays"], X=np.asarray(X, dtype=np.float64), Y=np.asarray(Y, dtype=np.float64), Mu=np.asarray(Mu, dtype=np.float64))
    _write_csv(paths["parameters_csv"], X)
    _write_csv(paths["variables_csv"], Y)
    _write_csv(paths["ineq_multipliers_csv"], Mu)
    _write_json(paths["data"], _canonical_data_cfg(data_cfg))
    if problem_data is not None:
        np.savez(paths["problem_data"], **{key: np.asarray(value) for key, value in dict(problem_data).items()})
    artifact_paths = [
        paths["arrays"],
        paths["parameters_csv"],
        paths["variables_csv"],
        paths["ineq_multipliers_csv"],
        paths["data"],
    ]
    if problem_data is not None:
        artifact_paths.append(paths["problem_data"])
    enriched_metadata = enrich_optimizer_generation_metadata(
        metadata,
        num_points=int(np.asarray(X).shape[0]),
        artifact_paths=artifact_paths,
    )
    _write_json(paths["metadata"], enriched_metadata)
    return enriched_metadata


def _load_dataset(base: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    paths = _paths(base)
    arrays = np.load(paths["arrays"])
    metadata = _load_json(paths["metadata"])
    return np.asarray(arrays["X"]), np.asarray(arrays["Y"]), np.asarray(arrays["Mu"]), metadata


def ensure_cached_dataset(
    case_dir: Path,
    data_cfg: Mapping[str, Any],
    generate_fn: Callable[[], tuple],
    *,
    force: bool = False,
) -> DatasetBundle:
    local_cfg = _normalize_local_data_cfg(data_cfg)
    base = dataset_dir(case_dir, local_cfg)
    dataset_id = build_dataset_id(local_cfg)
    if _dataset_exists(base) and not force:
        X, Y, Mu, metadata = _load_dataset(base)
        return DatasetBundle(dataset_dir=base, dataset_id=dataset_id, generated=False, X=X, Y=Y, Mu=Mu, metadata=metadata)

    generated = generate_fn()
    if len(generated) == 3:
        X, Y, metadata = generated
        Mu = np.zeros((int(np.asarray(X).shape[0]), 0), dtype=np.float64)
        problem_data = None
    elif len(generated) == 4:
        X, Y, Mu, metadata = generated
        problem_data = None
    elif len(generated) == 5:
        X, Y, Mu, metadata, problem_data = generated
    else:
        raise ValueError(
            "Dataset generator must return (X, Y, metadata), (X, Y, Mu, metadata), "
            "or (X, Y, Mu, metadata, problem_data)."
        )
    metadata = _save_dataset(base, data_cfg=local_cfg, X=X, Y=Y, Mu=Mu, metadata=metadata, problem_data=problem_data)
    return DatasetBundle(dataset_dir=base, dataset_id=dataset_id, generated=True, X=np.asarray(X), Y=np.asarray(Y), Mu=np.asarray(Mu), metadata=dict(metadata))

def _generate_dataset(generator: NonconvexGenerator, data_cfg: Mapping[str, Any]):
    start_time = time.perf_counter()
    local_cfg = _normalize_local_data_cfg(data_cfg)
    target = int(local_cfg["num_samples"])
    kept_x: list[np.ndarray] = []
    kept_y: list[np.ndarray] = []
    kept_mu: list[np.ndarray] = []
    objectives: list[float] = []
    status_counts: dict[str, int] = {}
    attempts = 0
    max_attempts = max(4 * target, target + 100)

    while len(kept_x) < target and attempts < max_attempts:
        batch_size = min(max(32, target - len(kept_x)), target)
        xs = generator.sample_parameters(batch_size)
        for x in xs:
            result = generator.solve_for_x(x)
            attempts += 1
            status = str(result["status"])
            status_counts[status] = status_counts.get(status, 0) + 1
            if status == "optimal" and result["y"] is not None:
                kept_x.append(np.asarray(x, dtype=np.float64))
                kept_y.append(np.asarray(result["y"], dtype=np.float64))
                kept_mu.append(coerce_ineq_multipliers(result.get("mu"), generator.n_ineq))
                objectives.append(float(result["objective"]) if result["objective"] is not None else np.nan)
                if len(kept_x) >= target:
                    break
        if len(kept_x) >= target:
            break

    if len(kept_x) < target:
        raise RuntimeError(
            f"Only collected {len(kept_x)} successful nonconvex points out of requested {target}. "
            "Increase num_samples or improve solver robustness."
        )

    metadata = {
        "problem_type": DATASET_KIND,
        "n_x": int(generator.n_x),
        "n_y": int(generator.n_y),
        "n_eq": int(generator.n_eq),
        "n_ineq": int(generator.n_ineq),
        "num_samples": target,
        "seed": int(local_cfg["seed"]),
        "solver": str(generator.requested_solver),
        "is_diag_Q": bool(local_cfg["is_diag_Q"]),
        "objective_min": float(np.nanmin(objectives)) if objectives else np.nan,
        "objective_max": float(np.nanmax(objectives)) if objectives else np.nan,
        "objective_mean": float(np.nanmean(objectives)) if objectives else np.nan,
        "status_counts": status_counts,
        "attempts": int(attempts),
        "optimizer_generation_wall_time_sec": time.perf_counter() - start_time,
    }
    return (
        np.asarray(kept_x, dtype=np.float64),
        np.asarray(kept_y, dtype=np.float64),
        np.stack(kept_mu, axis=0) if kept_mu else np.zeros((len(kept_x), generator.n_ineq), dtype=np.float64),
        metadata,
        generator.get_problem_data(),
    )


@contextmanager
def _patch_poly_nlpopt_for_nonconvex():
    original_normalize_problem_type = poly_nlpopt.normalize_problem_type
    original_build_problem_generator = poly_nlpopt.build_problem_generator
    original_build_problem_model = poly_nlpopt.build_problem_model
    original_build_problem_model_from_data = poly_nlpopt.build_problem_model_from_data
    original_uses_nonconvex_generator = poly_nlpopt.uses_nonconvex_generator
    original_ensure_cached_dataset = poly_nlpopt.ensure_cached_dataset

    poly_nlpopt.normalize_problem_type = normalize_problem_type
    poly_nlpopt.build_problem_generator = build_problem_generator
    poly_nlpopt.build_problem_model = lambda data_cfg, dtype=jnp.float64: build_problem_model(data_cfg, dtype=dtype)
    poly_nlpopt.build_problem_model_from_data = lambda problem_data, dtype=jnp.float64: build_problem_model_from_data(problem_data, dtype=dtype)
    poly_nlpopt.uses_nonconvex_generator = lambda _data_cfg: True
    poly_nlpopt.ensure_cached_dataset = ensure_cached_dataset
    try:
        yield
    finally:
        poly_nlpopt.normalize_problem_type = original_normalize_problem_type
        poly_nlpopt.build_problem_generator = original_build_problem_generator
        poly_nlpopt.build_problem_model = original_build_problem_model
        poly_nlpopt.build_problem_model_from_data = original_build_problem_model_from_data
        poly_nlpopt.uses_nonconvex_generator = original_uses_nonconvex_generator
        poly_nlpopt.ensure_cached_dataset = original_ensure_cached_dataset


def _run_nlpopt(
    case_dir: Path,
    data_cfg: dict,
    cfg_dict: dict,
    proj_cfg: dict,
    *,
    output_dir: Path,
) -> unified.RunArtifacts:
    translated_cfg = _translated_poly_cfg(data_cfg)
    with _patch_poly_nlpopt_for_nonconvex():
        poly_nlpopt.run_case(
            case_dir,
            data_cfg_override=translated_cfg,
            cfg_dict_override=unified._training_cfg_only(cfg_dict),
            proj_cfg_override=proj_cfg,
            output_dir_override=output_dir,
        )
    dataset_root = dataset_dir(case_dir, data_cfg)
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        summary_payload = _load_json(summary_path)
        summary_payload["space_mb"] = unified._directory_size_mb(output_dir)
        summary_payload["dataset_dir"] = str(dataset_root)
        summary_payload["save_dir"] = str(output_dir)
        _write_json(summary_path, summary_payload)
    return unified.RunArtifacts(
        framework="nlpopt",
        dataset_dir=dataset_root,
        run_dir=output_dir,
        history_path=output_dir / "run_history.json",
        metrics_path=summary_path,
        plot_path=output_dir / "training_metrics.png",
    )


def _print_run_header(case_dir: Path, data_cfg: dict, cfg_dict: dict, framework: str) -> None:
    dataset_target = dataset_dir(case_dir, data_cfg)
    print("=" * 80)
    print(f"Standalone runner | framework={unified._framework_label(framework)}")
    print(
        f"NONCONVEX  n_x={int(data_cfg['n_eq'])} n_y={int(data_cfg['n_y'])} "
        f"n_eq={int(data_cfg['n_eq'])} n_ineq={int(data_cfg['n_ineq'])}"
    )
    print(f"Workspace: {case_dir}")
    print(f"Dataset target: {dataset_target}")
    print(
        f"Config: seed={int(cfg_dict.get('seed', 42))} "
        f"epochs={int(cfg_dict.get('epochs', 1000))} "
        f"batch_size={int(cfg_dict.get('batch_size', 200))} "
        f"lr={float(cfg_dict.get('learning_rate', 1e-4)):.3e}"
    )
    print("=" * 80)


def _run_single_case(
    case_dir: Path,
    data_cfg: dict,
    cfg_dict: dict,
    proj_cfg: dict,
    *,
    output_dir_override: Optional[Path] = None,
) -> unified.RunArtifacts:
    data_cfg = _normalize_local_data_cfg(data_cfg)
    framework = unified._normalize_model_name(str(cfg_dict.get("model", "nlpopt")))
    _print_run_header(case_dir, data_cfg, cfg_dict, framework)
    dataset_root = dataset_dir(case_dir, data_cfg)
    output_dir = Path(output_dir_override) if output_dir_override is not None else unified._framework_dir(dataset_root, framework)
    return _run_nlpopt(case_dir, data_cfg, cfg_dict, proj_cfg, output_dir=output_dir)


def default_dataset_dir() -> Path:
    data_cfg, _cfg, _proj = _load_local_configs()
    return dataset_dir(_case_workspace(), data_cfg)


def main() -> int:
    data_cfg, cfg_dict, proj_cfg = _load_local_configs()
    data_cfg = _normalize_local_data_cfg(data_cfg)
    case_dir = _case_workspace()
    cfg_dict["model"] = "nlpopt"

    artifacts = _run_single_case(case_dir, data_cfg, cfg_dict, proj_cfg)
    metadata_path = unified._append_family_metadata(
        artifacts.dataset_dir,
        mode="single_model",
        output_dir=artifacts.run_dir,
        data_cfg=data_cfg,
        cfg_dict=cfg_dict,
        proj_cfg=proj_cfg,
        framework=artifacts.framework,
        seeds=[int(cfg_dict.get("seed", 42))],
        extra={"summary_path": str(artifacts.metrics_path), "history_path": str(artifacts.history_path)},
    )
    print(f"[run] Updated family metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
