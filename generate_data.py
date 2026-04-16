#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "nlpopt" / "src"

for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


_TYPE_ALIASES = {
    "nonconvx": "nonconvex",
}

_SUPPORTED_TYPES = ("qp", "qcqp", "nlp", "nonconvex")
_DIMENSION_ORDER = ("param", "decision", "eq", "ineq")
_DIMENSION_ALIASES = {
    "param": "param",
    "parameter": "param",
    "parameters": "param",
    "p": "param",
    "n_x": "param",
    "decision": "decision",
    "variable": "decision",
    "variables": "decision",
    "n": "decision",
    "n_y": "decision",
    "eq": "eq",
    "equality": "eq",
    "equalities": "eq",
    "me": "eq",
    "n_eq": "eq",
    "ineq": "ineq",
    "inequality": "ineq",
    "inequalities": "ineq",
    "mi": "ineq",
    "n_ineq": "ineq",
}


def _normalize_problem_type(problem_type: str) -> str:
    normalized = _TYPE_ALIASES.get(str(problem_type).strip().lower(), str(problem_type).strip().lower())
    if normalized not in _SUPPORTED_TYPES:
        raise ValueError(
            f"Unsupported type '{problem_type}'. Supported types: {', '.join(_SUPPORTED_TYPES)}."
        )
    return normalized


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate notebook data artifacts under notebooks/data/{problem_type}: "
            "parameters.csv, variables.csv, and problem.npz."
        )
    )
    parser.add_argument(
        "--type",
        required=True,
        choices=("qp", "qcqp", "nlp", "nonconvex", "nonconvx"),
        help="Problem family to generate.",
    )
    parser.add_argument(
        "--dimension",
        default=None,
        help=(
            "Optional dimension override. "
            "Use either keyed syntax like 'p=2,n=4,me=1,mi=1' "
            "or positional syntax like '2,4,1,1'. "
            "For NLP, p/n/me/mi are accepted as aliases for n_x/n_y/n_eq/n_ineq."
        ),
    )
    parser.add_argument(
        "--data_json",
        required=True,
        help="Path to the data.json file that seeds problem generation.",
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _write_csv(path: Path, arr: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(np.asarray(arr, dtype=np.float64).tolist())


def _resolve_path(path_arg: str) -> Path:
    candidate = Path(path_arg).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    return candidate


def _broadcast_or_validate_vector(values: list[Any], target_len: int, *, key: str) -> list[float]:
    if target_len <= 0:
        raise ValueError(f"{key} target length must be positive.")
    if len(values) == target_len:
        return [float(v) for v in values]
    if len(values) == 1:
        return [float(values[0])] * target_len
    raise ValueError(
        f"{key} must either have length 1 or match the parameter dimension {target_len}; "
        f"got length {len(values)}."
    )


def _parse_dimension_override(raw: str | None) -> dict[str, int]:
    if raw is None or str(raw).strip() == "":
        return {}

    parts = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not parts:
        return {}

    overrides: dict[str, int] = {}
    if all("=" in part for part in parts):
        for part in parts:
            key_text, value_text = part.split("=", 1)
            logical_key = _DIMENSION_ALIASES.get(key_text.strip().lower())
            if logical_key is None:
                raise ValueError(f"Unsupported dimension key '{key_text}'.")
            overrides[logical_key] = int(value_text.strip())
        return overrides

    if all("=" not in part for part in parts):
        if len(parts) not in {1, len(_DIMENSION_ORDER)}:
            raise ValueError(
                "Positional dimension syntax must provide either one value "
                "(parameter dimension only) or four values in the order "
                "param,decision,eq,ineq."
            )
        if len(parts) == 1:
            return {"param": int(parts[0])}
        return {key: int(value) for key, value in zip(_DIMENSION_ORDER, parts)}

    raise ValueError("Use either fully keyed dimension syntax or fully positional syntax.")


def _apply_dimension_override(problem_type: str, data_cfg: dict[str, Any], raw_dimension: str | None) -> dict[str, Any]:
    data = copy.deepcopy(data_cfg)
    overrides = _parse_dimension_override(raw_dimension)
    if not overrides:
        return data

    if problem_type in {"qp", "qcqp"}:
        key_map = {"param": "p", "decision": "n", "eq": "me", "ineq": "mi"}
        for logical_key, target_key in key_map.items():
            if logical_key in overrides:
                data[target_key] = int(overrides[logical_key])
        return data

    if problem_type == "nlp":
        key_map = {"param": "n_x", "decision": "n_y", "eq": "n_eq", "ineq": "n_ineq"}
        for logical_key, target_key in key_map.items():
            if logical_key in overrides:
                data[target_key] = int(overrides[logical_key])
        return data

    if problem_type == "nonconvex":
        param_dim = overrides.get("param")
        eq_dim = overrides.get("eq")
        if param_dim is not None and eq_dim is not None and int(param_dim) != int(eq_dim):
            raise ValueError("For nonconvex problems, parameter dimension must equal the equality dimension.")

        shared_eq_dim = eq_dim if eq_dim is not None else param_dim
        if shared_eq_dim is not None:
            data["p"] = int(shared_eq_dim)
            data["me"] = int(shared_eq_dim)
            if "n_eq" in data:
                data["n_eq"] = int(shared_eq_dim)
            if "n_x" in data:
                data["n_x"] = int(shared_eq_dim)
        if "decision" in overrides:
            data["n"] = int(overrides["decision"])
            if "n_y" in data:
                data["n_y"] = int(overrides["decision"])
        if "ineq" in overrides:
            data["mi"] = int(overrides["ineq"])
            if "n_ineq" in data:
                data["n_ineq"] = int(overrides["ineq"])
        return data

    raise ValueError(f"Unsupported type '{problem_type}'.")


def _parameter_dimension(problem_type: str, data_cfg: dict[str, Any]) -> int:
    if problem_type in {"qp", "qcqp"}:
        return int(data_cfg["p"])
    if problem_type == "nlp":
        return int(data_cfg["n_x"])
    if problem_type == "nonconvex":
        return int(data_cfg.get("me", data_cfg.get("n_eq", data_cfg.get("p", data_cfg.get("n_x")))))
    raise ValueError(f"Unsupported type '{problem_type}'.")


def _normalize_parameter_bounds(problem_type: str, data_cfg: dict[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(data_cfg)
    param_dim = _parameter_dimension(problem_type, data)
    if param_dim <= 0:
        raise ValueError("Parameter dimension must be positive.")

    if problem_type == "nonconvex":
        x_l_raw = list(data.get("x_L", [-1.0]))
        x_u_raw = list(data.get("x_U", [1.0]))
        data["p"] = int(param_dim)
        data["me"] = int(param_dim)
    else:
        if "x_L" not in data or "x_U" not in data:
            raise ValueError("data.json must define both x_L and x_U.")
        x_l_raw = list(data["x_L"])
        x_u_raw = list(data["x_U"])

    data["x_L"] = _broadcast_or_validate_vector(x_l_raw, param_dim, key="x_L")
    data["x_U"] = _broadcast_or_validate_vector(x_u_raw, param_dim, key="x_U")
    if any(lo > hi for lo, hi in zip(data["x_L"], data["x_U"])):
        raise ValueError("Each component of x_L must be <= the matching component of x_U.")

    if problem_type == "nlp":
        if "N_points" not in data and "N_samples" in data:
            data["N_points"] = int(data["N_samples"])
        if "N_samples" not in data and "N_points" in data:
            data["N_samples"] = int(data["N_points"])
    return data


def _prepare_data_cfg(problem_type: str, raw_data_cfg: dict[str, Any], raw_dimension: str | None) -> dict[str, Any]:
    data = copy.deepcopy(raw_data_cfg)
    file_type = data.get("type")
    if file_type is not None and _normalize_problem_type(str(file_type)) != problem_type:
        raise ValueError(f"Type mismatch: --type={problem_type} but data.json has type={file_type}.")

    data["type"] = problem_type
    data = _apply_dimension_override(problem_type, data, raw_dimension)
    data = _normalize_parameter_bounds(problem_type, data)
    return data


def _status_ok(status: str) -> bool:
    return str(status).strip().lower() in {"optimal", "optimal_inaccurate"}


def _generate_poly_dataset(data_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import jax
    import jax.numpy as jnp

    from scripts.factory.poly_factory import build_problem_data, build_problem_model_from_data
    from scripts.misc.solver_config import resolve_solver_name
    from solgen import SolGenModel

    jax.config.update("jax_enable_x64", True)

    problem_data = dict(build_problem_data(data_cfg))
    model = build_problem_model_from_data(problem_data, dtype=jnp.float64)
    solver = SolGenModel(model)
    solver_name = resolve_solver_name(data_cfg, default="SCS")

    target = int(data_cfg["num_samples"])
    rng = np.random.default_rng(int(data_cfg["seed"]))
    x_l = np.asarray(data_cfg["x_L"], dtype=np.float64)
    x_u = np.asarray(data_cfg["x_U"], dtype=np.float64)

    kept_x: list[np.ndarray] = []
    kept_y: list[np.ndarray] = []
    attempts = 0
    max_attempts = max(4 * target, target + 100)

    while len(kept_x) < target and attempts < max_attempts:
        batch_size = min(max(32, target - len(kept_x)), target)
        xs = rng.uniform(x_l, x_u, size=(batch_size, x_l.shape[0]))
        for x in xs:
            result = solver.solve({"x": jnp.asarray(x, dtype=model.dtype)}, solver=solver_name)
            attempts += 1
            if _status_ok(result.status) and result.y is not None:
                kept_x.append(np.asarray(x, dtype=np.float64))
                kept_y.append(np.asarray(result.y, dtype=np.float64))
                if len(kept_x) >= target:
                    break

    if len(kept_x) < target:
        raise RuntimeError(
            f"Only collected {len(kept_x)} successful samples out of requested {target}."
        )

    return np.asarray(kept_x, dtype=np.float64), np.asarray(kept_y, dtype=np.float64), problem_data


def _generate_nlp_dataset(data_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from scripts.factory.nlp_factory import build_problem_generator

    generator = build_problem_generator(data_cfg)
    target = int(data_cfg["N_points"])
    sampled = int(data_cfg["N_samples"])
    xs = generator.sample_parameters(sampled)

    kept_x: list[np.ndarray] = []
    kept_y: list[np.ndarray] = []
    for x in xs:
        result = generator.solve_for_x(x)
        if _status_ok(result["status"]) and result["y"] is not None:
            kept_x.append(np.asarray(x, dtype=np.float64))
            kept_y.append(np.asarray(result["y"], dtype=np.float64))
            if len(kept_x) >= target:
                break

    if len(kept_x) < target:
        raise RuntimeError(
            f"Only collected {len(kept_x)} successful samples out of requested {target}. "
            "Increase N_samples or adjust the solver settings in data.json."
        )

    return np.asarray(kept_x, dtype=np.float64), np.asarray(kept_y, dtype=np.float64), dict(generator.get_problem_data())


def _generate_nonconvex_dataset(data_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from scripts.factory.nonconvx_factory import build_problem_generator

    generator = build_problem_generator(data_cfg)
    target = int(data_cfg["num_samples"])
    x_l = np.asarray(data_cfg["x_L"], dtype=np.float64)
    x_u = np.asarray(data_cfg["x_U"], dtype=np.float64)
    rng = np.random.default_rng(int(data_cfg["seed"]) + 1)

    kept_x: list[np.ndarray] = []
    kept_y: list[np.ndarray] = []
    attempts = 0
    max_attempts = max(4 * target, target + 100)

    while len(kept_x) < target and attempts < max_attempts:
        batch_size = min(max(32, target - len(kept_x)), target)
        xs = rng.uniform(x_l, x_u, size=(batch_size, x_l.shape[0]))
        for x in xs:
            result = generator.solve_for_x(x)
            attempts += 1
            if str(result["status"]).strip().lower() == "optimal" and result["y"] is not None:
                kept_x.append(np.asarray(x, dtype=np.float64))
                kept_y.append(np.asarray(result["y"], dtype=np.float64))
                if len(kept_x) >= target:
                    break

    if len(kept_x) < target:
        raise RuntimeError(
            f"Only collected {len(kept_x)} successful nonconvex samples out of requested {target}."
        )

    return np.asarray(kept_x, dtype=np.float64), np.asarray(kept_y, dtype=np.float64), dict(generator.get_problem_data())


def _generate_dataset(problem_type: str, data_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if problem_type in {"qp", "qcqp"}:
        return _generate_poly_dataset(data_cfg)
    if problem_type == "nlp":
        return _generate_nlp_dataset(data_cfg)
    if problem_type == "nonconvex":
        return _generate_nonconvex_dataset(data_cfg)
    raise ValueError(f"Unsupported type '{problem_type}'.")


def _problem_npz_payload(problem_type: str, data_cfg: dict[str, Any], problem_data: dict[str, Any]) -> dict[str, np.ndarray]:
    x_l = np.asarray(data_cfg["x_L"], dtype=np.float64)
    x_u = np.asarray(data_cfg["x_U"], dtype=np.float64)
    payload = {key: np.asarray(value) for key, value in dict(problem_data).items()}
    payload["problem_type"] = np.asarray(problem_type)
    payload["x_L"] = x_l
    payload["x_U"] = x_u
    payload["M"] = np.stack([x_l, x_u], axis=0)
    return payload


def _save_outputs(problem_type: str, data_cfg: dict[str, Any], X: np.ndarray, Y: np.ndarray, problem_data: dict[str, Any]) -> Path:
    output_dir = ROOT / "notebooks" / "data" / problem_type
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(output_dir / "parameters.csv", X)
    _write_csv(output_dir / "variables.csv", Y)
    np.savez(output_dir / "problem.npz", **_problem_npz_payload(problem_type, data_cfg, problem_data))
    _write_json(output_dir / "data.json", data_cfg)
    return output_dir


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    problem_type = _normalize_problem_type(args.type)
    data_path = _resolve_path(args.data_json)
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data.json at {data_path}")

    raw_data_cfg = _load_json(data_path)
    data_cfg = _prepare_data_cfg(problem_type, raw_data_cfg, args.dimension)
    X, Y, problem_data = _generate_dataset(problem_type, data_cfg)
    output_dir = _save_outputs(problem_type, data_cfg, X, Y, problem_data)

    print("")
    print("=" * 80)
    print(f"Generated {problem_type} data")
    print(f"Source data.json: {data_path}")
    print(f"Output directory: {output_dir}")
    print(f"parameters.csv shape: {X.shape}")
    print(f"variables.csv shape: {Y.shape}")
    print(f"problem.npz keys: {sorted(_problem_npz_payload(problem_type, data_cfg, problem_data).keys())}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
