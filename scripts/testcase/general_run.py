#!/usr/bin/env python3
from __future__ import annotations

import copy
import csv
import hashlib
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "nlpopt" / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from case.general import (  # noqa: E402
    DATASET_KIND,
    build_problem_generator,
    build_problem_model,
    build_problem_model_from_data,
    default_case_dir,
    normalize_problem_type,
)
from case.general.factory import model_definition_path  # noqa: E402
from scripts.misc.cli_overrides import apply_cli_overrides  # noqa: E402
from scripts.misc.inequality_multipliers import default_ineq_multipliers  # noqa: E402
from scripts.misc.nlpopt_prediction_export import export_ordered_projected_predictions  # noqa: E402
from scripts.misc.optimizer_profile import enrich_optimizer_generation_metadata  # noqa: E402
from scripts.misc.runtime import resolve_dtype, runtime_summary, select_device  # noqa: E402
from scripts.misc.training_timing import should_track_epoch, summarize_timing_profile, timing_window_label  # noqa: E402
from scripts.misc.console_format import fmt_dec, fmt_pct, fmt_sci, fmt_sec  # noqa: E402
from scripts.plot_utils.plotting import (  # noqa: E402
    save_objective_value_violation_plot,
)
from scripts.testcase import run_helpers as unified  # noqa: E402
from opt.training import (  # noqa: E402
    TrainConfig,
    apply_projection_layers,
    build_epoch_fns,
    build_train_fns_from_jaxmodel,
    build_violation_fn_from_jaxmodel,
    cfg_from_dict,
    make_batched_objective,
    make_fixed_batches,
    make_subproblem_layer_from_model,
    warmup_compile,
)

jax.config.update("jax_enable_x64", True)

SCHEMA_VERSION = 1
_LOCAL_REQUIRED_KEYS = ("type", "num_samples", "seed", "x_L", "x_U")
_METRIC_KEYS = ("loss", "obj", "mse_y", "mse_lam", "mse_mu")


@dataclass(frozen=True)
class DatasetBundle:
    dataset_dir: Path
    dataset_id: str
    generated: bool
    X: np.ndarray
    Y: np.ndarray
    Mu: np.ndarray
    metadata: dict[str, Any]


def _metrics_tuple_to_dict(values):
    return {k: v for k, v in zip(_METRIC_KEYS, values)}


def _consistency(metrics: Mapping[str, Any]) -> float:
    return float(metrics["mse_y"] + metrics["mse_lam"] + metrics["mse_mu"])


def _max_violation(violations) -> float:
    return float(jnp.max(violations))


def _block_until_ready(tree):
    return jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        tree,
    )


def _measure_time(fn, *args, repeats: int = 5) -> float:
    durations = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        _block_until_ready(out)
        durations.append(time.perf_counter() - t0)
    return float(sum(durations) / len(durations))


def _print_multiplier_activity_summary(summary: dict | None) -> None:
    if not summary or int(summary.get("total_entries", 0)) <= 0:
        return
    print("=== Multiplier activity summary (optimizer active if mu > 1e-6) ===")
    print(f"Optimizer active fraction: {fmt_dec(summary['optimizer_active_fraction'])}")
    print(
        "Predicted mu mean on optimizer-active/inactive: "
        f"{fmt_dec(summary['predicted_mean_on_optimizer_active'])} / "
        f"{fmt_dec(summary['predicted_mean_on_optimizer_inactive'])}"
    )
    print(f"Activity agreement @1e-6: {fmt_dec(summary['activity_agreement_rate_at_tol'])}")
    print("")


def _artifact_size_mb(*paths: Path) -> float:
    total_bytes = 0
    for path in paths:
        if path.exists() and path.is_file():
            total_bytes += int(path.stat().st_size)
    return float(total_bytes) / (1024.0 * 1024.0)


def _resolve_workspace(path_arg: str | None = None) -> Path:
    if path_arg is None:
        case_dir = default_case_dir()
    else:
        candidate = Path(path_arg).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        case_dir = candidate
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def _local_paths(case_dir: Path) -> tuple[Path, Path, Path, Path]:
    return (
        case_dir / "data.json",
        case_dir / "config.json",
        case_dir / "proj.json",
        model_definition_path(case_dir),
    )


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _load_local_configs(case_dir: Path) -> tuple[dict, dict, dict, Path]:
    data_path, cfg_path, proj_path, model_def_path = _local_paths(case_dir)
    missing = [path.name for path in (data_path, cfg_path, proj_path, model_def_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Expected local files in {case_dir}: " + ", ".join(missing))
    data_cfg = _load_json(data_path)
    cfg_dict = _load_json(cfg_path)
    proj_cfg = _load_json(proj_path)
    data_cfg, cfg_dict = apply_cli_overrides(data_cfg, cfg_dict)
    return (
        data_cfg,
        cfg_dict,
        proj_cfg,
        model_def_path,
    )


def _json_hash(payload: dict) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:12]


def _file_hash(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:12]


def _config_hash(cfg_dict: dict, proj_cfg: dict) -> str:
    payload = json.dumps({"config": cfg_dict, "proj": proj_cfg}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def _normalize_local_data_cfg(data_cfg: Mapping[str, Any], model_def_path: Path) -> dict[str, Any]:
    if "num_samples" in data_cfg:
        num_samples = int(data_cfg["num_samples"])
        seed = int(data_cfg["seed"])
        x_l = np.asarray(data_cfg["x_L"], dtype=np.float64)
        x_u = np.asarray(data_cfg["x_U"], dtype=np.float64)
    else:
        translated_missing = [key for key in ("N_samples", "seed", "x_L", "x_U") if key not in data_cfg]
        if translated_missing:
            raise ValueError(
                "case/general data config is missing required keys: " + ", ".join(sorted(translated_missing))
            )
        num_samples = int(data_cfg["N_samples"])
        seed = int(data_cfg["seed"])
        x_l = np.asarray(data_cfg["x_L"], dtype=np.float64)
        x_u = np.asarray(data_cfg["x_U"], dtype=np.float64)
    normalize_problem_type(str(data_cfg["type"]))
    if x_l.shape != x_u.shape:
        raise ValueError("x_L and x_U must have the same shape.")
    return {
        "type": DATASET_KIND,
        "num_samples": num_samples,
        "seed": seed,
        "x_L": x_l.astype(float).tolist(),
        "x_U": x_u.astype(float).tolist(),
        "schema_version": int(data_cfg.get("schema_version", SCHEMA_VERSION)),
        "model_definition_hash": _file_hash(model_def_path),
    }


def _canonical_data_cfg(data_cfg: Mapping[str, Any], model_def_path: Path) -> dict[str, Any]:
    local = _normalize_local_data_cfg(data_cfg, model_def_path)
    return copy.deepcopy(local)


def _translated_general_cfg(data_cfg: Mapping[str, Any], model_def_path: Path) -> dict[str, Any]:
    local = _normalize_local_data_cfg(data_cfg, model_def_path)
    model = build_problem_model(dtype=jnp.float64, case_dir=model_def_path.parent)
    y0 = jnp.zeros((model.var_spec.total_size,), dtype=model.dtype)
    n_x = int(np.asarray(local["x_L"], dtype=np.float64).shape[0])
    x0 = jnp.zeros((n_x,), dtype=model.dtype)
    params = {"x": x0}
    if np.asarray(local["x_U"], dtype=np.float64).shape[0] != n_x:
        raise ValueError("x_L and x_U must have the same shape.")
    return {
        "type": DATASET_KIND,
        "n_x": n_x,
        "n_y": int(model.var_spec.total_size),
        "n_eq": int(model.eq_residual(params, y0).shape[0]),
        "n_ineq": int(model.ineq_residual(params, y0).shape[0]),
        "N_samples": int(local["num_samples"]),
        "N_points": int(local["num_samples"]),
        "seed": int(local["seed"]),
        "x_L": copy.deepcopy(local["x_L"]),
        "x_U": copy.deepcopy(local["x_U"]),
        "model_definition_hash": str(local["model_definition_hash"]),
    }


def build_dataset_id(data_cfg: Mapping[str, Any], model_def_path: Path) -> str:
    canonical = _canonical_data_cfg(data_cfg, model_def_path)
    translated = _translated_general_cfg(data_cfg, model_def_path)
    stem = (
        f"general_nx{translated['n_x']}_ny{translated['n_y']}_"
        f"neq{translated['n_eq']}_nineq{translated['n_ineq']}_"
        f"ns{translated['N_points']}_seed{translated['seed']}"
    )
    return f"{stem}_{_json_hash(canonical)}"


def dataset_dir(case_dir: Path, data_cfg: Mapping[str, Any], model_def_path: Path) -> Path:
    return case_dir / "problem_data" / DATASET_KIND / build_dataset_id(data_cfg, model_def_path)


def _paths(base: Path) -> dict[str, Path]:
    return {
        "arrays": base / "dataset.npz",
        "parameters_csv": base / "parameters.csv",
        "variables_csv": base / "variables.csv",
        "ineq_multipliers_csv": base / "ineq_multipliers.csv",
        "metadata": base / "metadata.json",
        "data": base / "data.json",
        "model_definition": base / "model_definition.py",
        "problem_data": base / "problem_data.json",
    }


def _write_csv(path: Path, arr: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(np.asarray(arr).tolist())


def _dataset_exists(base: Path) -> bool:
    return all(path.exists() for path in _paths(base).values())


def _save_dataset(
    base: Path,
    *,
    data_cfg: Mapping[str, Any],
    model_def_path: Path,
    X: np.ndarray,
    Y: np.ndarray,
    Mu: np.ndarray,
    metadata: Mapping[str, Any],
    problem_data: Mapping[str, Any],
) -> dict[str, Any]:
    base.mkdir(parents=True, exist_ok=True)
    paths = _paths(base)
    np.savez(paths["arrays"], X=np.asarray(X, dtype=np.float64), Y=np.asarray(Y, dtype=np.float64), Mu=np.asarray(Mu, dtype=np.float64))
    _write_csv(paths["parameters_csv"], X)
    _write_csv(paths["variables_csv"], Y)
    _write_csv(paths["ineq_multipliers_csv"], Mu)
    _write_json(paths["data"], _canonical_data_cfg(data_cfg, model_def_path))
    paths["problem_data"].write_text(json.dumps(copy.deepcopy(dict(problem_data)), indent=2, sort_keys=True), encoding="utf-8")
    shutil.copyfile(model_def_path, paths["model_definition"])
    enriched = enrich_optimizer_generation_metadata(
        metadata,
        num_points=int(np.asarray(X).shape[0]),
        artifact_paths=(
            paths["arrays"],
            paths["parameters_csv"],
            paths["variables_csv"],
            paths["ineq_multipliers_csv"],
            paths["data"],
            paths["problem_data"],
            paths["model_definition"],
        ),
    )
    _write_json(paths["metadata"], enriched)
    return enriched


def _load_dataset(base: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    arrays = np.load(_paths(base)["arrays"])
    metadata = _load_json(_paths(base)["metadata"])
    return np.asarray(arrays["X"]), np.asarray(arrays["Y"]), np.asarray(arrays["Mu"]), metadata


def ensure_cached_dataset(
    case_dir: Path,
    data_cfg: Mapping[str, Any],
    model_def_path: Path,
    generate_fn,
    *,
    force: bool = False,
) -> DatasetBundle:
    base = dataset_dir(case_dir, data_cfg, model_def_path)
    dataset_id = build_dataset_id(data_cfg, model_def_path)
    if _dataset_exists(base) and not force:
        X, Y, Mu, metadata = _load_dataset(base)
        return DatasetBundle(base, dataset_id, False, X, Y, Mu, metadata)

    X, Y, Mu, metadata, problem_data = generate_fn()
    metadata = _save_dataset(
        base,
        data_cfg=data_cfg,
        model_def_path=model_def_path,
        X=X,
        Y=Y,
        Mu=Mu,
        metadata=metadata,
        problem_data=problem_data,
    )
    return DatasetBundle(base, dataset_id, True, X, Y, Mu, dict(metadata))


def _generate_dataset(generator, data_cfg: Mapping[str, Any]):
    start_time = time.perf_counter()
    n_samples = int(data_cfg["num_samples"] if "num_samples" in data_cfg else data_cfg["N_samples"])
    xs = generator.sample_parameters(n_samples)
    X = np.asarray(xs, dtype=np.float64)
    Y = np.zeros((X.shape[0], 0), dtype=np.float64)
    Mu_row = default_ineq_multipliers(generator.n_ineq)
    Mu = np.repeat(Mu_row[None, :], X.shape[0], axis=0) if X.shape[0] > 0 else np.zeros((0, generator.n_ineq), dtype=np.float64)
    metadata = {
        "problem_type": DATASET_KIND,
        "n_x": int(generator.n_x),
        "n_y": int(generator.n_y),
        "n_eq": int(generator.n_eq),
        "n_ineq": int(generator.n_ineq),
        "N_samples": n_samples,
        "N_points": n_samples,
        "reference_labels": False,
        "optimizer_generation_mode": "parameter_sampling_only",
        "seed": int(data_cfg["seed"]),
        "optimizer_generation_wall_time_sec": time.perf_counter() - start_time,
    }
    return X, Y, Mu, metadata, generator.get_problem_data()


def _problem_shape_text(data_cfg: Mapping[str, Any], model_def_path: Path) -> str:
    translated = _translated_general_cfg(data_cfg, model_def_path)
    return (
        f"GENERAL  n_x={translated['n_x']} n_y={translated['n_y']} "
        f"n_eq={translated['n_eq']} n_ineq={translated['n_ineq']}"
    )


def _validate_translated_data_cfg(data_cfg: Mapping[str, Any]) -> None:
    if str(data_cfg["type"]).strip().lower() != DATASET_KIND:
        raise ValueError("Unsupported general testcase type.")
    for key in ("n_x", "n_y", "n_eq", "n_ineq", "N_samples", "N_points", "seed", "x_L", "x_U"):
        if key not in data_cfg:
            raise ValueError(f"Missing required translated general field '{key}'.")


def _write_local_run_configs(run_dir: Path, data_cfg: dict, cfg_dict: dict, proj_cfg: dict, model_def_path: Path) -> None:
    unified._write_run_configs(run_dir, data_cfg, cfg_dict, proj_cfg)
    shutil.copyfile(model_def_path, run_dir / "model_definition.py")


def _run_nlpopt(
    case_dir: Path,
    data_cfg: dict,
    cfg_dict: dict,
    proj_cfg: dict,
    model_def_path: Path,
    *,
    output_dir: Path,
) -> unified.RunArtifacts:
    translated_cfg = _translated_general_cfg(data_cfg, model_def_path)
    merged_cfg = {key: value for key, value in unified._training_cfg_only(cfg_dict).items() if key != "print_every"}
    merged_cfg["IS_FIXED"] = str(proj_cfg.get("cp_mode", "fixed")).lower() == "fixed"
    merged_cfg["safety"] = float(proj_cfg.get("safety", 0.95))
    merged_cfg["knorm_iters"] = int(proj_cfg.get("knorm_iters", 20))
    merged_cfg["knorm_seed"] = int(proj_cfg.get("knorm_seed", 42))
    merged_cfg["adjoint_iters"] = int(proj_cfg.get("adjoint_iters", 30))
    merged_cfg["use_ruiz"] = bool(proj_cfg.get("use_ruiz", True))
    merged_cfg["ruiz_iters"] = int(proj_cfg.get("ruiz_iters", 4))
    merged_cfg["k_layer"] = int(proj_cfg.get("k_layer", 1))
    cfg: TrainConfig = cfg_from_dict(merged_cfg)
    print_every = max(1, int(cfg_dict.get("print_every", 10)))

    n_x = int(translated_cfg["n_x"])
    n_y = int(translated_cfg["n_y"])
    n_eq = int(translated_cfg["n_eq"])
    n_ineq = int(translated_cfg["n_ineq"])
    train_dtype = resolve_dtype(cfg.dtype)
    device = select_device(cfg.device)

    generator = build_problem_generator(data_cfg, case_dir=case_dir)
    problem_data = generator.get_problem_data()
    train_model_def = build_problem_model_from_data(problem_data, dtype=train_dtype, case_dir=case_dir)

    dataset = ensure_cached_dataset(
        case_dir,
        data_cfg,
        model_def_path,
        lambda: _generate_dataset(generator, data_cfg),
        force=bool(data_cfg.get("force_regenerate", False)),
    )

    print(f"Runtime: {runtime_summary(device, train_dtype)}")
    print(f"Dataset: {dataset.dataset_dir}")
    print(f"Dataset status: {'generated' if dataset.generated else 'reused'}")

    X_np = np.asarray(dataset.X, dtype=np.float64)
    rng = np.random.default_rng(int(cfg.seed))
    idx = np.arange(X_np.shape[0])
    rng.shuffle(idx)
    X_np = X_np[idx]

    total_samples = X_np.shape[0]
    n_train = int(cfg.train_frac * total_samples)
    bs = int(cfg.batch_size)
    n_train2 = (n_train // bs) * bs
    n_val2 = ((total_samples - n_train) // bs) * bs
    if n_train2 < bs or n_val2 < bs:
        raise ValueError("Not enough samples for at least one train and one validation batch.")

    X_train = jax.device_put(jnp.asarray(X_np[:n_train2], dtype=train_dtype), device)
    X_val = jax.device_put(jnp.asarray(X_np[n_train2:n_train2 + n_val2], dtype=train_dtype), device)
    train_batches = jax.device_put(make_fixed_batches(X_train, bs), device)
    val_batches = jax.device_put(make_fixed_batches(X_val, bs), device)

    print(f"JAXMODEL GENERAL  n_x={n_x} n_y={n_y} n_eq={n_eq} n_ineq={n_ineq}")
    print(f"batch_size={bs}  train_batches={train_batches.shape[0]}  val_batches={val_batches.shape[0]}")

    model, init_state, train_step, eval_step = build_train_fns_from_jaxmodel(
        model_def=train_model_def,
        cfg=cfg,
        p=n_x,
        param_name="x",
    )
    state = jax.device_put(init_state(jax.random.PRNGKey(cfg.seed)), device)
    sub_layer = make_subproblem_layer_from_model(train_model_def, param_name="x")
    batched_objective = make_batched_objective(train_model_def, param_name="x")

    @jax.jit
    def predict_y_tilde(params, x_batch):
        y_hat, lam_hat, mu_hat = model.apply({"params": params}, x_batch)
        y_tilde, _, _ = apply_projection_layers(
            sub_layer=sub_layer,
            x_batch=x_batch,
            y0=y_hat,
            lam0=lam_hat,
            mu0=mu_hat,
            cfg=cfg,
        )
        return y_tilde

    if cfg.jit_warmup:
        state = warmup_compile(
            cfg=cfg,
            state=state,
            train_step_fn=train_step,
            eval_step_fn=eval_step,
            p=n_x,
            dtype=train_dtype,
            device=device,
        )
        print("JIT warmup: completed for train/eval step.")

    train_epoch, eval_epoch = build_epoch_fns(train_step, eval_step)
    viol_fn = build_violation_fn_from_jaxmodel(train_model_def, cfg=cfg, p=n_x, param_name="x")
    n_train_batches = int(train_batches.shape[0])
    n_val_batches = int(val_batches.shape[0])
    train_epoch_time_total = 0.0
    val_epoch_time_total = 0.0
    train_epoch_time_tracked = 0.0
    val_epoch_time_tracked = 0.0
    history_epochs: list[int] = []
    history_train_objective: list[float] = []
    history_val_objective: list[float] = []
    history_train_violation: list[float] = []
    history_val_violation: list[float] = []

    @jax.jit
    def reduce_violation_stats(params, batches):
        def one_batch(xb):
            v = viol_fn(params, xb)
            return jnp.asarray(
                [
                    v["eq_inf"],
                    v.get("eq_mean", v["eq_inf"]),
                    v["ineq_inf"],
                    v.get("ineq_mean", v["ineq_inf"]),
                    v["bound_inf"],
                    v.get("bound_mean", v["bound_inf"]),
                ],
                dtype=train_dtype,
            )

        vals = jax.vmap(one_batch)(batches)
        max_vals = jnp.max(vals[:, [0, 2, 4]], axis=0)
        mean_vals = jnp.mean(vals[:, [1, 3, 5]], axis=0)
        return {
            "eq_inf": max_vals[0],
            "eq_mean": mean_vals[0],
            "ineq_inf": max_vals[1],
            "ineq_mean": mean_vals[1],
            "bound_inf": max_vals[2],
            "bound_mean": mean_vals[2],
        }

    train_wall_t0 = time.perf_counter()
    for ep in range(cfg.epochs):
        train_epoch_t0 = time.perf_counter()
        state, tr_vals = train_epoch(state, train_batches)
        _block_until_ready((state.params, tr_vals))
        train_epoch_time = time.perf_counter() - train_epoch_t0

        val_epoch_t0 = time.perf_counter()
        va_vals = eval_epoch(state.params, val_batches)
        _block_until_ready(va_vals)
        val_epoch_time = time.perf_counter() - val_epoch_t0

        train_epoch_time_total += train_epoch_time
        val_epoch_time_total += val_epoch_time
        if should_track_epoch(ep, cfg.epochs):
            train_epoch_time_tracked += train_epoch_time
            val_epoch_time_tracked += val_epoch_time
        tr_m = _metrics_tuple_to_dict(tr_vals)
        va_m = _metrics_tuple_to_dict(va_vals)
        tr_viol = reduce_violation_stats(state.params, train_batches)
        va_viol = reduce_violation_stats(state.params, val_batches)
        train_viol_max = _max_violation(jnp.asarray([tr_viol["eq_inf"], tr_viol["ineq_inf"], tr_viol["bound_inf"]], dtype=train_dtype))
        val_viol_max = _max_violation(jnp.asarray([va_viol["eq_inf"], va_viol["ineq_inf"], va_viol["bound_inf"]], dtype=train_dtype))
        history_epochs.append(ep)
        history_train_objective.append(float(tr_m["obj"]))
        history_val_objective.append(float(va_m["obj"]))
        history_train_violation.append(train_viol_max)
        history_val_violation.append(val_viol_max)
        if (ep % print_every) == 0 or ep == cfg.epochs - 1:
            print(
                f"ep {ep:05d} | "
                f"train loss {fmt_sci(tr_m['loss'])} obj {fmt_sci(tr_m['obj'])} "
                f"cons {fmt_sci(_consistency(tr_m))} viol {fmt_sci(train_viol_max)} || "
                f"val loss {fmt_sci(va_m['loss'])} obj {fmt_sci(va_m['obj'])} "
                f"cons {fmt_sci(_consistency(va_m))} viol {fmt_sci(val_viol_max)}"
            )
    training_wall_time = time.perf_counter() - train_wall_t0

    tr_viol = reduce_violation_stats(state.params, train_batches)
    va_viol = reduce_violation_stats(state.params, val_batches)
    total_batches = train_batches.shape[0] + val_batches.shape[0]
    eq_max = float(jnp.maximum(tr_viol["eq_inf"], va_viol["eq_inf"]))
    eq_mean = float((tr_viol["eq_mean"] * train_batches.shape[0] + va_viol["eq_mean"] * val_batches.shape[0]) / total_batches)
    ineq_max = float(jnp.maximum(tr_viol["ineq_inf"], va_viol["ineq_inf"]))
    ineq_mean = float((tr_viol["ineq_mean"] * train_batches.shape[0] + va_viol["ineq_mean"] * val_batches.shape[0]) / total_batches)
    bnd_max = float(jnp.maximum(tr_viol["bound_inf"], va_viol["bound_inf"]))
    bnd_mean = float((tr_viol["bound_mean"] * train_batches.shape[0] + va_viol["bound_mean"] * val_batches.shape[0]) / total_batches)

    objective_value = float((float(tr_m["obj"]) * train_batches.shape[0] + float(va_m["obj"]) * val_batches.shape[0]) / total_batches)
    consistency_value = float((_consistency(tr_m) * train_batches.shape[0] + _consistency(va_m) * val_batches.shape[0]) / total_batches)

    print("\n=== Constraint violation (max over train+val) ===")
    print(f"Equality   ||A y - (b+Bx)||_inf : {fmt_sci(eq_max)}")
    print(f"Inequality max(·,0)_inf         : {fmt_sci(ineq_max)}")
    print(f"Bounds     max(lb,ub)_inf       : {fmt_sci(bnd_max)}\n")
    print("=== Training evaluation ===")
    print(f"Projected objective: {fmt_sci(objective_value)}")
    print(f"Consistency term  : {fmt_sci(consistency_value)}\n")

    @jax.jit
    def backbone_forward_fn(params, x_batch):
        return model.apply({"params": params}, x_batch)

    @jax.jit
    def projection_only_fn(x_batch, y_hat, lam_hat, mu_hat):
        return apply_projection_layers(
            sub_layer=sub_layer,
            x_batch=x_batch,
            y0=y_hat,
            lam0=lam_hat,
            mu0=mu_hat,
            cfg=cfg,
        )

    @jax.jit
    def loss_scalar_fn(params, x_batch):
        y_hat, lam_hat, mu_hat = model.apply({"params": params}, x_batch)
        y_tilde, lam_tilde, mu_tilde = apply_projection_layers(
            sub_layer=sub_layer,
            x_batch=x_batch,
            y0=y_hat,
            lam0=lam_hat,
            mu0=mu_hat,
            cfg=cfg,
        )
        obj = jnp.mean(batched_objective(x_batch, y_tilde))
        mse_y = jnp.mean((y_hat - y_tilde) ** 2)
        mse_lam = jnp.mean((lam_hat - lam_tilde) ** 2) if n_eq > 0 else jnp.asarray(0.0, dtype=train_dtype)
        mse_mu = jnp.mean((mu_hat - mu_tilde) ** 2) if n_ineq > 0 else jnp.asarray(0.0, dtype=train_dtype)
        return obj + jnp.asarray(cfg.alpha_consistency, dtype=train_dtype) * (mse_y + mse_lam + mse_mu)

    grad_only_fn = jax.jit(jax.grad(loss_scalar_fn))

    @jax.jit
    def optimizer_update_fn(state_in, grads):
        return state_in.apply_gradients(grads=grads)

    sample_train_batch = train_batches[0]
    sample_val_batch = val_batches[0]
    sample_train_pred = backbone_forward_fn(state.params, sample_train_batch)
    sample_val_pred = backbone_forward_fn(state.params, sample_val_batch)
    sample_grads = grad_only_fn(state.params, sample_train_batch)
    _block_until_ready(sample_train_pred)
    _block_until_ready(sample_val_pred)
    _block_until_ready(sample_grads)
    _block_until_ready(projection_only_fn(sample_train_batch, *sample_train_pred))
    _block_until_ready(projection_only_fn(sample_val_batch, *sample_val_pred))
    _block_until_ready(loss_scalar_fn(state.params, sample_train_batch))
    _block_until_ready(optimizer_update_fn(state, sample_grads))

    bb_train_t = _measure_time(backbone_forward_fn, state.params, sample_train_batch)
    bb_val_t = _measure_time(backbone_forward_fn, state.params, sample_val_batch)
    proj_train_t = _measure_time(projection_only_fn, sample_train_batch, *sample_train_pred)
    proj_val_t = _measure_time(projection_only_fn, sample_val_batch, *sample_val_pred)
    forward_total_t = _measure_time(loss_scalar_fn, state.params, sample_train_batch)
    grad_total_t = _measure_time(grad_only_fn, state.params, sample_train_batch)
    opt_update_t = _measure_time(optimizer_update_fn, state, sample_grads)
    backward_t = max(0.0, grad_total_t - forward_total_t)

    timing_profile = {
        "training_wall_time_sec": training_wall_time,
        "train_epoch_time_total_sec": train_epoch_time_total,
        "val_epoch_time_total_sec": val_epoch_time_total,
        "train_epoch_time_tracked_sec": train_epoch_time_tracked,
        "val_epoch_time_tracked_sec": val_epoch_time_tracked,
        "epochs": int(cfg.epochs),
        "train_batches_per_epoch": n_train_batches,
        "val_batches_per_epoch": n_val_batches,
    }
    timing_summary = summarize_timing_profile(timing_profile)

    train_steps = int(timing_summary["timing_epochs_recorded"]) * int(train_batches.shape[0])
    val_steps = int(timing_summary["timing_epochs_recorded"]) * int(val_batches.shape[0])
    backbone_total = train_steps * bb_train_t + val_steps * bb_val_t
    projection_total = train_steps * proj_train_t + val_steps * proj_val_t
    backward_total = train_steps * backward_t
    optimizer_total = train_steps * opt_update_t
    timing_profile.update(
        {
            "backbone_total_sec": backbone_total,
            "projection_total_sec": projection_total,
            "backward_total_sec": backward_total,
            "optimizer_total_sec": optimizer_total,
        }
    )
    timing_summary = summarize_timing_profile(timing_profile)

    print("=== Profiled training time distribution ===")
    print(f"Training wall time: {fmt_sec(training_wall_time)}")
    print(
        f"Average epoch time ({timing_window_label(cfg.epochs)}): "
        f"train {fmt_sec(timing_summary['avg_train_epoch_time_sec'])} "
        f"val {fmt_sec(timing_summary['avg_val_epoch_time_sec'])} "
        f"total {fmt_sec(timing_summary['avg_total_epoch_time_sec'])}"
    )
    print(
        f"Average batch time ({timing_window_label(cfg.epochs)}): "
        f"train {fmt_sec(timing_summary['avg_train_batch_time_sec'])} "
        f"val {fmt_sec(timing_summary['avg_val_batch_time_sec'])} "
        f"overall {fmt_sec(timing_summary['avg_total_batch_time_sec'])}"
    )
    if float(timing_summary["backbone_total_sec"]) + float(timing_summary["projection_total_sec"]) + float(timing_summary["backward_total_sec"]) + float(timing_summary["optimizer_total_sec"]) > 0.0:
        print(f"Backbone forward : {fmt_pct(timing_summary['time_backbone_percent'])} (train + val)")
        print(f"Projection       : {fmt_pct(timing_summary['time_projection_percent'])} (train + val)")
        print(f"Backward         : {fmt_pct(timing_summary['time_backward_percent'])} (train only)")
        print(f"Optimizer update : {fmt_pct(timing_summary['time_optimizer_percent'])} (train only)")
    print("")

    dataset_root = dataset_dir(case_dir, data_cfg, model_def_path)
    _write_local_run_configs(output_dir, _normalize_local_data_cfg(data_cfg, model_def_path), cfg_dict, proj_cfg, model_def_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_hash = _config_hash(cfg_dict, proj_cfg)
    params_path = output_dir / "trained_params.npz"
    metrics_path = output_dir / "summary.json"
    history_path = output_dir / "run_history.json"
    plot_path = output_dir / "training_metrics.png"

    np.savez(params_path, params=np.array(jax.device_get(state.params), dtype=object), allow_pickle=True)
    predicted_var_path, predicted_mu_path, multiplier_activity = export_ordered_projected_predictions(
        model=model,
        params=state.params,
        sub_layer=sub_layer,
        cfg=cfg,
        X=dataset.X,
        dtype=train_dtype,
        device=device,
        batch_size=bs,
        output_dir=output_dir,
        optimizer_multipliers=dataset.Mu,
    )
    _print_multiplier_activity_summary(multiplier_activity)
    _write_json(
        history_path,
        {
            "run_hash": run_hash,
            "config_seed": int(cfg_dict["seed"]),
            "data_seed": int(data_cfg["seed"]),
            "epochs": history_epochs,
            "train_objective": history_train_objective,
            "val_objective": history_val_objective,
            "train_violation": history_train_violation,
            "val_violation": history_val_violation,
            "timing_start_epoch": int(timing_summary["timing_start_epoch"]),
            "timing_epochs_recorded": int(timing_summary["timing_epochs_recorded"]),
        },
    )
    save_objective_value_violation_plot(
        plot_path,
        epochs=history_epochs,
        train_objective=history_train_objective,
        val_objective=history_val_objective,
        train_violation=history_train_violation,
        val_violation=history_val_violation,
        title="General Training",
    )
    metrics_payload = {
        "framework": "nlpopt",
        "run_hash": run_hash,
        "dataset_dir": str(dataset.dataset_dir),
        "save_dir": str(output_dir),
        "predicted_variables_path": str(predicted_var_path),
        "predicted_multipliers_path": str(predicted_mu_path),
        "multiplier_activity": multiplier_activity,
        "objective_value": objective_value,
        "max_equality": eq_max,
        "mean_equality": eq_mean,
        "max_inequality": ineq_max,
        "mean_inequality": ineq_mean,
        "max_bound": bnd_max,
        "mean_bound": bnd_mean,
        "consistency": consistency_value,
        "optimality_gap": None,
        "eq_inf": eq_max,
        "ineq_inf": ineq_max,
        "bound_inf": bnd_max,
        "mse_y_tilde_vs_label": None,
        "relative_objective_gap": None,
        "training_wall_time_sec": training_wall_time,
        "device": device.platform,
        "dtype": jnp.dtype(train_dtype).name,
    }
    metrics_payload.update(timing_summary)
    metrics_payload["space_mb"] = _artifact_size_mb(
        params_path,
        history_path,
        plot_path,
        predicted_var_path,
        predicted_mu_path,
    )
    _write_json(metrics_path, metrics_payload)
    print(f"Saved params: {params_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved history: {history_path}")
    print(f"Saved predicted variables: {predicted_var_path}")
    print(f"Saved predicted multipliers: {predicted_mu_path}")
    return unified.RunArtifacts(
        framework="nlpopt",
        dataset_dir=dataset_root,
        run_dir=output_dir,
        history_path=history_path,
        metrics_path=metrics_path,
        plot_path=plot_path,
    )


def _print_run_header(case_dir: Path, data_cfg: dict, cfg_dict: dict, model_def_path: Path) -> None:
    dataset_target = dataset_dir(case_dir, data_cfg, model_def_path)
    print("=" * 80)
    print("Standalone runner | framework=NLPOpt")
    print(_problem_shape_text(data_cfg, model_def_path))
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
    model_def_path: Path,
    *,
    output_dir_override: Optional[Path] = None,
) -> unified.RunArtifacts:
    framework = unified._normalize_model_name(str(cfg_dict.get("model", "nlpopt")))
    if framework != "nlpopt":
        raise ValueError("case/general currently supports model='nlpopt' only.")
    _print_run_header(case_dir, data_cfg, cfg_dict, model_def_path)
    dataset_root = dataset_dir(case_dir, data_cfg, model_def_path)
    output_dir = Path(output_dir_override) if output_dir_override is not None else unified._framework_dir(dataset_root, framework)
    return _run_nlpopt(case_dir, data_cfg, cfg_dict, proj_cfg, model_def_path, output_dir=output_dir)


def default_dataset_dir() -> Path:
    case_dir = _resolve_workspace()
    data_cfg, _cfg, _proj, model_def_path = _load_local_configs(case_dir)
    return dataset_dir(case_dir, data_cfg, model_def_path)


def run_case(
    _case_dir: Path | None = None,
    _path_arg: str | None = None,
    *,
    data_cfg_override: dict | None = None,
    cfg_dict_override: dict | None = None,
    proj_cfg_override: dict | None = None,
    output_dir_override: Path | None = None,
) -> int:
    return main(
        case_dir=_case_dir,
        path_arg=_path_arg,
        data_cfg_override=data_cfg_override,
        cfg_dict_override=cfg_dict_override,
        proj_cfg_override=proj_cfg_override,
        output_dir_override=output_dir_override,
    )


def main(
    *,
    case_dir: Path | None = None,
    path_arg: str | None = None,
    data_cfg_override: dict | None = None,
    cfg_dict_override: dict | None = None,
    proj_cfg_override: dict | None = None,
    output_dir_override: Path | None = None,
) -> int:
    workspace = _resolve_workspace(path_arg if path_arg is not None else (str(case_dir) if case_dir is not None else None))
    data_cfg, cfg_dict, proj_cfg, model_def_path = _load_local_configs(workspace)
    if data_cfg_override is not None:
        data_cfg = copy.deepcopy(data_cfg_override)
    if cfg_dict_override is not None:
        cfg_dict = copy.deepcopy(cfg_dict_override)
    if proj_cfg_override is not None:
        proj_cfg = copy.deepcopy(proj_cfg_override)
    cfg_dict["model"] = "nlpopt"
    data_cfg = _normalize_local_data_cfg(data_cfg, model_def_path)

    artifacts = _run_single_case(
        workspace,
        data_cfg,
        cfg_dict,
        proj_cfg,
        model_def_path,
        output_dir_override=output_dir_override,
    )
    metadata_path = unified._append_family_metadata(
        artifacts.dataset_dir,
        mode="single_model",
        output_dir=artifacts.run_dir,
        data_cfg=data_cfg,
        cfg_dict=cfg_dict,
        proj_cfg=proj_cfg,
        framework=artifacts.framework,
        seeds=[int(cfg_dict.get("seed", 42))],
        extra={
            "summary_path": str(artifacts.metrics_path),
            "history_path": str(artifacts.history_path),
        },
    )
    print(f"[run] Updated family metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
