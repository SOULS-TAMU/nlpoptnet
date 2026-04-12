#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "nlpopt" / "src"

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.factory.poly_factory import (  # noqa: E402
    build_problem_generator,
    build_problem_model,
    build_problem_model_from_data,
    normalize_problem_type,
    uses_nonconvex_generator,
)
from scripts.misc.poly_dataset_cache import ensure_cached_dataset  # noqa: E402
from scripts.misc.inequality_multipliers import coerce_ineq_multipliers  # noqa: E402
from scripts.misc.nlpopt_prediction_export import export_ordered_projected_predictions  # noqa: E402
from scripts.misc.json_io import load_json as _load_json_from_path, write_json_atomic  # noqa: E402
from scripts.misc.runtime import resolve_dtype, runtime_summary, select_device  # noqa: E402
from scripts.misc.optimizer_profile import history_optimizer_timing_fields  # noqa: E402
from scripts.misc.solver_config import resolve_solver_name  # noqa: E402
from scripts.misc.training_timing import should_track_epoch, summarize_timing_profile, timing_window_label  # noqa: E402
from scripts.misc.console_format import fmt_dec, fmt_pct, fmt_sci, fmt_sec  # noqa: E402

from opt.training import (  # noqa: E402
    TrainConfig,
    apply_projection_layers,
    build_epoch_fns,
    build_train_fns_from_jaxmodel,
    build_violation_fn_from_jaxmodel,
    cfg_from_dict,
    make_fixed_batches,
    make_subproblem_layer_from_model,
    warmup_compile,
)
from scripts.plot_utils.plotting import save_objective_violation_plot  # noqa: E402
from solgen import SolGenModel  # noqa: E402

jax.config.update("jax_enable_x64", True)
_METRIC_KEYS = ("loss", "obj", "mse_y", "mse_lam", "mse_mu")


def _metrics_tuple_to_dict(values):
    return {k: v for k, v in zip(_METRIC_KEYS, values)}


def _load_json(path: Path):
    return _load_json_from_path(path)


def _write_json(path: Path, payload: dict) -> None:
    write_json_atomic(path, payload)


def _artifact_size_mb(*paths: Path) -> float:
    total_bytes = 0
    for path in paths:
        if path.exists() and path.is_file():
            total_bytes += int(path.stat().st_size)
    return float(total_bytes) / (1024.0 * 1024.0)


def _validate_data_cfg(data_cfg: dict) -> None:
    problem_type = normalize_problem_type(str(data_cfg["type"]))
    p = int(data_cfg["p"])
    n = int(data_cfg["n"])
    me = int(data_cfg["me"])
    mi = int(data_cfg["mi"])
    x_l = np.asarray(data_cfg["x_L"], dtype=float)
    x_u = np.asarray(data_cfg["x_U"], dtype=float)

    if p <= 0 or n <= 0:
        raise ValueError("p and n must be positive.")
    if me < 0 or mi < 0:
        raise ValueError("me and mi must be nonnegative.")
    if me > n:
        raise ValueError("me cannot exceed n.")
    if x_l.shape != (p,) or x_u.shape != (p,):
        raise ValueError(f"x_L and x_U must both have shape ({p},).")


def _model_dims(model_def, *, p: int, param_name: str = "x"):
    y0 = jnp.zeros((model_def.var_spec.total_size,), dtype=model_def.dtype)
    x0 = jnp.zeros((p,), dtype=model_def.dtype)
    me = int(model_def.eq_residual({param_name: x0}, y0).shape[0])
    mi = int(model_def.ineq_residual({param_name: x0}, y0).shape[0])
    n = int(model_def.var_spec.total_size)
    return n, me, mi


def _validate_model_matches_cfg(model_def, data_cfg: dict, *, p: int, param_name: str = "x") -> None:
    n, me, mi = _model_dims(model_def, p=p, param_name=param_name)
    if n != int(data_cfg["n"]) or me != int(data_cfg["me"]) or mi != int(data_cfg["mi"]):
        raise ValueError(
            "Built model does not match data.json: "
            f"expected (n={data_cfg['n']}, me={data_cfg['me']}, mi={data_cfg['mi']}) "
            f"but got (n={n}, me={me}, mi={mi})."
        )


def _generate_dataset(model_def, data_cfg: dict):
    start_time = time.perf_counter()
    rng = np.random.default_rng(int(data_cfg["seed"]))
    p = int(data_cfg["p"])
    num_samples = int(data_cfg["num_samples"])
    x_l = np.asarray(data_cfg["x_L"], dtype=float)
    x_u = np.asarray(data_cfg["x_U"], dtype=float)
    xs = rng.uniform(x_l, x_u, size=(num_samples, p))

    solver = SolGenModel(model_def)
    ys = []
    mus = []
    solve_time_sec = 0.0
    solver_name = resolve_solver_name(data_cfg, default="SCS")
    mi = int(data_cfg["mi"])
    for x in xs:
        result = solver.solve({"x": jnp.asarray(x, dtype=model_def.dtype)}, solver=solver_name)
        ys.append(np.asarray(result.y, dtype=np.float64))
        mus.append(coerce_ineq_multipliers(result.mu, mi))
        solve_time_sec += float(result.solve_time_sec or 0.0)

    metadata = {
        "problem_type": normalize_problem_type(str(data_cfg["type"])),
        "p": p,
        "n": int(data_cfg["n"]),
        "me": int(data_cfg["me"]),
        "mi": int(data_cfg["mi"]),
        "num_samples": num_samples,
        "solver": solver_name,
        "seed": int(data_cfg["seed"]),
        "optimizer_generation_wall_time_sec": solve_time_sec,
        "optimizer_generation_total_wall_time_sec": time.perf_counter() - start_time,
    }
    Mu = np.stack(mus, axis=0) if mus else np.zeros((num_samples, mi), dtype=np.float64)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), Mu, metadata


def _generate_dataset_from_generator(generator, data_cfg: dict):
    start_time = time.perf_counter()
    p = int(data_cfg["p"])
    num_samples = int(data_cfg["num_samples"])
    xs = generator.sample_parameters(num_samples)

    kept_x = []
    kept_y = []
    kept_mu = []
    objectives = []
    status_counts = {}
    solve_time_sec = 0.0
    mi = int(getattr(generator, "n_ineq", data_cfg["mi"]))

    for x in xs:
        result = generator.solve_for_x(x)
        status = str(result["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        solve_time_sec += float(result.get("solve_time_sec") or 0.0)
        if status in ("optimal", "optimal_inaccurate") and result["y"] is not None:
            kept_x.append(np.asarray(x, dtype=np.float64))
            kept_y.append(np.asarray(result["y"], dtype=np.float64))
            kept_mu.append(coerce_ineq_multipliers(result.get("mu"), mi))
            objectives.append(float(result["objective"]) if result["objective"] is not None else np.nan)
            if len(kept_x) >= num_samples:
                break

    if len(kept_x) < num_samples:
        raise RuntimeError(
            f"Only collected {len(kept_x)} successful points out of requested {num_samples}. "
            "Increase num_samples or switch solver settings."
        )

    metadata = {
        "problem_type": normalize_problem_type(str(data_cfg["type"])),
        "p": p,
        "n": int(data_cfg["n"]),
        "me": int(data_cfg["me"]),
        "mi": int(data_cfg["mi"]),
        "num_samples": num_samples,
        "solver": resolve_solver_name(data_cfg, default="SCS"),
        "seed": int(data_cfg["seed"]),
        "objective_min": float(np.nanmin(objectives)) if objectives else np.nan,
        "objective_max": float(np.nanmax(objectives)) if objectives else np.nan,
        "objective_mean": float(np.nanmean(objectives)) if objectives else np.nan,
        "status_counts": status_counts,
        "optimizer_generation_wall_time_sec": solve_time_sec,
        "optimizer_generation_total_wall_time_sec": time.perf_counter() - start_time,
    }
    Mu = np.stack(kept_mu, axis=0) if kept_mu else np.zeros((len(kept_x), mi), dtype=np.float64)
    return np.asarray(kept_x, dtype=np.float64), np.asarray(kept_y, dtype=np.float64), Mu, metadata


def _config_hash(cfg_dict: dict, proj_cfg: dict) -> str:
    payload = json.dumps({"config": cfg_dict, "proj": proj_cfg}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def _consistency(metrics: dict) -> float:
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


def run_case(
    case_dir: Path,
    *,
    data_cfg_override: Optional[dict] = None,
    cfg_dict_override: Optional[dict] = None,
    proj_cfg_override: Optional[dict] = None,
    output_dir_override: Optional[Path] = None,
) -> int:
    data_cfg = copy.deepcopy(data_cfg_override) if data_cfg_override is not None else _load_json(case_dir / "data.json")
    cfg_dict = copy.deepcopy(cfg_dict_override) if cfg_dict_override is not None else _load_json(case_dir / "config.json")
    proj_cfg = copy.deepcopy(proj_cfg_override) if proj_cfg_override is not None else _load_json(case_dir / "proj.json")
    _validate_data_cfg(data_cfg)
    print_every = max(1, int(cfg_dict.get("print_every", 10)))

    merged_cfg = {
        key: value
        for key, value in cfg_dict.items()
        if key not in {"print_every", "model"}
    }
    merged_cfg["IS_FIXED"] = str(proj_cfg.get("cp_mode", "fixed")).lower() == "fixed"
    merged_cfg["safety"] = float(proj_cfg.get("safety", 0.95))
    merged_cfg["knorm_iters"] = int(proj_cfg.get("knorm_iters", 20))
    merged_cfg["knorm_seed"] = int(proj_cfg.get("knorm_seed", 42))
    merged_cfg["adjoint_iters"] = int(proj_cfg.get("adjoint_iters", 30))
    merged_cfg["use_ruiz"] = bool(proj_cfg.get("use_ruiz", True))
    merged_cfg["ruiz_iters"] = int(proj_cfg.get("ruiz_iters", 4))
    merged_cfg["k_layer"] = int(proj_cfg.get("k_layer", 1))
    cfg: TrainConfig = cfg_from_dict(merged_cfg)

    p = int(data_cfg["p"])
    n = int(data_cfg["n"])
    me = int(data_cfg["me"])
    mi = int(data_cfg["mi"])
    train_dtype = resolve_dtype(cfg.dtype)
    device = select_device(cfg.device)

    if uses_nonconvex_generator(data_cfg):
        generator = build_problem_generator(data_cfg)
        problem_data = dict(generator.get_problem_data())
        problem_data["problem_type"] = normalize_problem_type(str(data_cfg["type"]))
        label_model_def = build_problem_model_from_data(problem_data, dtype=jnp.float64)
        dataset = ensure_cached_dataset(
            case_dir,
            data_cfg,
            lambda: _generate_dataset_from_generator(generator, data_cfg),
            force=bool(data_cfg.get("force_regenerate", False)),
        )
        train_model_def = build_problem_model_from_data(problem_data, dtype=train_dtype)
    else:
        label_model_def = build_problem_model(data_cfg, dtype=jnp.float64)
        dataset = ensure_cached_dataset(
            case_dir,
            data_cfg,
            lambda: _generate_dataset(label_model_def, data_cfg),
            force=bool(data_cfg.get("force_regenerate", False)),
        )
        train_model_def = build_problem_model(data_cfg, dtype=train_dtype)

    _validate_model_matches_cfg(label_model_def, data_cfg, p=p)

    print(f"Runtime: {runtime_summary(device, train_dtype)}")
    print(f"Dataset: {dataset.dataset_dir}")
    print(f"Dataset status: {'generated' if dataset.generated else 'reused'}")
    _validate_model_matches_cfg(train_model_def, data_cfg, p=p)

    X_np = dataset.X
    Y_np = dataset.Y

    rng = np.random.default_rng(cfg.seed)
    idx = np.arange(X_np.shape[0])
    rng.shuffle(idx)
    X_np = X_np[idx]
    Y_np = Y_np[idx]

    total_samples = X_np.shape[0]
    n_train = int(cfg.train_frac * total_samples)
    bs = cfg.batch_size
    n_train2 = (n_train // bs) * bs
    n_val2 = ((total_samples - n_train) // bs) * bs
    if n_train2 < bs or n_val2 < bs:
        raise ValueError("Not enough samples for at least one train and one validation batch.")

    X_train = jax.device_put(jnp.asarray(X_np[:n_train2], dtype=train_dtype), device)
    X_val = jax.device_put(jnp.asarray(X_np[n_train2:n_train2 + n_val2], dtype=train_dtype), device)
    Y_train = jax.device_put(jnp.asarray(Y_np[:n_train2], dtype=train_dtype), device)
    Y_val = jax.device_put(jnp.asarray(Y_np[n_train2:n_train2 + n_val2], dtype=train_dtype), device)

    Y_train_batches = Y_train.reshape((n_train2 // bs, bs, n))
    Y_val_batches = Y_val.reshape((n_val2 // bs, bs, n))
    train_batches = jax.device_put(make_fixed_batches(X_train, bs), device)
    val_batches = jax.device_put(make_fixed_batches(X_val, bs), device)

    print(f"JAXMODEL {normalize_problem_type(str(data_cfg['type'])).upper()}  p={p} n={n} me={me} mi={mi}")
    print(f"batch_size={bs}  train_batches={train_batches.shape[0]}  val_batches={val_batches.shape[0]}")

    model, init_state, train_step, eval_step = build_train_fns_from_jaxmodel(
        model_def=train_model_def,
        cfg=cfg,
        p=p,
        param_name="x",
    )
    state = jax.device_put(init_state(jax.random.PRNGKey(cfg.seed)), device)
    sub_layer = make_subproblem_layer_from_model(train_model_def, param_name="x")

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
            p=p,
            dtype=train_dtype,
            device=device,
        )
        print("JIT warmup: completed for train/eval step.")

    train_epoch, eval_epoch = build_epoch_fns(train_step, eval_step)
    viol_fn = build_violation_fn_from_jaxmodel(train_model_def, cfg=cfg, p=p, param_name="x")
    n_train_batches = int(train_batches.shape[0])
    n_val_batches = int(val_batches.shape[0])
    train_epoch_time_total = 0.0
    val_epoch_time_total = 0.0
    train_epoch_time_tracked = 0.0
    val_epoch_time_tracked = 0.0
    history_epochs: list[int] = []
    history_train_objective: list[float] = []
    history_val_objective: list[float] = []
    history_train_worst_gap_pct: list[float] = []
    history_val_worst_gap_pct: list[float] = []
    history_train_violation: list[float] = []
    history_val_violation: list[float] = []

    @jax.jit
    def batched_reference_objective_for_gap(x_batch, y_batch):
        def one(x, y):
            return label_model_def.objective_value(
                {"x": jnp.asarray(x, dtype=label_model_def.dtype)},
                jnp.asarray(y, dtype=label_model_def.dtype),
            )

        return jax.vmap(one)(x_batch, y_batch)

    @jax.jit
    def worst_relative_gap_pct_over_batches(params, x_batches, y_label_batches):
        def one_batch(xb, yb):
            y_pred = predict_y_tilde(params, xb)
            pred_obj = batched_reference_objective_for_gap(xb, y_pred)
            ref_obj = batched_reference_objective_for_gap(xb, yb)
            rel_gap_pct = 100.0 * jnp.abs(pred_obj - ref_obj) / jnp.maximum(1.0, jnp.abs(ref_obj))
            return jnp.max(rel_gap_pct)

        vals = jax.vmap(one_batch)(x_batches, y_label_batches)
        return jnp.max(vals)

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
        train_worst_gap_pct = float(worst_relative_gap_pct_over_batches(state.params, train_batches, Y_train_batches))
        val_worst_gap_pct = float(worst_relative_gap_pct_over_batches(state.params, val_batches, Y_val_batches))
        history_epochs.append(ep)
        history_train_objective.append(float(tr_m["obj"]))
        history_val_objective.append(float(va_m["obj"]))
        history_train_worst_gap_pct.append(train_worst_gap_pct)
        history_val_worst_gap_pct.append(val_worst_gap_pct)
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
    eq_mean = float(
        (tr_viol["eq_mean"] * train_batches.shape[0] + va_viol["eq_mean"] * val_batches.shape[0]) / total_batches
    )
    ineq_max = float(jnp.maximum(tr_viol["ineq_inf"], va_viol["ineq_inf"]))
    ineq_mean = float(
        (tr_viol["ineq_mean"] * train_batches.shape[0] + va_viol["ineq_mean"] * val_batches.shape[0]) / total_batches
    )
    bnd_max = float(jnp.maximum(tr_viol["bound_inf"], va_viol["bound_inf"]))
    bnd_mean = float(
        (tr_viol["bound_mean"] * train_batches.shape[0] + va_viol["bound_mean"] * val_batches.shape[0]) / total_batches
    )

    print("\n=== ORIGINAL constraint violation (max over train+val) ===")
    print(f"Equality   ||A y - (b+Bx)||_inf : {fmt_sci(eq_max)}")
    print(f"Inequality max(·,0)_inf         : {fmt_sci(ineq_max)}")
    print(f"Bounds     max(lb,ub)_inf       : {fmt_sci(bnd_max)}\n")

    @jax.jit
    def mse_over_batches(params, x_batches, y_batches):
        def one_batch(xb, yb):
            y_pred = predict_y_tilde(params, xb)
            return jnp.mean((y_pred - yb) ** 2)

        vals = jax.vmap(one_batch)(x_batches, y_batches)
        return jnp.mean(vals)

    tr_mse = mse_over_batches(state.params, train_batches, Y_train_batches)
    va_mse = mse_over_batches(state.params, val_batches, Y_val_batches)
    mse_final = float((tr_mse * train_batches.shape[0] + va_mse * val_batches.shape[0]) / total_batches)
    print("\n=== Supervised evaluation (against solgen labels) ===")
    print(f"MSE(y_tilde, y_true): {fmt_sci(mse_final)}\n")

    @jax.jit
    def batched_objective_train(x_batch, y_batch):
        def one(x, y):
            return train_model_def.objective_value(
                {"x": jnp.asarray(x, dtype=train_model_def.dtype)},
                jnp.asarray(y, dtype=train_model_def.dtype),
            )

        return jax.vmap(one)(x_batch, y_batch)

    @jax.jit
    def batched_objective_reference(x_batch, y_batch):
        def one(x, y):
            return label_model_def.objective_value(
                {"x": jnp.asarray(x, dtype=label_model_def.dtype)},
                jnp.asarray(y, dtype=label_model_def.dtype),
            )

        return jax.vmap(one)(x_batch, y_batch)

    @jax.jit
    def mean_reference_objective_over_batches(x_batches, y_batches):
        def one_batch(xb, yb):
            return jnp.mean(batched_objective_reference(xb, yb))

        vals = jax.vmap(one_batch)(x_batches, y_batches)
        return jnp.mean(vals)

    @jax.jit
    def relative_gap_over_batches(params, x_batches, y_label_batches):
        def one_batch(xb, yb):
            y_pred = predict_y_tilde(params, xb)
            pred_obj = batched_objective_reference(xb, y_pred)
            ref_obj = batched_objective_reference(xb, yb)
            return jnp.mean(jnp.abs(pred_obj - ref_obj) / jnp.maximum(1.0, jnp.abs(ref_obj)))

        vals = jax.vmap(one_batch)(x_batches, y_label_batches)
        return jnp.mean(vals)

    rel_gap_train = relative_gap_over_batches(state.params, train_batches, Y_train_batches)
    rel_gap_val = relative_gap_over_batches(state.params, val_batches, Y_val_batches)
    rel_gap = float(
        (rel_gap_train * train_batches.shape[0] + rel_gap_val * val_batches.shape[0]) / total_batches
    )
    ref_obj_train = float(mean_reference_objective_over_batches(train_batches, Y_train_batches))
    ref_obj_val = float(mean_reference_objective_over_batches(val_batches, Y_val_batches))
    objective_value = float(
        (float(tr_m["obj"]) * train_batches.shape[0] + float(va_m["obj"]) * val_batches.shape[0]) / total_batches
    )
    consistency_value = float(
        (_consistency(tr_m) * train_batches.shape[0] + _consistency(va_m) * val_batches.shape[0]) / total_batches
    )
    print("=== Optimality gap (relative objective difference vs solgen) ===")
    print(f"Relative objective gap: {fmt_sci(rel_gap)}\n")

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
        obj = jnp.mean(batched_objective_train(x_batch, y_tilde))
        mse_y = jnp.mean((y_hat - y_tilde) ** 2)
        mse_lam = jnp.mean((lam_hat - lam_tilde) ** 2) if me > 0 else jnp.asarray(0.0, dtype=train_dtype)
        mse_mu = jnp.mean((mu_hat - mu_tilde) ** 2) if mi > 0 else jnp.asarray(0.0, dtype=train_dtype)
        consistency = mse_y + mse_lam + mse_mu
        return obj + jnp.asarray(cfg.alpha_consistency, dtype=train_dtype) * consistency

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

    run_hash = _config_hash(cfg_dict, proj_cfg)
    if output_dir_override is not None:
        output_dir = Path(output_dir_override)
        output_dir.mkdir(parents=True, exist_ok=True)
        params_path = output_dir / "trained_params.npz"
        metrics_path = output_dir / "summary.json"
        history_path = output_dir / "run_history.json"
        plot_path = output_dir / "training_metrics.png"
        _write_json(output_dir / "data.json", data_cfg)
        _write_json(output_dir / "config.json", cfg_dict)
        _write_json(output_dir / "proj.json", proj_cfg)
    else:
        output_dir = dataset.dataset_dir
        params_path = dataset.dataset_dir / f"trained_params_{run_hash}.npz"
        metrics_path = dataset.dataset_dir / f"run_metrics_{run_hash}.json"
        history_path = dataset.dataset_dir / f"run_history_{run_hash}.json"
        plot_path = dataset.dataset_dir / "training_metrics.png"
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
        "optimality_gap": rel_gap,
        "eq_inf": eq_max,
        "ineq_inf": ineq_max,
        "bound_inf": bnd_max,
        "mse_y_tilde_vs_label": mse_final,
        "relative_objective_gap": rel_gap,
        "training_wall_time_sec": training_wall_time,
        "device": device.platform,
        "dtype": jnp.dtype(train_dtype).name,
    }
    metrics_payload.update(timing_summary)
    history_payload = {
        "run_hash": run_hash,
        "config_seed": int(cfg_dict["seed"]),
        "data_seed": int(data_cfg["seed"]),
        "epochs": history_epochs,
        "train_objective": history_train_objective,
        "val_objective": history_val_objective,
        "train_worst_relative_gap_pct": history_train_worst_gap_pct,
        "val_worst_relative_gap_pct": history_val_worst_gap_pct,
        "train_violation": history_train_violation,
        "val_violation": history_val_violation,
        "train_reference_objective": ref_obj_train,
        "val_reference_objective": ref_obj_val,
        "timing_start_epoch": int(timing_summary["timing_start_epoch"]),
        "timing_epochs_recorded": int(timing_summary["timing_epochs_recorded"]),
    }
    history_payload.update(history_optimizer_timing_fields(dataset.metadata))
    _write_json(history_path, history_payload)
    save_objective_violation_plot(
        plot_path,
        epochs=history_epochs,
        train_gap_pct=history_train_worst_gap_pct,
        val_gap_pct=history_val_worst_gap_pct,
        train_violation=history_train_violation,
        val_violation=history_val_violation,
        title=f"NLPOpt {normalize_problem_type(str(data_cfg['type'])).upper()} Training",
    )
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
    return 0


def main() -> int:
    return run_case(Path(__file__).resolve().parent)


if __name__ == "__main__":
    raise SystemExit(main())
