"""Fit resources for the NLPOptNet frontend."""

from __future__ import annotations
from typing import Any
from nlpoptnet.api.network import Backbone
from nlpoptnet.api.formulation.compiler import JaxNLPModel
from nlpoptnet.api.configuration import TrainConfig
from nlpoptnet.api.reporting import _fmt_metric
from nlpoptnet.api.configuration import _metrics_tuple_to_dict
from ohagi.training.stateful import build_stateful_steps
from datetime import datetime
import jax
import jax.numpy as jnp
from nlpoptnet.api.objectives import make_batched_objective
from nlpoptnet.api.objectives import make_batched_residuals
from nlpoptnet.api.formulation.approximation import make_subproblem_layer_from_model
from ohagi.losses.lagrangian import modified_lagrangian_loss
import numpy as np
from nlpoptnet.api.projection import projected_model as projected_model
from nlpoptnet.serialization.artifacts import save_run
import time


def optimize(self) -> dict[str, Any]:
    """Train the model, save artifacts, and return run metadata."""
    if self._build_state is None:
        self.build()
    assert self._build_state is not None

    build_state = self._build_state
    state = build_state.init_state_fn()
    history: list[dict[str, Any]] = []
    start = time.perf_counter()
    if bool(self.config.get("verbose", True)):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"▶️ Model training started! [{now}]\n")
        COL_W = 12

        print("=" * 120)
        print(
            f"{'Epoch':>{COL_W}} | {'Training':^{COL_W * 4 + 3}} | {'Validation':^{COL_W * 4 + 3}}"
        )
        print("-" * 120)
        print(
            f"{'':>{COL_W}} | "
            f"{'Loss':>{COL_W}} {'Obj':>{COL_W}} {'Eq':>{COL_W}} {'Ineq':>{COL_W}} | "
            f"{'Loss':>{COL_W}} {'Obj':>{COL_W}} {'Eq':>{COL_W}} {'Ineq':>{COL_W}}"
        )
        print("=" * 120)

    from ohagi.training.engine import Engine, EpochPlan
    from ohagi.training.runner import EpochTransition

    def epoch_policy(epoch, state):
        epoch_result = yield EpochPlan(
            state, build_state.train_batches, kernel=build_state.train_epoch_fn
        )
        state, train_tuple = epoch_result.state, epoch_result.metrics
        val_tuple = build_state.eval_epoch_fn(state.params, build_state.val_batches)
        train_metrics = _metrics_tuple_to_dict(train_tuple)
        val_metrics = _metrics_tuple_to_dict(val_tuple)
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_objective": train_metrics["obj"],
            "val_objective": val_metrics["obj"],
            "train_consistency": train_metrics["consistency"],
            "val_consistency": val_metrics["consistency"],
            "train_eq_violation": train_metrics["eq_violation"],
            "val_eq_violation": val_metrics["eq_violation"],
            "train_ineq_violation": train_metrics["ineq_violation"],
            "val_ineq_violation": val_metrics["ineq_violation"],
            "train_mse_y": train_metrics["mse_y"],
            "val_mse_y": val_metrics["mse_y"],
            "train_mse_lam": train_metrics["mse_lam"],
            "val_mse_lam": val_metrics["mse_lam"],
            "train_mse_mu": train_metrics["mse_mu"],
            "val_mse_mu": val_metrics["mse_mu"],
        }
        history.append(record)
        if bool(self.config.get("verbose", True)) and (
            epoch == 1
            or epoch == int(build_state.cfg.epochs)
            or epoch % max(1, int(self.config.get("print_every", 10))) == 0
        ):
            print(
                f"{f'{epoch:03d}/{int(build_state.cfg.epochs):03d}':>{COL_W}} | "
                f"{_fmt_metric(record['train_loss'])} "
                f"{_fmt_metric(record['train_objective'])} "
                f"{_fmt_metric(record['train_eq_violation'])} "
                f"{_fmt_metric(record['train_ineq_violation'])} | "
                f"{_fmt_metric(record['val_loss'])} "
                f"{_fmt_metric(record['val_objective'])} "
                f"{_fmt_metric(record['val_eq_violation'])} "
                f"{_fmt_metric(record['val_ineq_violation'])}"
            )
        return EpochTransition(state)

    state = Engine.run(state, int(build_state.cfg.epochs), epoch_policy).state
    if bool(self.config.get("verbose", True)):
        print("=" * 120)
        print("\n")
    training_wall_time = time.perf_counter() - start
    if bool(self.config.get("verbose", True)):
        print(f"⏱️ Model training time (wall time) = {training_wall_time:.4f} seconds")
        print("✅ Model training finished!")
        print("💾 Model saved!")
        if bool(self.config.get("verbose", True)):
            # print("📦 Collecting summary...")
            print("📦 Postprocessing...")
            print("Meanwhile you can take a break and stay hydrated! 🧊💧😃")

    X_all = jnp.asarray(build_state.X, dtype=build_state.dtype)
    predictions_scaled = self._project_with_state(
        params=state.params,
        backbone=build_state.backbone,
        sub_layer=build_state.sub_layer,
        cfg=build_state.cfg,
        X=X_all,
    )
    objective_values = np.asarray(
        build_state.objective_fn(X_all, predictions_scaled), dtype=np.float64
    )
    if self.scaling.enabled:
        objective_values = objective_values * float(self.scaling.D_obj)
    violations = build_state.violation_fn(state.params, X_all)
    max_violation = max(
        float(np.asarray(violations["eq_inf"])),
        float(np.asarray(violations["ineq_inf"])),
        float(np.asarray(violations["bound_inf"])),
    )
    summary = {
        "framework": "nlpoptnet",
        "model_name": self.name,
        "problem_type": self.problem_type,
        "num_parameters": int(len(self.parameter_names)),
        "num_variables": int(len(self.variable_names)),
        "num_equalities": int(build_state.backbone.equality_size),
        "num_inequalities": int(build_state.backbone.inequality_size),
        "num_samples": int(build_state.X.shape[0]),
        "train_samples": int(build_state.train_idx.shape[0]),
        "val_samples": int(build_state.val_idx.shape[0]),
        "epochs": int(build_state.cfg.epochs),
        "batch_size": int(build_state.cfg.batch_size),
        "objective_mean": float(np.mean(objective_values)),
        "objective_min": float(np.min(objective_values)),
        "objective_max": float(np.max(objective_values)),
        "eq_inf": float(np.asarray(violations["eq_inf"])),
        "eq_mean": float(np.asarray(violations["eq_mean"])),
        "ineq_inf": float(np.asarray(violations["ineq_inf"])),
        "ineq_mean": float(np.asarray(violations["ineq_mean"])),
        "bound_inf": float(np.asarray(violations["bound_inf"])),
        "bound_mean": float(np.asarray(violations["bound_mean"])),
        "max_violation": float(max_violation),
        "training_wall_time_sec": float(training_wall_time),
        "final_train_loss": float(history[-1]["train_loss"]),
        "final_val_loss": float(history[-1]["val_loss"]),
        "final_train_objective": float(history[-1]["train_objective"]),
        "final_val_objective": float(history[-1]["val_objective"]),
        "final_train_consistency": float(history[-1]["train_consistency"]),
        "final_val_consistency": float(history[-1]["val_consistency"]),
        "final_train_eq_violation": float(history[-1]["train_eq_violation"]),
        "final_val_eq_violation": float(history[-1]["val_eq_violation"]),
        "final_train_ineq_violation": float(history[-1]["train_ineq_violation"]),
        "final_val_ineq_violation": float(history[-1]["val_ineq_violation"]),
    }

    # if bool(self.config.get("verbose", True)):
    #     print("✅ Summary collected")

    return save_run(
        self,
        build_state=build_state,
        state=state,
        history=history,
        summary=summary,
        predictions_scaled=predictions_scaled,
    )


def build_training(
    *,
    model_def: JaxNLPModel,
    cfg: TrainConfig,
    p: int,
    param_name: str = "x",
    rho: float = 1.0,
    use_diagonal_hessian: bool = True,
    diag_floor: float = 1e-8,
):
    """Build the backbone, initialization function, and train/eval steps."""
    from nlpoptnet.api.projection import projected_model

    example_x = jnp.zeros((p,), dtype=model_def.dtype)
    example_y = jnp.zeros((model_def.var_spec.total_size,), dtype=model_def.dtype)
    me = int(model_def.eq_residual({param_name: example_x}, example_y).shape[0])
    mi = int(model_def.ineq_residual({param_name: example_x}, example_y).shape[0])
    n = model_def.var_spec.total_size

    model = Backbone(
        p=p, n=n, me=me, mi=mi, hidden_size=cfg.hidden_size, hidden_dim=cfg.hidden_dim
    )
    sub_layer = make_subproblem_layer_from_model(
        model_def,
        param_name=param_name,
        rho=rho,
        use_diagonal_hessian=use_diagonal_hessian,
        diag_floor=diag_floor,
    )
    obj_fn = make_batched_objective(model_def, param_name=param_name)
    eq_resid_fn, ineq_resid_fn = make_batched_residuals(
        model_def, param_name=param_name
    )

    shared_model = projected_model(model, sub_layer, cfg)
    shared_loss = modified_lagrangian_loss(
        obj_fn, eq_resid_fn, ineq_resid_fn, alpha_consistency=cfg.alpha_consistency
    )

    @jax.jit
    def loss_fn(params, x_batch):
        outputs, _ = shared_model.apply(params, {}, x_batch)
        return shared_loss(outputs, {"theta": x_batch})

    init_state, train_step, eval_step = build_stateful_steps(
        loss_fn,
        model,
        batch_size=cfg.batch_size,
        input_size=p,
        dtype=model_def.dtype,
        learning_rate=cfg.learning_rate,
        invalid_batch_policy="propagate",
        include_update_accepted=False,
    )
    return model, init_state, train_step, eval_step
