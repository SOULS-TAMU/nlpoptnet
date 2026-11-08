"""Problem resources for the NLPOptNet frontend."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nlpoptnet.api.model import NLPOptNet

from nlpoptnet.api.formulation.expressions import Constraint
from nlpoptnet.api.formulation.builder import HighLevelNLPBuilder
from nlpoptnet.api.formulation.expressions import _BlockConstraint
from nlpoptnet.api.configuration import _BuildState
from nlpoptnet.api.formulation.expressions import _EvalContext
from nlpoptnet.api.configuration import _METRIC_KEYS
from nlpoptnet.api.configuration import _REQUIRED_CONFIG_KEYS
from nlpoptnet.api.configuration import _apply_config_aliases
from ohagi.training.engine import build_compiled_epoch_fns as build_epoch_fns
from nlpoptnet.api.fit import build_training
from nlpoptnet.api.objectives import build_violation
from nlpoptnet.api.configuration import cfg_from_dict
from nlpoptnet.api.formulation.constraints import eq_block as eq_block
from nlpoptnet.api.formulation.constraints import ineq_block as ineq_block
import jax
import jax.numpy as jnp
from nlpoptnet.api.objectives import make_batched_objective
from ohagi.data.batching import trimmed_batches as make_fixed_batches
from nlpoptnet.api.formulation.approximation import make_subproblem_layer_from_model
import numpy as np
from purin.execution.runtime import resolve_jax_dtype as resolve_dtype
from ohagi.data.sampling import split_train_val
from ohagi.training.stateful import warmup_stateful as warmup_compile


def build(self) -> "NLPOptNet":
    """Build the symbolic model, dataset split, and train-time state."""
    _validate_nlp_definition(self)
    _apply_config_aliases(self.config)

    n_x = len(self.parameter_names)
    n_y = len(self.variable_names)
    dtype = resolve_dtype(str(self.config["dtype"]))

    # Bounds are first built in ORIGINAL coordinates:
    #   lower_original(x_original) <= y_original <= upper_original(x_original)
    lower_M_orig, lower_c_orig, upper_M_orig, upper_c_orig = self._build_box_bounds(
        n_x=n_x, n_y=n_y
    )

    # Load/generate ORIGINAL parameter samples before building the JAX model.
    X_original = self._load_or_sample_parameters(n_x=n_x)

    # Compute D_p and D_v from original samples/bounds.
    self._compute_basic_scales(
        X_original=X_original,
        lower_M=lower_M_orig,
        lower_c=lower_c_orig,
        upper_M=upper_M_orig,
        upper_c=upper_c_orig,
    )

    X = self._scaled_parameters(X_original)

    D_p_jax = jnp.asarray(self.scaling.D_p, dtype=dtype)
    D_v_jax = jnp.asarray(self.scaling.D_v, dtype=dtype)

    # Convert original affine box bounds into scaled-coordinate box bounds.
    # Original:
    #   lower = lower_M_orig @ x_orig + lower_c_orig
    #   upper = upper_M_orig @ x_orig + upper_c_orig
    # With x_orig = D_p x_scaled and y_orig = D_v y_scaled:
    #   lower_scaled = D_v^-1 (lower_M_orig @ D_p x_scaled + lower_c_orig)
    #   upper_scaled = D_v^-1 (upper_M_orig @ D_p x_scaled + upper_c_orig)
    lower_M, lower_c, upper_M, upper_c = _scale_nlp_box_bounds(
        self, lower_M_orig, lower_c_orig, upper_M_orig, upper_c_orig
    )

    def make_ctx(params, vars_dict):
        x_scaled = jnp.ravel(params["x"])
        y_scaled = jnp.ravel(vars_dict["y"])

        if self.scaling.enabled:
            x_eval = D_p_jax * x_scaled
            y_eval = D_v_jax * y_scaled
        else:
            x_eval = x_scaled
            y_eval = y_scaled

        return _EvalContext(
            y=y_eval,
            x=x_eval,
            variable_index={name: idx for idx, name in enumerate(self.variable_names)},
            parameter_index={
                name: idx for idx, name in enumerate(self.parameter_names)
            },
        )

    if self._objective_callable is not None:
        user_objective_fun = self._objective_callable

        def objective_fun_unscaled(params, vars_dict):
            if not self.scaling.enabled:
                return user_objective_fun(params, vars_dict)

            x_orig = D_p_jax * jnp.ravel(params["x"])
            y_orig = D_v_jax * jnp.ravel(vars_dict["y"])
            return user_objective_fun({"x": x_orig}, {"y": y_orig})
    else:
        objective_expr = self._objective_expr
        assert objective_expr is not None

        def objective_fun_unscaled(params, vars_dict):
            return objective_expr.eval(make_ctx(params, vars_dict))

    eq_blocks, eq_constraints, ineq_blocks, ineq_constraints = _collect_nlp_constraints(
        self
    )

    D_v_jax, lower_M, lower_c, upper_M, upper_c = _refine_nlp_variable_scales(
        self,
        X_original,
        dtype,
        eq_blocks,
        eq_constraints,
        ineq_blocks,
        ineq_constraints,
        lower_M_orig,
        lower_c_orig,
        make_ctx,
        n_y,
        objective_fun_unscaled,
        upper_M_orig,
        upper_c_orig,
        D_v_jax,
        lower_M,
        lower_c,
        upper_M,
        upper_c,
    )

    self._estimate_objective_and_constraint_scales(
        objective_fun=objective_fun_unscaled,
        eq_constraints=eq_constraints,
        eq_blocks=eq_blocks,
        ineq_constraints=ineq_constraints,
        ineq_blocks=ineq_blocks,
        lower_M=lower_M_orig,
        lower_c=lower_c_orig,
        upper_M=upper_M_orig,
        upper_c=upper_c_orig,
        make_ctx=make_ctx,
        dtype=dtype,
        X_original=X_original,
    )

    D_obj_jax = jnp.asarray(float(self.scaling.D_obj), dtype=dtype)

    def objective_fun(params, vars_dict):
        return objective_fun_unscaled(params, vars_dict) / D_obj_jax

    builder = (
        HighLevelNLPBuilder(dtype=dtype)
        .add_parameter("x", n_x)
        .add_variable("y", n_y)
        .set_objective(objective_fun)
        .set_affine_lower_bound(
            var_name="y",
            param_name="x",
            M=jnp.asarray(lower_M, dtype=dtype),
            c=jnp.asarray(lower_c, dtype=dtype),
        )
        .set_affine_upper_bound(
            var_name="y",
            param_name="x",
            M=jnp.asarray(upper_M, dtype=dtype),
            c=jnp.asarray(upper_c, dtype=dtype),
        )
    )

    builder = _bind_nlp_constraint_blocks(
        self,
        D_p_jax,
        D_v_jax,
        builder,
        dtype,
        eq_blocks,
        eq_constraints,
        ineq_blocks,
        ineq_constraints,
        make_ctx,
    )

    model = builder.build(
        example_params={"x": jnp.zeros((n_x,), dtype=dtype)}, jit_compile=True
    )
    cfg, train_batches, train_cfg, train_idx, val_batches, val_idx = (
        _prepare_nlp_batches(self, X, dtype)
    )

    backbone, init_state_fn, train_step_fn, eval_step_fn = build_training(
        model_def=model,
        cfg=cfg,
        p=n_x,
    )
    train_epoch_fn, eval_epoch_fn = build_epoch_fns(
        train_step_fn, eval_step_fn, _METRIC_KEYS
    )
    state = init_state_fn(jax.random.PRNGKey(int(cfg.seed)))
    if bool(self.config.get("jit_warmup", True)):
        state = warmup_compile(
            batch_size=cfg.batch_size,
            seed=cfg.seed,
            commit_update=True,
            state=state,
            train_step_fn=train_step_fn,
            eval_step_fn=eval_step_fn,
            input_size=n_x,
            dtype=dtype,
            device=None,
        )

    self._build_state = _BuildState(
        model=model,
        cfg=cfg,
        train_config=train_cfg,
        dtype=dtype,
        X=X,
        X_original=X_original,
        train_idx=train_idx,
        val_idx=val_idx,
        train_batches=train_batches,
        val_batches=val_batches,
        backbone=backbone,
        init_state_fn=lambda: state,
        train_step_fn=train_step_fn,
        eval_step_fn=eval_step_fn,
        train_epoch_fn=train_epoch_fn,
        eval_epoch_fn=eval_epoch_fn,
        violation_fn=build_violation(model, cfg=cfg, p=n_x),
        objective_fn=make_batched_objective(model),
        sub_layer=make_subproblem_layer_from_model(model),
        lower_M=lower_M,
        lower_c=lower_c,
        upper_M=upper_M,
        upper_c=upper_c,
    )
    if bool(self.config.get("verbose", True)):
        print("🤖 Model build successfully!")
    return None


def _prepare_nlp_batches(self, X, dtype):
    """Prepare deterministic train/validation splits and fixed-shape batches."""
    train_idx, val_idx = split_train_val(
        X, train_frac=float(self.config["train_frac"]), seed=int(self.config["seed"])
    )
    effective_batch = int(
        min(int(self.config["batch_size"]), len(train_idx), len(val_idx))
    )
    train_cfg = self._train_cfg_dict(effective_batch)
    cfg = cfg_from_dict(train_cfg)

    X_train = jnp.asarray(X[train_idx], dtype=dtype)
    X_val = jnp.asarray(X[val_idx], dtype=dtype)
    train_batches = make_fixed_batches(X_train, cfg.batch_size)
    val_batches = make_fixed_batches(X_val, cfg.batch_size)
    if train_batches.shape[0] == 0 or val_batches.shape[0] == 0:
        raise ValueError(
            "Not enough samples for the configured batch size after the train/validation split."
        )
    return cfg, train_batches, train_cfg, train_idx, val_batches, val_idx


def _bind_nlp_constraint_blocks(
    self,
    D_p_jax,
    D_v_jax,
    builder,
    dtype,
    eq_blocks,
    eq_constraints,
    ineq_blocks,
    ineq_constraints,
    make_ctx,
):
    """Attach normalized equality and inequality evaluators to the problem builder."""
    if eq_constraints or eq_blocks:

        def eq_block(params, vars_dict):
            ctx = make_ctx(params, vars_dict)
            pieces = [jnp.ravel(entry.residual(ctx)) for entry in eq_constraints]

            for entry in eq_blocks:
                if self.scaling.enabled:
                    x_orig = D_p_jax * jnp.ravel(params["x"])
                    y_orig = D_v_jax * jnp.ravel(vars_dict["y"])
                    pieces.append(jnp.ravel(entry.fn({"x": x_orig}, {"y": y_orig})))
                else:
                    pieces.append(jnp.ravel(entry.fn(params, vars_dict)))

            out = (
                jnp.concatenate(pieces, axis=0)
                if pieces
                else jnp.zeros((0,), dtype=dtype)
            )

            if self.scaling.enabled and self.scaling.D_eq is not None:
                return out / jnp.asarray(self.scaling.D_eq, dtype=dtype).reshape(-1)

            return out

        builder = builder.add_nonlinear_equality(eq_block, name="nlpoptnet_eq_block")

    if ineq_constraints or ineq_blocks:

        def ineq_block(params, vars_dict):
            ctx = make_ctx(params, vars_dict)
            pieces = [jnp.ravel(entry.residual(ctx)) for entry in ineq_constraints]

            for entry in ineq_blocks:
                if self.scaling.enabled:
                    x_orig = D_p_jax * jnp.ravel(params["x"])
                    y_orig = D_v_jax * jnp.ravel(vars_dict["y"])
                    pieces.append(jnp.ravel(entry.fn({"x": x_orig}, {"y": y_orig})))
                else:
                    pieces.append(jnp.ravel(entry.fn(params, vars_dict)))

            out = (
                jnp.concatenate(pieces, axis=0)
                if pieces
                else jnp.zeros((0,), dtype=dtype)
            )

            if self.scaling.enabled and self.scaling.D_ineq is not None:
                return out / jnp.asarray(self.scaling.D_ineq, dtype=dtype).reshape(-1)

            return out

        builder = builder.add_nonlinear_inequality(
            ineq_block, name="nlpoptnet_ineq_block"
        )
    return builder


def _refine_nlp_variable_scales(
    self,
    X_original,
    dtype,
    eq_blocks,
    eq_constraints,
    ineq_blocks,
    ineq_constraints,
    lower_M_orig,
    lower_c_orig,
    make_ctx,
    n_y,
    objective_fun_unscaled,
    upper_M_orig,
    upper_c_orig,
    D_v_jax,
    lower_M,
    lower_c,
    upper_M,
    upper_c,
):
    """Apply the existing optional variable-scale sensitivity refinement."""
    if self.scaling.enabled and self.scaling.manual_D_v is None:
        D_v_raw = getattr(self.scaling, "_D_v_from_bounds_raw", self.scaling.D_v)

        D_v_linear = self._estimate_variable_scale_from_linear_inequalities(
            X_original=X_original,
            ineq_constraints=ineq_constraints,
            make_ctx=make_ctx,
            dtype=dtype,
        )

        if D_v_linear is not None:
            base = np.asarray(D_v_raw, dtype=np.float64)
            missing = base <= self.scaling.eps
            D_v_new = np.where(missing, D_v_linear, base)
            self.scaling.D_v = self._positive_scale(
                D_v_new,
                size=n_y,
                name="variable scale",
            )
        else:
            use_sensitivity = self.problem_type in {"nlp", "nonconvex"} and bool(
                self.scaling.use_jacobian_scaling
            )

            if use_sensitivity:
                self.scaling.D_v = self._estimate_variable_scale_from_sensitivities(
                    objective_fun=objective_fun_unscaled,
                    eq_constraints=eq_constraints,
                    eq_blocks=eq_blocks,
                    ineq_constraints=ineq_constraints,
                    ineq_blocks=ineq_blocks,
                    make_ctx=make_ctx,
                    dtype=dtype,
                    X_original=X_original,
                    D_v_base=np.asarray(D_v_raw, dtype=np.float64),
                )
            else:
                base = np.asarray(D_v_raw, dtype=np.float64)
                self.scaling.D_v = self._positive_scale(
                    np.where(base > self.scaling.eps, base, 1.0),
                    size=n_y,
                    name="variable scale",
                )

        D_v_jax = jnp.asarray(self.scaling.D_v, dtype=dtype)

        lower_M = (
            np.asarray(lower_M_orig, dtype=np.float64)
            * np.asarray(self.scaling.D_p, dtype=np.float64).reshape(1, -1)
        ) / np.asarray(self.scaling.D_v, dtype=np.float64).reshape(-1, 1)

        upper_M = (
            np.asarray(upper_M_orig, dtype=np.float64)
            * np.asarray(self.scaling.D_p, dtype=np.float64).reshape(1, -1)
        ) / np.asarray(self.scaling.D_v, dtype=np.float64).reshape(-1, 1)

        lower_c = np.asarray(lower_c_orig, dtype=np.float64) / np.asarray(
            self.scaling.D_v, dtype=np.float64
        )
        upper_c = np.asarray(upper_c_orig, dtype=np.float64) / np.asarray(
            self.scaling.D_v, dtype=np.float64
        )
    return D_v_jax, lower_M, lower_c, upper_M, upper_c


def _scale_nlp_box_bounds(self, lower_M_orig, lower_c_orig, upper_M_orig, upper_c_orig):
    """Convert affine box bounds to the existing scaled optimization coordinates."""
    lower_M = (
        np.asarray(lower_M_orig, dtype=np.float64)
        * np.asarray(self.scaling.D_p, dtype=np.float64).reshape(1, -1)
    ) / np.asarray(self.scaling.D_v, dtype=np.float64).reshape(-1, 1)
    upper_M = (
        np.asarray(upper_M_orig, dtype=np.float64)
        * np.asarray(self.scaling.D_p, dtype=np.float64).reshape(1, -1)
    ) / np.asarray(self.scaling.D_v, dtype=np.float64).reshape(-1, 1)
    lower_c = np.asarray(lower_c_orig, dtype=np.float64) / np.asarray(
        self.scaling.D_v, dtype=np.float64
    )
    upper_c = np.asarray(upper_c_orig, dtype=np.float64) / np.asarray(
        self.scaling.D_v, dtype=np.float64
    )
    return lower_M, lower_c, upper_M, upper_c


def _collect_nlp_constraints(self):
    """Partition symbolic and callable equality/inequality blocks."""
    eq_entries = list(self.constraints.equality.items)
    ineq_entries = list(self.constraints.inequality.items)

    eq_constraints = [entry for entry in eq_entries if isinstance(entry, Constraint)]
    eq_blocks = [entry for entry in eq_entries if isinstance(entry, _BlockConstraint)]
    ineq_constraints = [
        entry for entry in ineq_entries if isinstance(entry, Constraint)
    ]
    ineq_blocks = [
        entry for entry in ineq_entries if isinstance(entry, _BlockConstraint)
    ]
    return eq_blocks, eq_constraints, ineq_blocks, ineq_constraints


def _validate_nlp_definition(self):
    """Validate required nonlinear symbols, objective and configuration."""
    if self._objective_expr is None and self._objective_callable is None:
        raise ValueError("Set the objective before build().")
    if not self.parameter_names:
        raise ValueError("Add at least one parameter.")
    if not self.variable_names:
        raise ValueError("Add at least one variable.")
    missing_config = [
        key
        for key in _REQUIRED_CONFIG_KEYS
        if key not in self.config or self.config[key] is None
    ]
    if missing_config:
        raise ValueError(
            "Missing required config values: "
            + ", ".join(missing_config)
            + ".\nRequired values are epochs, batch_size, learning_rate, hidden_size, and hidden_layers.\n"
            + "Optional values are train_frac, seed, alpha_consistency, cp_mode, cp_iters, cp_tol, safety, knorm_iters, knorm_seed,\n"
            + "adjoint_iters, k_layer, use_ruiz, ruiz_iters, dtype, print_every, device and verbose"
        )
