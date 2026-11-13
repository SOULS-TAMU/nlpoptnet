"""Optional original-problem diagnostics and adaptive inference."""

from inspect import signature
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np
from ohagi.execution.inference import require_jax_backend
from purin.execution.runtime import resolve_jax_dtype
from purin.solvers.chambolle_pock.dense_layer import solve_dense_implicit

from .projection import solve_cp_implicit


def _scale(value):
    return 1.0 if value is None else value


def _maximum(values):
    return jnp.max(values, initial=0.0)


def _primal(model, scaling, x, y):
    params = {"x": x}
    equality = model.eq_residual(params, y) * _scale(scaling.D_eq)
    inequality = model.ineq_residual(params, y) * _scale(scaling.D_ineq)
    lower = model.lower_bounds(params)
    upper = model.upper_bounds(params)
    lower = -jnp.inf if lower is None else lower
    upper = jnp.inf if upper is None else upper
    lower = (lower - y) * _scale(scaling.D_v)
    upper = (y - upper) * _scale(scaling.D_v)
    violations = {
        "equality": _maximum(jnp.abs(equality)),
        "inequality": _maximum(inequality),
        "bounds": jnp.maximum(_maximum(lower), _maximum(upper)),
    }
    # A -inf bound residual denotes an absent bound. Constraint residuals
    # must be finite even if a nonfinite inequality would pass a sign check.
    valid = jnp.all(jnp.isfinite(y)) & jnp.all(jnp.isfinite(equality))
    valid &= jnp.all(jnp.isfinite(inequality))
    primal = jnp.max(jnp.stack(tuple(violations.values())))
    violations["primal"] = jnp.where(valid, primal, jnp.inf)
    return violations, inequality, lower, upper


def _bound_dual(residual, gradient, tolerance):
    finite = jnp.isfinite(residual)
    multiplier = jnp.where(
        finite & (residual >= -tolerance), jnp.maximum(gradient, 0), 0
    )
    product = multiplier * jnp.where(finite, residual, 0)
    return multiplier, product


def _certificate(model, scaling, x, y, equality, inequality, primal_tolerance):
    residuals, g, lower, upper = _primal(model, scaling, x, y)
    params = {"x": x}
    gradient = model.grad_y_objective(params, y)
    gradient += model.jac_y_eq(params, y).T @ equality
    gradient += model.jac_y_ineq(params, y).T @ inequality
    gradient *= scaling.D_obj / _scale(scaling.D_v)
    lower_dual, lower_product = _bound_dual(lower, gradient, primal_tolerance)
    upper_dual, upper_product = _bound_dual(upper, -gradient, primal_tolerance)
    equality = equality * (scaling.D_obj / _scale(scaling.D_eq))
    inequality = inequality * (scaling.D_obj / _scale(scaling.D_ineq))
    products = jnp.concatenate((inequality * g, lower_product, upper_product))
    residuals.update(
        dual=_maximum(-inequality),
        complementarity=_maximum(jnp.abs(products)),
        stationarity=_maximum(jnp.abs(gradient - lower_dual + upper_dual)),
    )
    multipliers = {
        "equality": equality,
        "inequality": inequality,
        "lower": lower_dual,
        "upper": upper_dual,
    }
    return residuals, multipliers


def _compiled_inference(frontend, state):
    cached = getattr(frontend, "_diagnostics_cache", None)
    if cached is not None and cached[0] is state:
        return cached[1:]
    model, scaling = state.model, frontend.scaling
    count_supported = "return_iterations" in signature(solve_dense_implicit).parameters

    @jax.jit
    def initialize(params, inputs):
        return state.backbone.apply({"params": params}, inputs)

    @jax.jit
    def step(inputs, y, equality, inequality):
        return solve_cp_implicit(
            *state.sub_layer(inputs, y),
            cfg=state.cfg,
            y0=y,
            lam0=equality,
            mu0=inequality,
            return_iterations=count_supported,
        )

    @jax.jit
    def primal(inputs, y):
        return jax.vmap(lambda x, z: _primal(model, scaling, x, z)[0]["primal"])(
            inputs, y
        )

    @jax.jit
    def certificate(inputs, y, equality, inequality, tolerance):
        def single(x, z, eq, iq):
            return _certificate(model, scaling, x, z, eq, iq, tolerance)

        return jax.vmap(single)(inputs, y, equality, inequality)

    frontend._diagnostics_cache = (state, initialize, step, primal, certificate)
    return frontend._diagnostics_cache[1:]


def _options(state, adaptive, max_layers, tolerances):
    if max_layers is not None:
        if not adaptive:
            raise ValueError("max_layers requires adaptive=True.")
        if (
            isinstance(max_layers, bool)
            or not isinstance(max_layers, Integral)
            or max_layers < 1
        ):
            raise ValueError("max_layers must be a positive integer.")
    if any(not np.isfinite(value) or value < 0 for value in tolerances.values()):
        raise ValueError("KKT tolerances must be finite and nonnegative.")
    return int(state.cfg.k_layer if max_layers is None else max_layers)


def _run_layers(step, primal, inputs, initial, depth, adaptive, tolerance):
    active = np.arange(inputs.shape[0])
    used = np.zeros(inputs.shape[0], dtype=np.int32)
    counts = []
    solution = initial
    for _ in range(depth):
        if adaptive:
            result = step(inputs[active], *(value[active] for value in solution))
            solution = tuple(
                value.at[active].set(new) for value, new in zip(solution, result)
            )
        else:
            result = step(inputs, *solution)
            solution = result[:3]
        used[active] += 1
        if len(result) == 4:
            row_counts = np.zeros_like(used)
            row_counts[active] = np.asarray(result[3])
            counts.append(row_counts)
        if adaptive:
            violation = np.asarray(primal(inputs[active], result[0]))
            active = active[~(violation <= tolerance)]
            if active.size == 0:
                break
    return solution, used, np.stack(counts, axis=1) if counts else None


def predict_with_info(
    self,
    values,
    *,
    adaptive=False,
    max_layers=None,
    primal_tolerance=1e-6,
    dual_tolerance=1e-6,
    complementarity_tolerance=1e-6,
    stationarity_tolerance=1e-6,
    projection_backend="auto",
):
    """Predict with original-problem KKT residuals for a sample or batch.

    Args:
        values: One parameter vector or a matrix of parameter vectors.
        adaptive: Stop projecting each sample once its original primal
            residual satisfies primal_tolerance. Otherwise use trained depth.
        max_layers: Positive projection cap for adaptive inference. Defaults
            to the trained depth.
        primal_tolerance: Absolute feasibility tolerance, including bounds.
        dual_tolerance: Absolute inequality multiplier tolerance.
        complementarity_tolerance: Absolute multiplier-slack tolerance.
        stationarity_tolerance: Absolute Lagrangian gradient tolerance.
        projection_backend: JAX backend selector, as in predict.

    Returns:
        A dictionary with prediction, residuals, tolerances, multipliers,
        primal_feasible, kkt_satisfied, layers_used, and reached_layer_limit.
        cp_iterations contains actual counts per sample and executed layer
        (zero for an inactive sample), or None with older PURIN versions.
        A single input vector removes the leading sample axis. Residuals and
        multipliers use the original problem coordinates.

    Notes:
        Bound multipliers are estimated from the gradient at active bounds.
        For nonlinear or nonconvex problems, passing the reported tolerances
        certifies approximate first-order conditions only. Adaptive batches
        remove completed samples, which can compile new JAX batch shapes.
    """
    state = self._inference_state
    if state is None:
        raise RuntimeError(
            "Please train or load the model before calling predict_with_info()."
        )
    require_jax_backend(projection_backend)
    data = np.asarray(values, dtype=np.float64)
    squeeze = data.ndim == 1
    data = np.atleast_2d(data)
    if data.ndim != 2 or data.shape[1] != state.n_x or data.shape[0] == 0:
        raise ValueError(
            f"Expected one or more vectors with {state.n_x} parameter values."
        )
    if self.scaling.enabled:
        data = data / self.scaling.D_p
    inputs = jnp.asarray(
        data, dtype=resolve_jax_dtype(str(state.train_config["dtype"]))
    )
    tolerances = dict(
        primal=primal_tolerance,
        dual=dual_tolerance,
        complementarity=complementarity_tolerance,
        stationarity=stationarity_tolerance,
    )
    depth = _options(state, adaptive, max_layers, tolerances)
    initialize, step, primal, certificate = _compiled_inference(self, state)
    initial = initialize(state.params, inputs)
    solution, used, counts = _run_layers(
        step, primal, inputs, initial, depth, adaptive, primal_tolerance
    )
    residuals, multipliers = certificate(inputs, *solution, primal_tolerance)
    feasible = residuals["primal"] <= primal_tolerance
    checks = [
        jnp.isfinite(residuals[key]) & (residuals[key] <= value)
        for key, value in tolerances.items()
    ]
    result = dict(
        prediction=self._unscale_variables(np.asarray(solution[0], dtype=np.float64)),
        residuals=residuals,
        multipliers=multipliers,
        primal_feasible=feasible,
        kkt_satisfied=jnp.all(jnp.stack(checks), axis=0),
        layers_used=used,
        reached_layer_limit=(used == depth) & ~feasible,
        cp_iterations=counts,
    )
    output = jax.tree.map(
        lambda value: np.asarray(value)[0] if squeeze else np.asarray(value), result
    )
    output["tolerances"] = tolerances
    return output
