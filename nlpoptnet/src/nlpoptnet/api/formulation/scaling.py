"""Scaling resources for the NLPOptNet frontend."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nlpoptnet.api.model import NLPOptNet

from nlpoptnet.api.formulation.expressions import _EvalContext
import jax
import jax.numpy as jnp
import numpy as np


def scale(
    self,
    *,
    parameter=None,
    variable=None,
    objective: float | None = None,
    equality=None,
    inequality=None,
    eps: float = 1e-12,
    objective_percentile: float = 95.0,
    constraint_percentile: float = 95.0,
    probe_points: int = 8,
    mode: str = "fast",
    use_jacobian_scaling: bool = False,
) -> "NLPOptNet":
    """
    Enable scaling.

    Internal scaled variables:
        x_original = D_p * x_scaled
        y_original = D_v * y_scaled

    Internal scaled model:
        f_scaled = f_original / D_obj
        h_scaled = h_original / D_eq
        g_scaled = g_original / D_ineq
    """
    self.scaling.enabled = True
    self.scaling.eps = float(eps)
    self.scaling.objective_percentile = float(objective_percentile)
    self.scaling.constraint_percentile = float(constraint_percentile)
    self.scaling.probe_points = int(probe_points)
    self.scaling.mode = str(mode).lower()
    self.scaling.use_jacobian_scaling = bool(use_jacobian_scaling)

    self.scaling.manual_D_p = (
        None
        if parameter is None
        else np.asarray(parameter, dtype=np.float64).reshape(-1)
    )
    self.scaling.manual_D_v = (
        None if variable is None else np.asarray(variable, dtype=np.float64).reshape(-1)
    )
    self.scaling.manual_D_obj = None if objective is None else float(objective)
    self.scaling.manual_D_eq = (
        None if equality is None else np.asarray(equality, dtype=np.float64).reshape(-1)
    )
    self.scaling.manual_D_ineq = (
        None
        if inequality is None
        else np.asarray(inequality, dtype=np.float64).reshape(-1)
    )

    self._build_state = None
    self._inference_state = None
    return self


def _positive_scale(
    self, value, *, size: int | None = None, name: str = "scale"
) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)

    if size is not None:
        if arr.size == 1:
            arr = np.full((size,), float(arr[0]), dtype=np.float64)
        if arr.shape != (size,):
            raise ValueError(f"{name} must have shape ({size},), got {arr.shape}.")

    arr = np.abs(arr)
    arr = np.where(np.isfinite(arr), arr, 1.0)
    arr = np.where(arr > float(self.scaling.eps), arr, 1.0)
    return arr


def _positive_scalar_scale(self, value, *, name: str = "scale") -> float:
    arr = self._positive_scale([value], size=1, name=name)
    return float(arr[0])


def _scaled_parameters(self, X_original: np.ndarray) -> np.ndarray:
    if not self.scaling.enabled:
        return np.asarray(X_original, dtype=np.float64)

    assert self.scaling.D_p is not None
    return np.asarray(X_original, dtype=np.float64) / self.scaling.D_p.reshape(1, -1)


def _unscale_variables(self, Y_scaled: np.ndarray) -> np.ndarray:
    if not self.scaling.enabled:
        return np.asarray(Y_scaled, dtype=np.float64)

    assert self.scaling.D_v is not None
    return np.asarray(Y_scaled, dtype=np.float64) * self.scaling.D_v.reshape(1, -1)


def _eval_affine_bounds_numpy(
    self, X_original: np.ndarray, lower_M, lower_c, upper_M, upper_c
):
    L = np.asarray(X_original, dtype=np.float64) @ np.asarray(
        lower_M, dtype=np.float64
    ).T + np.asarray(lower_c, dtype=np.float64)
    U = np.asarray(X_original, dtype=np.float64) @ np.asarray(
        upper_M, dtype=np.float64
    ).T + np.asarray(upper_c, dtype=np.float64)
    return L, U


def _estimate_variable_scale_from_linear_inequalities(
    self,
    *,
    X_original: np.ndarray,
    ineq_constraints,
    make_ctx,
    dtype,
) -> np.ndarray | None:
    n_y = len(self.variable_names)
    sample_count = min(int(self.scaling.probe_points), int(X_original.shape[0]))

    y0 = jnp.zeros((n_y,), dtype=dtype)
    scales = np.zeros((n_y,), dtype=np.float64)

    for i in range(sample_count):
        x0 = jnp.asarray(self._scaled_parameters(X_original[[i]])[0], dtype=dtype)

        def residual_y(y_scaled):
            params = {"x": x0}
            vars_dict = {"y": y_scaled}
            ctx = make_ctx(params, vars_dict)
            pieces = [jnp.ravel(entry.residual(ctx)) for entry in ineq_constraints]
            return (
                jnp.concatenate(pieces, axis=0)
                if pieces
                else jnp.zeros((0,), dtype=dtype)
            )

        try:
            r0 = np.asarray(residual_y(y0), dtype=np.float64)
            J = np.asarray(jax.jacobian(residual_y)(y0), dtype=np.float64)

            # residual form is A*y - rhs <= 0.
            # At y=0, rhs scale is about abs(r0).
            for j in range(n_y):
                coeff = np.abs(J[:, j])
                valid = coeff > self.scaling.eps
                if np.any(valid):
                    candidate = np.max(np.abs(r0[valid]) / coeff[valid])
                    scales[j] = max(scales[j], candidate)
        except Exception:
            pass

    if np.all(scales <= self.scaling.eps):
        return None

    return self._positive_scale(scales, size=n_y, name="variable scale")


def _estimate_variable_scale_from_sensitivities(
    self,
    *,
    objective_fun,
    eq_constraints,
    eq_blocks,
    ineq_constraints,
    ineq_blocks,
    make_ctx,
    dtype,
    X_original: np.ndarray,
    D_v_base: np.ndarray,
) -> np.ndarray:
    """Estimate variable normalization from objective and constraint sensitivities."""
    n_y = len(self.variable_names)
    sample_count = min(int(self.scaling.probe_points), int(X_original.shape[0]))

    X_probe_original = np.asarray(X_original[:sample_count], dtype=np.float64)
    X_probe_scaled = self._scaled_parameters(X_probe_original)

    D_v_tmp = np.where(D_v_base > self.scaling.eps, D_v_base, 1.0)

    grad_values = []

    def residual_sum(y_scaled, x_scaled):
        params = {"x": x_scaled}
        vars_dict = {"y": y_scaled}
        ctx = make_ctx(params, vars_dict)

        val = objective_fun(params, vars_dict)

        pieces = []
        for entry in eq_constraints:
            pieces.append(jnp.ravel(entry.residual(ctx)))
        for entry in ineq_constraints:
            pieces.append(jnp.ravel(entry.residual(ctx)))
        for entry in eq_blocks:
            pieces.append(jnp.ravel(entry.fn(params, vars_dict)))
        for entry in ineq_blocks:
            pieces.append(jnp.ravel(entry.fn(params, vars_dict)))

        if pieces:
            res = jnp.concatenate(pieces, axis=0)
            val = val + jnp.sum(res**2)

        return val

    grad_fn = jax.grad(residual_sum, argnums=0)

    for i in range(sample_count):
        x_s = jnp.asarray(X_probe_scaled[i], dtype=dtype)
        y_s = jnp.zeros((n_y,), dtype=dtype)

        try:
            g = np.asarray(grad_fn(y_s, x_s), dtype=np.float64).reshape(-1)
            grad_values.append(np.abs(g))
        except Exception:
            pass

    if not grad_values:
        return D_v_tmp

    G = np.stack(grad_values, axis=0)
    sensitivity = np.percentile(G, float(self.scaling.constraint_percentile), axis=0)

    sensitivity = np.where(np.isfinite(sensitivity), sensitivity, 0.0)
    sensitivity = np.where(sensitivity > self.scaling.eps, sensitivity, 1.0)

    # Only fill variables whose scale was not determined by finite bounds.
    missing = D_v_base <= self.scaling.eps
    D_v_new = D_v_tmp.copy()
    D_v_new[missing] = sensitivity[missing]

    return self._positive_scale(D_v_new, size=n_y, name="variable scale")


def _compute_basic_scales(
    self, *, X_original: np.ndarray, lower_M, lower_c, upper_M, upper_c
) -> None:
    """Choose baseline parameter and variable scales for the nonlinear formulation."""
    n_x = len(self.parameter_names)
    n_y = len(self.variable_names)

    if not self.scaling.enabled:
        self.scaling.D_p = np.ones((n_x,), dtype=np.float64)
        self.scaling.D_v = np.ones((n_y,), dtype=np.float64)
        self.scaling.D_obj = 1.0
        self.scaling.D_eq = None
        self.scaling.D_ineq = None
        return

    if self.scaling.manual_D_p is not None:
        self.scaling.D_p = self._positive_scale(
            self.scaling.manual_D_p, size=n_x, name="parameter scale"
        )
    else:
        self.scaling.D_p = self._positive_scale(
            np.max(np.abs(X_original), axis=0), size=n_x, name="parameter scale"
        )

    if self.scaling.manual_D_v is not None:
        self.scaling.D_v = self._positive_scale(
            self.scaling.manual_D_v, size=n_y, name="variable scale"
        )
    else:
        L, U = self._eval_affine_bounds_numpy(
            X_original,
            lower_M,
            lower_c,
            upper_M,
            upper_c,
        )

        finite_L = np.where(np.isfinite(L), np.abs(L), 0.0)
        finite_U = np.where(np.isfinite(U), np.abs(U), 0.0)

        D_v = np.maximum(
            np.max(finite_L, axis=0),
            np.max(finite_U, axis=0),
        )

        # Do not finalize missing entries here.
        # Missing entries will be filled later using sensitivities.
        self.scaling.D_v = self._positive_scale(
            np.where(D_v > self.scaling.eps, D_v, 1.0),
            size=n_y,
            name="variable scale",
        )

        self.scaling._D_v_from_bounds_raw = D_v


def _estimate_linear_objective_scale(self) -> float | None:
    if self._objective_expr is None:
        return None

    n_x = len(self.parameter_names)
    n_y = len(self.variable_names)

    y0 = jnp.zeros((n_y,), dtype=jnp.float64)
    x0 = jnp.zeros((n_x,), dtype=jnp.float64)

    variable_index = {name: idx for idx, name in enumerate(self.variable_names)}
    parameter_index = {name: idx for idx, name in enumerate(self.parameter_names)}

    def obj_y(y):
        ctx = _EvalContext(
            y=y,
            x=x0,
            variable_index=variable_index,
            parameter_index=parameter_index,
        )
        return self._objective_expr.eval(ctx)

    try:
        c = np.asarray(jax.grad(obj_y)(y0), dtype=np.float64).reshape(-1)
    except Exception:
        return None

    if c.shape != (n_y,) or not np.all(np.isfinite(c)):
        return None

    if self.scaling.D_v is None:
        D_v = np.ones((n_y,), dtype=np.float64)
    else:
        D_v = np.asarray(self.scaling.D_v, dtype=np.float64).reshape(-1)

    scale = float(np.max(np.abs(c * D_v)))

    if not np.isfinite(scale) or scale <= self.scaling.eps:
        return None

    return scale


def _estimate_objective_and_constraint_scales(
    self,
    *,
    objective_fun,
    eq_constraints,
    eq_blocks,
    ineq_constraints,
    ineq_blocks,
    lower_M,
    lower_c,
    upper_M,
    upper_c,
    make_ctx,
    dtype,
    X_original: np.ndarray,
) -> None:
    """Estimate objective and residual scales from sampled derivative magnitudes."""
    if not self.scaling.enabled:
        return

    n_y = len(self.variable_names)
    sample_count = min(int(self.scaling.probe_points), int(X_original.shape[0]))

    X_probe_original = np.asarray(X_original[:sample_count], dtype=np.float64)
    X_probe_scaled = self._scaled_parameters(X_probe_original)

    rng = np.random.default_rng(int(self.config.get("seed", 42)))

    if self.scaling.use_jacobian_scaling:
        y_probes = [
            np.zeros((n_y,), dtype=np.float64),
            0.5 * np.ones((n_y,), dtype=np.float64),
            np.ones((n_y,), dtype=np.float64),
            rng.uniform(-1.0, 1.0, size=(n_y,)),
        ]
    else:
        y_probes = [
            np.zeros((n_y,), dtype=np.float64),
        ]

    obj_values = []
    obj_grad_values = []

    eq_values = []
    eq_jac_values = []

    ineq_values = []
    ineq_jac_values = []

    def eval_eq(y_scaled, x_scaled):
        params = {"x": x_scaled}
        vars_dict = {"y": y_scaled}
        ctx = make_ctx(params, vars_dict)

        pieces = []
        for entry in eq_constraints:
            pieces.append(jnp.ravel(entry.residual(ctx)))
        for entry in eq_blocks:
            pieces.append(jnp.ravel(entry.fn(params, vars_dict)))

        return (
            jnp.concatenate(pieces, axis=0) if pieces else jnp.zeros((0,), dtype=dtype)
        )

    def eval_ineq(y_scaled, x_scaled):
        params = {"x": x_scaled}
        vars_dict = {"y": y_scaled}
        ctx = make_ctx(params, vars_dict)

        pieces = []
        for entry in ineq_constraints:
            pieces.append(jnp.ravel(entry.residual(ctx)))
        for entry in ineq_blocks:
            pieces.append(jnp.ravel(entry.fn(params, vars_dict)))

        return (
            jnp.concatenate(pieces, axis=0) if pieces else jnp.zeros((0,), dtype=dtype)
        )

    def eval_obj(y_scaled, x_scaled):
        return objective_fun({"x": x_scaled}, {"y": y_scaled})

    use_jac = self.problem_type in {"nlp", "nonconvex"} and bool(
        self.scaling.use_jacobian_scaling
    )

    if use_jac:
        grad_obj = jax.grad(eval_obj, argnums=0)
        jac_eq = jax.jacobian(eval_eq, argnums=0)
        jac_ineq = jax.jacobian(eval_ineq, argnums=0)
    else:
        grad_obj = None
        jac_eq = None
        jac_ineq = None

    _collect_scaling_probes(
        X_probe_scaled,
        dtype,
        eq_jac_values,
        eq_values,
        eval_eq,
        eval_ineq,
        eval_obj,
        grad_obj,
        ineq_jac_values,
        ineq_values,
        jac_eq,
        jac_ineq,
        obj_grad_values,
        obj_values,
        sample_count,
        y_probes,
    )

    _p_obj = float(self.scaling.objective_percentile)
    p_con = float(self.scaling.constraint_percentile)

    _assign_objective_scale(self, obj_grad_values, obj_values)

    _assign_equality_scales(self, eq_jac_values, eq_values, p_con)

    _assign_inequality_scales(self, ineq_jac_values, ineq_values, p_con)


def _assign_inequality_scales(self, ineq_jac_values, ineq_values, p_con):
    """Choose inequality row scales using the existing magnitude/Jacobian policy."""
    if self.scaling.manual_D_ineq is not None:
        ineq_size = None
        if ineq_values:
            ineq_size = int(np.stack(ineq_values, axis=0).shape[1])
        self.scaling.D_ineq = self._positive_scale(
            self.scaling.manual_D_ineq,
            size=ineq_size,
            name="inequality scale",
        )
    elif ineq_values or ineq_jac_values:
        candidates = []

        if ineq_values:
            G = np.stack(ineq_values, axis=0)
            candidates.append(np.percentile(G, p_con, axis=0))

        if ineq_jac_values:
            JG = np.stack(ineq_jac_values, axis=0)
            candidates.append(np.percentile(JG, p_con, axis=0))

        D_ineq = np.maximum.reduce(candidates)
        self.scaling.D_ineq = self._positive_scale(
            D_ineq,
            size=D_ineq.shape[0],
            name="inequality scale",
        )
    else:
        self.scaling.D_ineq = None


def _assign_equality_scales(self, eq_jac_values, eq_values, p_con):
    """Choose equality row scales using the existing magnitude/Jacobian policy."""
    if self.scaling.manual_D_eq is not None:
        eq_size = None
        if eq_values:
            eq_size = int(np.stack(eq_values, axis=0).shape[1])
        self.scaling.D_eq = self._positive_scale(
            self.scaling.manual_D_eq,
            size=eq_size,
            name="equality scale",
        )
    elif eq_values or eq_jac_values:
        candidates = []

        if eq_values:
            H = np.stack(eq_values, axis=0)
            candidates.append(np.percentile(H, p_con, axis=0))

        if eq_jac_values:
            JH = np.stack(eq_jac_values, axis=0)
            candidates.append(np.percentile(JH, p_con, axis=0))

        D_eq = np.maximum.reduce(candidates)
        self.scaling.D_eq = self._positive_scale(
            D_eq,
            size=D_eq.shape[0],
            name="equality scale",
        )
    else:
        self.scaling.D_eq = None


def _assign_objective_scale(self, obj_grad_values, obj_values):
    """Choose the objective scale using the existing manual or percentile policy."""
    if self.scaling.manual_D_obj is not None:
        self.scaling.D_obj = self._positive_scalar_scale(
            self.scaling.manual_D_obj,
            name="objective scale",
        )
    else:
        candidates = []

        linear_obj_scale = self._estimate_linear_objective_scale()
        if linear_obj_scale is not None:
            candidates.append(linear_obj_scale)

        if obj_values:
            candidates.append(
                np.percentile(
                    np.asarray(obj_values, dtype=np.float64),
                    float(self.scaling.objective_percentile),
                )
            )

        if obj_grad_values:
            candidates.append(
                np.percentile(
                    np.asarray(obj_grad_values, dtype=np.float64),
                    float(self.scaling.objective_percentile),
                )
            )

        self.scaling.D_obj = self._positive_scalar_scale(
            max(candidates) if candidates else 1.0,
            name="objective scale",
        )


def _collect_scaling_probes(
    X_probe_scaled,
    dtype,
    eq_jac_values,
    eq_values,
    eval_eq,
    eval_ineq,
    eval_obj,
    grad_obj,
    ineq_jac_values,
    ineq_values,
    jac_eq,
    jac_ineq,
    obj_grad_values,
    obj_values,
    sample_count,
    y_probes,
):
    """Evaluate the existing objective/residual probes without changing fallback rules."""
    for i in range(sample_count):
        x_s = jnp.asarray(X_probe_scaled[i], dtype=dtype)

        for y_np in y_probes:
            y_s = jnp.asarray(y_np, dtype=dtype)

            try:
                val = float(np.asarray(eval_obj(y_s, x_s)))
                if np.isfinite(val):
                    obj_values.append(abs(val))
            except Exception:
                pass

            if grad_obj is not None:
                try:
                    g = np.asarray(grad_obj(y_s, x_s), dtype=np.float64).reshape(-1)
                    if np.all(np.isfinite(g)):
                        obj_grad_values.append(np.linalg.norm(g, ord=np.inf))
                except Exception:
                    pass

            try:
                h = np.asarray(eval_eq(y_s, x_s), dtype=np.float64).reshape(-1)
                if h.size > 0 and np.all(np.isfinite(h)):
                    eq_values.append(np.abs(h))
            except Exception:
                pass

            if jac_eq is not None:
                try:
                    Jh = np.asarray(jac_eq(y_s, x_s), dtype=np.float64)
                    if Jh.size > 0 and np.all(np.isfinite(Jh)):
                        eq_jac_values.append(np.max(np.abs(Jh), axis=1))
                except Exception:
                    pass

            try:
                gval = np.asarray(eval_ineq(y_s, x_s), dtype=np.float64).reshape(-1)
                if gval.size > 0 and np.all(np.isfinite(gval)):
                    ineq_values.append(np.abs(gval))
            except Exception:
                pass

            if jac_ineq is not None:
                try:
                    Jg = np.asarray(jac_ineq(y_s, x_s), dtype=np.float64)
                    if Jg.size > 0 and np.all(np.isfinite(Jg)):
                        ineq_jac_values.append(np.max(np.abs(Jg), axis=1))
                except Exception:
                    pass
