from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
import numpy as np

from jaxmodel import JaxNLPModel

from .CP_jax import CP_fixed


@dataclass(frozen=True)
class CPSolveResult:
    y: np.ndarray
    lam: np.ndarray
    mu: np.ndarray
    objective: float


def _default_initial_point(model: JaxNLPModel, params) -> np.ndarray:
    lb = model.lower_bounds(params)
    ub = model.upper_bounds(params)
    n = model.var_spec.total_size

    if lb is None and ub is None:
        return np.zeros((n,), dtype=np.float64)

    if lb is None:
        lb = -jnp.inf * jnp.ones((n,), dtype=model.dtype)
    if ub is None:
        ub = jnp.inf * jnp.ones((n,), dtype=model.dtype)

    lb_np = np.asarray(lb, dtype=np.float64)
    ub_np = np.asarray(ub, dtype=np.float64)
    return np.where(
        np.isfinite(lb_np) & np.isfinite(ub_np),
        0.5 * (lb_np + ub_np),
        np.where(np.isfinite(lb_np), lb_np + 1.0, np.where(np.isfinite(ub_np), ub_np - 1.0, 0.0)),
    )


def extract_sqp_subproblem(
    model: JaxNLPModel,
    params,
    *,
    y: Optional[np.ndarray] = None,
    rho: float = 1.0,
    use_diagonal_hessian: bool = True,
    diag_floor: float = 1e-8,
):
    y_ref = _default_initial_point(model, params) if y is None else np.asarray(y, dtype=np.float64).reshape((-1,))
    return model.sqp_subproblem_data(
        params,
        jnp.asarray(y_ref, dtype=model.dtype),
        rho=rho,
        use_diagonal_hessian=use_diagonal_hessian,
        diag_floor=diag_floor,
    )


def solve_sqp_subproblem_with_cp(
    model: JaxNLPModel,
    params,
    *,
    y: Optional[np.ndarray] = None,
    rho: float = 1.0,
    use_diagonal_hessian: bool = True,
    diag_floor: float = 1e-8,
    max_iter: int = 5000,
    tol: float = 1e-8,
    use_ruiz: bool = True,
    ruiz_iters: int = 4,
):
    sqp = extract_sqp_subproblem(
        model,
        params,
        y=y,
        rho=rho,
        use_diagonal_hessian=use_diagonal_hessian,
        diag_floor=diag_floor,
    )

    y0 = _default_initial_point(model, params) if y is None else np.asarray(y, dtype=np.float64).reshape((-1,))
    me = sqp.constraints.A.shape[0]
    mi = sqp.constraints.C.shape[0]

    y_sol, lam_sol, mu_sol = CP_fixed(
        np.asarray(sqp.objective.Q_diag, dtype=np.float64)[None, :],
        np.asarray(sqp.objective.c, dtype=np.float64)[None, :],
        np.asarray(sqp.constraints.A, dtype=np.float64)[None, :, :],
        np.asarray(sqp.constraints.b, dtype=np.float64)[None, :],
        np.asarray(sqp.constraints.C, dtype=np.float64)[None, :, :],
        np.asarray(sqp.constraints.d, dtype=np.float64)[None, :],
        np.asarray(sqp.l if sqp.l is not None else -jnp.inf * jnp.ones_like(sqp.objective.c), dtype=np.float64)[None, :],
        np.asarray(sqp.u if sqp.u is not None else jnp.inf * jnp.ones_like(sqp.objective.c), dtype=np.float64)[None, :],
        max_iter=max_iter,
        tol=tol,
        y0=y0[None, :],
        lam0=np.zeros((1, me), dtype=np.float64),
        mu0=np.zeros((1, mi), dtype=np.float64),
        use_ruiz=use_ruiz,
        ruiz_iters=ruiz_iters,
    )

    y_out = np.asarray(y_sol[0], dtype=np.float64)
    return CPSolveResult(
        y=y_out,
        lam=np.asarray(lam_sol[0], dtype=np.float64),
        mu=np.asarray(mu_sol[0], dtype=np.float64),
        objective=float(model.objective_value(params, jnp.asarray(y_out, dtype=model.dtype))),
    )


extract_projection_subproblem = extract_sqp_subproblem
solve_projection_subproblem_with_cp = solve_sqp_subproblem_with_cp
