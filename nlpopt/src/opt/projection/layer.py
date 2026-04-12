from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import jax.numpy as jnp

from ..CP_jax import ruiz_equilibrate
from ..CP_jax_implicit import CP_accelerated_implicit_jit, CP_fixed_implicit_jit

Array = jnp.ndarray

if TYPE_CHECKING:
    from ..training.config import TrainConfig


def solve_cp_implicit(
    Q_diag: Array,
    c: Array,
    A: Array,
    b: Array,
    C: Array,
    d: Array,
    l: Array,
    u: Array,
    *,
    cfg: TrainConfig,
    y0: Optional[Array] = None,
    lam0: Optional[Array] = None,
    mu0: Optional[Array] = None,
):
    Q_diag = jnp.asarray(Q_diag)
    c = jnp.asarray(c)
    A = jnp.asarray(A)
    b = jnp.asarray(b)
    C = jnp.asarray(C)
    d = jnp.asarray(d)
    l = jnp.asarray(l)
    u = jnp.asarray(u)

    B, n = Q_diag.shape
    me = A.shape[1] if A.ndim == 3 else A.shape[0]
    mi = C.shape[1] if C.ndim == 3 else C.shape[0]

    dtype = jnp.result_type(Q_diag.dtype, c.dtype, A.dtype, b.dtype, C.dtype, d.dtype, l.dtype, u.dtype)
    Q_diag = Q_diag.astype(dtype)
    c = c.astype(dtype)
    A = A.astype(dtype)
    b = b.astype(dtype)
    C = C.astype(dtype)
    d = d.astype(dtype)
    l = l.astype(dtype)
    u = u.astype(dtype)

    if A.ndim == 2:
        A = jnp.broadcast_to(A[None, :, :], (B, A.shape[0], A.shape[1]))
    if C.ndim == 2:
        C = jnp.broadcast_to(C[None, :, :], (B, C.shape[0], C.shape[1]))
    if b.ndim == 1:
        b = jnp.broadcast_to(b[None, :], (B, b.shape[0]))
    if d.ndim == 1:
        d = jnp.broadcast_to(d[None, :], (B, d.shape[0]))
    if l.ndim == 1:
        l = jnp.broadcast_to(l[None, :], (B, l.shape[0]))
    if u.ndim == 1:
        u = jnp.broadcast_to(u[None, :], (B, u.shape[0]))

    if y0 is None:
        y0 = jnp.zeros((B, n), dtype=dtype)
    else:
        y0 = jnp.asarray(y0, dtype=dtype)
    if lam0 is None:
        lam0 = jnp.zeros((B, me), dtype=dtype)
    else:
        lam0 = jnp.asarray(lam0, dtype=dtype)
    if mu0 is None:
        mu0 = jnp.zeros((B, mi), dtype=dtype)
    else:
        mu0 = jnp.asarray(mu0, dtype=dtype)

    q_s = Q_diag
    c_s = c
    A_s = A
    b_s = b
    C_s = C
    d_s = d
    l_s = l
    u_s = u
    col_scale = jnp.ones((B, n), dtype=dtype)
    eq_scale = jnp.ones((B, me), dtype=dtype)
    ineq_scale = jnp.ones((B, mi), dtype=dtype)

    if cfg.use_ruiz:
        q_s, c_s, A_s, b_s, C_s, d_s, l_s, u_s, col_scale, eq_scale, ineq_scale = ruiz_equilibrate(
            Q_diag,
            c,
            A,
            b,
            C,
            d,
            l,
            u,
            iterations=cfg.ruiz_iters,
        )
        y0 = y0 / col_scale
        lam0 = lam0 / jnp.where(eq_scale == 0.0, 1.0, eq_scale)
        mu0 = mu0 / jnp.where(ineq_scale == 0.0, 1.0, ineq_scale)

    solver = CP_fixed_implicit_jit if cfg.IS_FIXED else CP_accelerated_implicit_jit
    y_s, lam_s, mu_s = solver(
        q_s,
        c_s,
        A_s,
        b_s,
        C_s,
        d_s,
        l_s,
        u_s,
        safety=cfg.safety,
        knorm_iters=cfg.knorm_iters,
        knorm_seed=cfg.knorm_seed,
        max_iter=cfg.cp_iters,
        tol=cfg.cp_tol,
        y0=y0,
        lam0=lam0,
        mu0=mu0,
        adjoint_iters=cfg.adjoint_iters,
    )

    if cfg.use_ruiz:
        y_s = col_scale * y_s
        lam_s = eq_scale * lam_s
        mu_s = ineq_scale * mu_s

    return y_s, lam_s, mu_s
