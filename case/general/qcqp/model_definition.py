from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from jaxmodel import HighLevelNLPBuilder

PARAM_NAME = "x"

N_X = 10
N_Y = 10
N_EQ = 2
N_INEQ = 5

SEED = 43
BOUND_RADIUS = 2.0
IS_DIAG_Q = False

X_L = jnp.full((N_X,), -1.0, dtype=jnp.float64)
X_U = jnp.full((N_X,), 1.0, dtype=jnp.float64)


def _x_abs_max(dtype):
    return jnp.maximum(jnp.abs(jnp.asarray(X_L, dtype=dtype)), jnp.abs(jnp.asarray(X_U, dtype=dtype)))


def _make_objective_matrix(rng: np.random.Generator, *, dtype) -> jnp.ndarray:
    if IS_DIAG_Q:
        diag = 1.0 + rng.uniform(0.2, 1.0, size=(N_Y,))
        return jnp.diag(jnp.asarray(diag, dtype=dtype))

    mat = rng.normal(size=(N_Y, N_Y))
    spd = (mat.T @ mat) / max(1, N_Y)
    spd = spd + np.diag(1.0 + rng.uniform(0.2, 0.8, size=(N_Y,)))
    return jnp.asarray(spd, dtype=dtype)


def _make_objective_vector(rng: np.random.Generator, *, dtype) -> jnp.ndarray:
    return jnp.asarray(rng.uniform(-0.1, 0.1, size=(N_Y,)), dtype=dtype)


def _make_equality_block(rng: np.random.Generator, *, dtype):
    A = np.zeros((N_EQ, N_Y), dtype=np.float64)
    A[:, :N_EQ] = np.eye(N_EQ, dtype=np.float64)
    B = 0.08 * rng.normal(size=(N_EQ, N_X)) / max(1, N_X)
    b = np.zeros((N_EQ,), dtype=np.float64)
    return (
        jnp.asarray(A, dtype=dtype),
        jnp.asarray(B, dtype=dtype),
        jnp.asarray(b, dtype=dtype),
    )


def _make_qcqp_quadratic_block(
    rng: np.random.Generator,
    *,
    Q: jnp.ndarray,
    c: jnp.ndarray,
    B: jnp.ndarray,
    x_abs_max: jnp.ndarray,
    dtype,
):
    q_mats = np.zeros((N_INEQ, N_Y, N_Y), dtype=np.float64)
    c_vecs = np.zeros((N_INEQ, N_Y), dtype=np.float64)
    rhs = np.zeros((N_INEQ,), dtype=np.float64)
    E = np.zeros((N_INEQ, N_X), dtype=np.float64)

    free_start = N_EQ if N_Y > N_EQ else 0
    free_dim = N_Y - free_start
    if free_dim <= 0:
        return (
            jnp.asarray(q_mats, dtype=dtype),
            jnp.asarray(c_vecs, dtype=dtype),
            jnp.asarray(rhs, dtype=dtype),
            jnp.asarray(E, dtype=dtype),
        )

    Q_np = np.asarray(Q, dtype=np.float64)
    c_np = np.asarray(c, dtype=np.float64).reshape((N_Y,))
    B_np = np.asarray(B, dtype=np.float64).reshape((N_EQ, N_X))
    x_abs_np = np.asarray(x_abs_max, dtype=np.float64).reshape((N_X,))

    Q_ff = Q_np[free_start:, free_start:]
    c_f = c_np[free_start:]
    eye = 1e-10 * np.eye(free_dim, dtype=np.float64)
    center = -np.linalg.solve(Q_ff + eye, c_f)
    Q_fe = Q_np[free_start:, :N_EQ]
    trend = -np.linalg.solve(Q_ff + eye, Q_fe @ B_np)

    for idx in range(N_INEQ):
        local_idx = idx % free_dim
        var_idx = free_start + local_idx
        max_abs = abs(float(center[local_idx])) + float(np.abs(trend[local_idx]) @ x_abs_np)
        radius = max(0.02, 0.3 * max_abs)
        q_mats[idx, var_idx, var_idx] = 2.0
        rhs[idx] = float(radius**2)

    return (
        jnp.asarray(q_mats, dtype=dtype),
        jnp.asarray(c_vecs, dtype=dtype),
        jnp.asarray(rhs, dtype=dtype),
        jnp.asarray(E, dtype=dtype),
    )


def build_model(*, dtype=jnp.float64):
    rng = np.random.default_rng(SEED)
    Q = _make_objective_matrix(rng, dtype=dtype)
    c = _make_objective_vector(rng, dtype=dtype)
    A, B, b = _make_equality_block(rng, dtype=dtype)
    quad_Q, quad_c, quad_rhs, quad_x_coeff = _make_qcqp_quadratic_block(
        rng,
        Q=Q,
        c=c,
        B=B,
        x_abs_max=_x_abs_max(dtype),
        dtype=dtype,
    )

    params0 = {PARAM_NAME: jnp.zeros((N_X,), dtype=dtype)}
    zeros_bounds = jnp.zeros((N_Y, N_X), dtype=dtype)
    lower_bounds = -BOUND_RADIUS * jnp.ones((N_Y,), dtype=dtype)
    upper_bounds = BOUND_RADIUS * jnp.ones((N_Y,), dtype=dtype)

    builder = (
        HighLevelNLPBuilder(dtype=dtype)
        .add_parameter(PARAM_NAME, N_X)
        .add_variable("y", N_Y)
        .set_quadratic_objective(Q=Q, c=c)
        .add_affine_equality(
            var_name="y",
            A=A,
            rhs_const=b,
            param_terms=[(B, PARAM_NAME)],
            name="eq_block",
        )
        .set_affine_lower_bound(
            var_name="y",
            param_name=PARAM_NAME,
            M=zeros_bounds,
            c=lower_bounds,
        )
        .set_affine_upper_bound(
            var_name="y",
            param_name=PARAM_NAME,
            M=zeros_bounds,
            c=upper_bounds,
        )
    )

    for idx in range(N_INEQ):
        builder = builder.add_quadratic_inequality(
            Q=quad_Q[idx],
            c=quad_c[idx],
            rhs_const=float(quad_rhs[idx]),
            x_coeff=quad_x_coeff[idx],
            x_name=PARAM_NAME,
            name=f"qc_{idx}",
        )

    return builder.build(example_params=params0, jit_compile=True)
