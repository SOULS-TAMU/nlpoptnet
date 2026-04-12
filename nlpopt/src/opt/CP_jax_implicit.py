import jax
import jax.numpy as jnp
from jax import tree_util
from jax.flatten_util import ravel_pytree
import jax.scipy.sparse.linalg as jsp

from .CP_jax import CP_accelerated, CP_fixed, estimate_AC_norm

jax.config.update("jax_enable_x64", True)

# ============================================================
# One CP step map (FIXED stepsizes): z -> F(z; params)
# z = (y, lam, mu, ybar)
# params = (Q_diag, c, A, b, C, d, l, u)
# steps = (tau, sigma, theta, P) treated as constants in backward
# ============================================================
def _cp_step_fixed(z, params, steps):
    y, lam, mu, ybar = z
    Q_diag, c, A, b, C, d, l, u = params
    tau, sigma, theta, P = steps

    B, n = Q_diag.shape
    me = A.shape[1]
    mi = C.shape[1]

    # dual updates
    lam_hat = lam
    if me != 0:
        Aybar = jnp.einsum("bmn,bn->bm", A, ybar)
        lam_hat = lam + sigma[:, None] * (Aybar - b)

    mu_hat = mu
    if mi != 0:
        Cybar = jnp.einsum("bmn,bn->bm", C, ybar)
        mu_hat = mu + sigma[:, None] * (Cybar - d)
        mu_hat = jnp.maximum(mu_hat, 0.0)

    # primal gradient
    grad = jnp.zeros((B, n), dtype=y.dtype)
    if me != 0:
        grad = grad + jnp.einsum("bmn,bm->bn", A, lam_hat)
    if mi != 0:
        grad = grad + jnp.einsum("bmn,bm->bn", C, mu_hat)

    # primal prox (diag Q) + box projection
    v = y - tau[:, None] * grad
    y_tilde = P * (v - tau[:, None] * c)
    y_new = jnp.minimum(jnp.maximum(y_tilde, l), u)

    # extrapolation
    ybar_new = y_new + theta[:, None] * (y_new - y)

    return (y_new, lam_hat, mu_hat, ybar_new)


# ============================================================
# Build frozen stepsizes for CP_fixed (same as your CP_fixed)
# NOTE: this is for backward linearization; tau/sigma/theta treated constant.
# ============================================================
def _steps_fixed(Q_diag, A, C, *, safety, knorm_iters, knorm_seed):
    Q_diag = jnp.asarray(Q_diag)
    A = jnp.asarray(A)
    C = jnp.asarray(C)

    B, n = Q_diag.shape
    dtype = jnp.result_type(Q_diag.dtype, A.dtype, C.dtype)
    Q_diag = Q_diag.astype(dtype)
    A = A.astype(dtype)
    C = C.astype(dtype)

    # Lf per batch (max diagonal of Q)
    Lf = jnp.max(Q_diag, axis=1) if n != 0 else jnp.zeros((B,), dtype=dtype)

    # ALWAYS estimate ||[A;C]|| using the AC estimator (works even if me/mi are 0)
    key = jax.random.PRNGKey(knorm_seed)
    K_norm = estimate_AC_norm(A, C, key, iters=knorm_iters)

    L = jnp.maximum(K_norm, 1e-12)
    s = jnp.asarray(safety, dtype=dtype)

    sigma = s / L
    tau   = s / (Lf + L)
    theta = jnp.ones((B,), dtype=dtype)
    P     = 1.0 / (1.0 + tau[:, None] * Q_diag)

    # freeze them for implicit backward
    return (jax.lax.stop_gradient(tau),
            jax.lax.stop_gradient(sigma),
            jax.lax.stop_gradient(theta),
            jax.lax.stop_gradient(P))


# ============================================================
# For accelerated forward: we still do implicit backward w.r.t.
# a stationary CP step map with FROZEN stepsizes.
# (We can reuse the same _steps_fixed or you can use a different init rule.)
# ============================================================
def _steps_for_accel_backward(Q_diag, A, C, *, safety, knorm_iters, knorm_seed):
    # simplest + stable: same as fixed rule
    return _steps_fixed(Q_diag, A, C, safety=safety, knorm_iters=knorm_iters, knorm_seed=knorm_seed)


# ============================================================
# CP_fixed implicit layer (custom_vjp)
# grads returned for: Q_diag,c,A,b,C,d,l,u  (NOT for y0/lam0/mu0)
# ============================================================
@jax.custom_vjp
def CP_fixed_implicit(
    Q_diag, c, A, b, C, d, l, u,
    safety=0.95, knorm_iters=20, knorm_seed=42, max_iter=5000, tol=1e-9,
    y0=None, lam0=None, mu0=None,
    adjoint_iters: int = 50,
):
    if (y0 is None) or (lam0 is None) or (mu0 is None):
        raise ValueError(
            "Pass warm-start arrays y0/lam0/mu0 (no None). "
            "Use opt.projection.solve_cp_implicit to canonicalize them."
        )
    return CP_fixed(Q_diag, c, A, b, C, d, l, u,
                    safety=safety, knorm_iters=knorm_iters, knorm_seed=knorm_seed,
                    max_iter=max_iter, tol=tol, y0=y0, lam0=lam0, mu0=mu0)


# -----------------------------
# FIXED implicit: fwd/bwd (do NOT store pullback in residuals)
# -----------------------------
def _CP_fixed_implicit_fwd(Q_diag, c, A, b, C, d, l, u,
                            safety, knorm_iters, knorm_seed, max_iter, tol,
                            y0=None, lam0=None, mu0=None, adjoint_iters: int = 50):
    # forward solve (your CP)
    y, lam, mu = CP_fixed(Q_diag, c, A, b, C, d, l, u,
                          safety=safety, knorm_iters=knorm_iters, knorm_seed=knorm_seed,
                          max_iter=max_iter, tol=tol, y0=y0, lam0=lam0, mu0=mu0)

    # frozen steps for backward linearization
    steps = _steps_fixed(Q_diag, A, C, safety=safety, knorm_iters=knorm_iters, knorm_seed=knorm_seed)

    # fixed-point state (ybar ~= y at convergence)
    zstar = (y, lam, mu, y)

    # params (stop grad to keep backward linearization “frozen”)
    params = (jnp.asarray(Q_diag), jnp.asarray(c), jnp.asarray(A), jnp.asarray(b),
              jnp.asarray(C), jnp.asarray(d), jnp.asarray(l), jnp.asarray(u))

    zstar = tree_util.tree_map(jax.lax.stop_gradient, zstar)
    params = tree_util.tree_map(jax.lax.stop_gradient, params)
    steps  = tree_util.tree_map(jax.lax.stop_gradient, steps)

    # residual must be a pytree of arrays (NO python callables)
    res = (zstar, params, steps, adjoint_iters)

    return (y, lam, mu), res


def _CP_fixed_implicit_bwd(res, cot):
    zstar, params, steps, adjoint_iters = res
    gy, glam, gmu = cot

    g_z = (gy, glam, gmu, jnp.zeros_like(gy))

    def step_joint(z, p):
        return _cp_step_fixed(z, p, steps)

    _, pullback = jax.vjp(step_joint, zstar, params)

    # ---- define linear operator A(v) = v - J^T v ----
    def Jt_apply(v):
        dz, _ = pullback(v)
        return dz

    # We need GMRES on flat vectors, so ravel/unravel the pytree
    g_flat, unravel = ravel_pytree(g_z)

    def A_mv(v_flat):
        v = unravel(v_flat)
        Jtv = Jt_apply(v)
        Jtv_flat, _ = ravel_pytree(Jtv)
        return v_flat - Jtv_flat

    # ---- GMRES solve: (I - J^T) v = g ----
    # You can tune restart/maxiter/tol. restart is important for memory.
    # v_flat, info = jsp.gmres(
    #     A_mv,
    #     g_flat,
    #     tol=1e-6,
    #     atol=0.0,
    #     restart=50,
    #     maxiter=adjoint_iters,
    # )

    v_flat, info = jsp.bicgstab(
        A_mv,
        g_flat,
        tol=1e-6,
        atol=0.0,
        maxiter=adjoint_iters,
    )

    # info=0 means converged; >0 means hit iteration limit; <0 breakdown
    v_z = unravel(v_flat)

    # parameter grads: v^T dF/dparams
    _, g_params = pullback(v_z)
    gQ, gc, gA, gb, gC, gd, gl, gu = g_params

    return (gQ, gc, gA, gb, gC, gd, gl, gu,
            None, None, None, None, None, None, None, None, None)

CP_fixed_implicit.defvjp(_CP_fixed_implicit_fwd, _CP_fixed_implicit_bwd)

CP_fixed_implicit_jit = jax.jit(
    CP_fixed_implicit,
    static_argnames=("safety", "knorm_iters", "knorm_seed", "max_iter", "tol", "adjoint_iters"),
)


# ============================================================
# CP_accelerated implicit layer (custom_vjp)
# Forward uses CP_accelerated (your accelerated solver).
# Backward uses the same implicit scheme with frozen stepsizes.
# ============================================================
@jax.custom_vjp
def CP_accelerated_implicit(
    Q_diag, c, A, b, C, d, l, u,
    safety=0.95, knorm_iters=20, knorm_seed=42, max_iter=5000, tol=1e-9,
    y0=None, lam0=None, mu0=None,
    adjoint_iters: int = 50,
):
    if (y0 is None) or (lam0 is None) or (mu0 is None):
        raise ValueError(
            "Pass warm-start arrays y0/lam0/mu0 (no None). "
            "Use opt.projection.solve_cp_implicit to canonicalize them."
        )
    # forward uses accelerated solver
    return CP_accelerated(Q_diag, c, A, b, C, d, l, u,
                          safety=safety, knorm_iters=knorm_iters, knorm_seed=knorm_seed,
                          max_iter=max_iter, tol=tol, y0=y0, lam0=lam0, mu0=mu0)


def _CP_accelerated_implicit_fwd(Q_diag, c, A, b, C, d, l, u,
                                safety, knorm_iters, knorm_seed, max_iter, tol,
                                y0, lam0, mu0, adjoint_iters):
    # ---- forward solve ----
    y, lam, mu = CP_accelerated(Q_diag, c, A, b, C, d, l, u,
                                safety=safety, knorm_iters=knorm_iters, knorm_seed=knorm_seed,
                                max_iter=max_iter, tol=tol, y0=y0, lam0=lam0, mu0=mu0)

    # ---- frozen steps for backward linearization ----
    # (use your chosen policy; here: same as fixed for stability)
    steps = _steps_for_accel_backward(Q_diag, A, C,
                                      safety=safety, knorm_iters=knorm_iters, knorm_seed=knorm_seed)

    # fixed-point state: at convergence ybar ~= y
    zstar = (y, lam, mu, y)

    # parameters for one-step map
    params = (jnp.asarray(Q_diag), jnp.asarray(c), jnp.asarray(A), jnp.asarray(b),
              jnp.asarray(C), jnp.asarray(d), jnp.asarray(l), jnp.asarray(u))

    # IMPORTANT: residual must be a pytree of ARRAYS only (no python callables)
    zstar = tree_util.tree_map(jax.lax.stop_gradient, zstar)
    params = tree_util.tree_map(jax.lax.stop_gradient, params)
    steps  = tree_util.tree_map(jax.lax.stop_gradient, steps)

    res = (zstar, params, steps, adjoint_iters)
    return (y, lam, mu), res


def _CP_accelerated_implicit_bwd(res, cot):
    zstar, params, steps, adjoint_iters = res
    gy, glam, gmu = cot

    # g in z-space (include ybar with 0)
    g_z = (gy, glam, gmu, jnp.zeros_like(gy))

    def step_joint(z, p):
        return _cp_step_fixed(z, p, steps)

    # VJP of one CP step at fixed point
    _, pullback = jax.vjp(step_joint, zstar, params)

    def Jt_apply(v):
        dz, _ = pullback(v)
        return dz  # = J_F^T v

    # Build linear operator A(v) = (I - J^T)v in flattened space
    g_flat, unravel = ravel_pytree(g_z)

    def A_mv(v_flat):
        v = unravel(v_flat)
        Jtv = Jt_apply(v)
        Jtv_flat, _ = ravel_pytree(Jtv)
        return v_flat - Jtv_flat

    # ---- Choose ONE solver ----
    # # 1) GMRES (recommended; restart controls memory)
    # v_flat, info = jsp.gmres(
    #     A_mv,
    #     g_flat,
    #     tol=1e-6,
    #     atol=0.0,
    #     restart=50,
    #     maxiter=adjoint_iters,
    # )

    # 2) BiCGSTAB (cheaper per iter, sometimes less robust)
    v_flat, info = jsp.bicgstab(
        A_mv,
        g_flat,
        tol=1e-6,
        atol=0.0,
        maxiter=adjoint_iters,
    )

    v_z = unravel(v_flat)

    # parameter grads: v^T dF/dparams
    _, g_params = pullback(v_z)
    gQ, gc, gA, gb, gC, gd, gl, gu = g_params

    # grads for primal args only, None for hyperparams/init
    return (gQ, gc, gA, gb, gC, gd, gl, gu,
            None, None, None, None, None, None, None, None, None)


CP_accelerated_implicit.defvjp(_CP_accelerated_implicit_fwd, _CP_accelerated_implicit_bwd)

CP_accelerated_implicit_jit = jax.jit(
    CP_accelerated_implicit,
    static_argnames=("safety", "knorm_iters", "knorm_seed", "max_iter", "tol", "adjoint_iters"),
)
