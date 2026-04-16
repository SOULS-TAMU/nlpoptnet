from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _init_x(key, batch_size, n, dtype):
    keys = jax.random.split(key, batch_size)
    x = jax.vmap(lambda k: jax.random.normal(k, (n,), dtype=dtype))(keys)
    return x / (jnp.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def canonicalize_constraints(Q_diag, A, b, C, d):
    Q_diag = jnp.asarray(Q_diag)
    B, n = Q_diag.shape
    dtype = Q_diag.dtype

    if A is None:
        A = jnp.zeros((B, 0, n), dtype=dtype)
        b = jnp.zeros((B, 0), dtype=dtype)
    else:
        A = jnp.asarray(A, dtype=dtype)
        b = jnp.asarray(b, dtype=dtype)

    if C is None:
        C = jnp.zeros((B, 0, n), dtype=dtype)
        d = jnp.zeros((B, 0), dtype=dtype)
    else:
        C = jnp.asarray(C, dtype=dtype)
        d = jnp.asarray(d, dtype=dtype)

    return A, b, C, d


def estimate_AC_norm(A, C, key, *, iters: int = 50):
    A = jnp.asarray(A)
    C = jnp.asarray(C)
    B = A.shape[0]
    n = A.shape[2]
    dtype = jnp.result_type(A.dtype, C.dtype)
    A = A.astype(dtype)
    C = C.astype(dtype)

    x = _init_x(key, B, n, dtype)

    def body(_, carry):
        x, lam = carry
        Ax = jnp.einsum("bmn,bn->bm", A, x)
        Cx = jnp.einsum("bmn,bn->bm", C, x)
        y = jnp.einsum("bmn,bm->bn", A, Ax) + jnp.einsum("bmn,bm->bn", C, Cx)
        lam = jnp.sum(x * y, axis=1)
        x = y / (jnp.linalg.norm(y, axis=1, keepdims=True) + 1e-12)
        return x, lam

    _, lam = jax.lax.fori_loop(0, iters, body, (x, jnp.zeros((B,), dtype=dtype)))
    return jnp.sqrt(jnp.maximum(lam, 0.0)) + 1e-12


def _safe_max_abs_rows(mat):
    if mat.shape[1] == 0:
        return jnp.zeros((mat.shape[0], 0), dtype=mat.dtype)
    return jnp.max(jnp.abs(mat), axis=2)


def _safe_max_abs_cols(mat):
    if mat.shape[1] == 0:
        return jnp.zeros((mat.shape[0], mat.shape[2]), dtype=mat.dtype)
    return jnp.max(jnp.abs(mat), axis=1)


def ruiz_equilibrate(
    Q_diag,
    c,
    A,
    b,
    C,
    d,
    l,
    u,
    *,
    iterations: int = 4,
    eps: float = 1e-6,
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
    me = A.shape[1]
    mi = C.shape[1]
    dtype = jnp.result_type(Q_diag.dtype, c.dtype, A.dtype, b.dtype, C.dtype, d.dtype, l.dtype, u.dtype)

    Qs = Q_diag.astype(dtype)
    cs = c.astype(dtype)
    As = A.astype(dtype)
    bs = b.astype(dtype)
    Cs = C.astype(dtype)
    ds = d.astype(dtype)
    ls = l.astype(dtype)
    us = u.astype(dtype)

    col_scale = jnp.ones((B, n), dtype=dtype)
    eq_scale = jnp.ones((B, me), dtype=dtype)
    ineq_scale = jnp.ones((B, mi), dtype=dtype)

    eps_arr = jnp.asarray(eps, dtype=dtype)

    def body(_, carry):
        Qs, cs, As, bs, Cs, ds, ls, us, col_scale, eq_scale, ineq_scale = carry

        col_metric = jnp.maximum(_safe_max_abs_cols(As), _safe_max_abs_cols(Cs))
        col_metric = jnp.maximum(col_metric, jnp.abs(Qs))
        col_update = 1.0 / jnp.sqrt(jnp.maximum(col_metric, eps_arr))

        As = As * col_update[:, None, :]
        Cs = Cs * col_update[:, None, :]
        Qs = Qs * (col_update ** 2)
        cs = cs * col_update
        ls = ls / col_update
        us = us / col_update
        col_scale = col_scale * col_update

        if me != 0:
            eq_metric = _safe_max_abs_rows(As)
            eq_update = 1.0 / jnp.sqrt(jnp.maximum(eq_metric, eps_arr))
            As = As * eq_update[:, :, None]
            bs = bs * eq_update
            eq_scale = eq_scale * eq_update

        if mi != 0:
            ineq_metric = _safe_max_abs_rows(Cs)
            ineq_update = 1.0 / jnp.sqrt(jnp.maximum(ineq_metric, eps_arr))
            Cs = Cs * ineq_update[:, :, None]
            ds = ds * ineq_update
            ineq_scale = ineq_scale * ineq_update

        return Qs, cs, As, bs, Cs, ds, ls, us, col_scale, eq_scale, ineq_scale

    return jax.lax.fori_loop(
        0,
        iterations,
        body,
        (Qs, cs, As, bs, Cs, ds, ls, us, col_scale, eq_scale, ineq_scale),
    )


def _primal_residual(y, A, b, C, d, l, u):
    B = y.shape[0]
    dtype = y.dtype
    eq_inf = jnp.zeros((B,), dtype=dtype)
    if A.shape[1] != 0:
        eq_inf = jnp.max(jnp.abs(jnp.einsum("bmn,bn->bm", A, y) - b), axis=1)

    ineq_inf = jnp.zeros((B,), dtype=dtype)
    if C.shape[1] != 0:
        ineq_viol = jnp.einsum("bmn,bn->bm", C, y) - d
        ineq_inf = jnp.maximum(jnp.max(ineq_viol, axis=1), 0.0)

    lower_inf = jnp.maximum(jnp.max(l - y, axis=1), 0.0)
    upper_inf = jnp.maximum(jnp.max(y - u, axis=1), 0.0)
    return jnp.maximum(jnp.maximum(eq_inf, ineq_inf), jnp.maximum(lower_inf, upper_inf))


def _primal_dual_gap(Q_diag, c, A, b, C, d, l, u, y, lam, mu):
    primal = 0.5 * jnp.sum(Q_diag * (y ** 2), axis=1) + jnp.sum(c * y, axis=1)

    t = c
    if A.shape[1] != 0:
        t = t + jnp.einsum("bmn,bm->bn", A, lam)
    if C.shape[1] != 0:
        t = t + jnp.einsum("bmn,bm->bn", C, mu)

    q_eps = 1e-12
    q_pos = Q_diag > q_eps
    y_free = -t / jnp.where(q_pos, Q_diag, 1.0)
    y_box = jnp.minimum(jnp.maximum(y_free, l), u)
    y_linear = jnp.where(t >= 0.0, l, u)
    y_dual = jnp.where(q_pos, y_box, y_linear)

    min_term = 0.5 * Q_diag * (y_dual ** 2) + t * y_dual
    dual = -jnp.sum(b * lam, axis=1) - jnp.sum(d * mu, axis=1) + jnp.sum(min_term, axis=1)

    gap = jnp.maximum(primal - dual, 0.0)
    scale = jnp.maximum(1.0, jnp.maximum(jnp.abs(primal), jnp.abs(dual)))
    return gap / scale


def _prepare_scaled_problem(Q_diag, c, A, b, C, d, l, u, *, use_ruiz, ruiz_iters):
    if not use_ruiz:
        B, n = Q_diag.shape
        me = A.shape[1]
        mi = C.shape[1]
        ones_y = jnp.ones((B, n), dtype=Q_diag.dtype)
        ones_eq = jnp.ones((B, me), dtype=Q_diag.dtype)
        ones_in = jnp.ones((B, mi), dtype=Q_diag.dtype)
        return Q_diag, c, A, b, C, d, l, u, ones_y, ones_eq, ones_in

    return ruiz_equilibrate(
        Q_diag,
        c,
        A,
        b,
        C,
        d,
        l,
        u,
        iterations=ruiz_iters,
    )


def CP_fixed(
    Q_diag,
    c,
    A,
    b,
    C,
    d,
    l,
    u,
    *,
    safety: float = 0.95,
    knorm_iters: int = 20,
    knorm_seed: int = 42,
    max_iter: int = 5000,
    tol: float = 1e-9,
    y0=None,
    lam0=None,
    mu0=None,
    use_ruiz: bool = False,
    ruiz_iters: int = 4,
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
    me = A.shape[1]
    mi = C.shape[1]

    dtype = jnp.result_type(Q_diag.dtype, c.dtype, A.dtype, b.dtype, C.dtype, d.dtype, l.dtype, u.dtype)
    Q_diag = Q_diag.astype(dtype)
    c = c.astype(dtype)
    A = A.astype(dtype)
    b = b.astype(dtype)
    C = C.astype(dtype)
    d = d.astype(dtype)
    l = l.astype(dtype)
    u = u.astype(dtype)

    Qs, cs, As, bs, Cs, ds, ls, us, col_scale, eq_scale, ineq_scale = _prepare_scaled_problem(
        Q_diag,
        c,
        A,
        b,
        C,
        d,
        l,
        u,
        use_ruiz=use_ruiz,
        ruiz_iters=ruiz_iters,
    )

    Lf = jnp.max(Qs, axis=1) if n != 0 else jnp.zeros((B,), dtype=dtype)
    knorm_key = jax.random.PRNGKey(knorm_seed)
    K_norm = estimate_AC_norm(As, Cs, knorm_key, iters=knorm_iters)

    L = jnp.maximum(K_norm, 1e-12)
    s = jnp.asarray(safety, dtype=dtype)
    sigma = s / L
    tau = s / (Lf + L)
    theta = jnp.ones((B,), dtype=dtype)
    P = 1.0 / (1.0 + tau[:, None] * Qs)

    y_init = jnp.zeros((B, n), dtype=dtype) if y0 is None else jnp.asarray(y0, dtype=dtype).reshape((B, n))
    lam_init = jnp.zeros((B, me), dtype=dtype) if lam0 is None else jnp.asarray(lam0, dtype=dtype).reshape((B, me))
    mu_init = jnp.zeros((B, mi), dtype=dtype) if mu0 is None else jnp.asarray(mu0, dtype=dtype).reshape((B, mi))

    z = y_init / col_scale
    lam = lam_init / jnp.where(eq_scale == 0.0, 1.0, eq_scale)
    mu = mu_init / jnp.where(ineq_scale == 0.0, 1.0, ineq_scale)
    zbar = z

    converged = jnp.zeros((B,), dtype=bool)
    it0 = jnp.array(0, dtype=jnp.int32)

    def cond_fn(state):
        it, z, lam, mu, zbar, converged = state
        return jnp.logical_and(it < max_iter, jnp.logical_not(jnp.all(converged)))

    def body_fn(state):
        it, z, lam, mu, zbar, converged = state

        if me != 0:
            Azbar = jnp.einsum("bmn,bn->bm", As, zbar)
            lam_hat = lam + sigma[:, None] * (Azbar - bs)
        else:
            lam_hat = lam

        if mi != 0:
            Czbar = jnp.einsum("bmn,bn->bm", Cs, zbar)
            mu_hat = jnp.maximum(mu + sigma[:, None] * (Czbar - ds), 0.0)
        else:
            mu_hat = mu

        grad = jnp.zeros_like(z)
        if me != 0:
            grad = grad + jnp.einsum("bmn,bm->bn", As, lam_hat)
        if mi != 0:
            grad = grad + jnp.einsum("bmn,bm->bn", Cs, mu_hat)

        v = z - tau[:, None] * grad
        z_next = P * (v - tau[:, None] * cs)
        z_next = jnp.minimum(jnp.maximum(z_next, ls), us)
        zbar_next = z_next + theta[:, None] * (z_next - z)

        primal_res = _primal_residual(z_next, As, bs, Cs, ds, ls, us)
        gap = _primal_dual_gap(Qs, cs, As, bs, Cs, ds, ls, us, z_next, lam_hat, mu_hat)
        newly_converged = jnp.logical_and(primal_res <= tol, gap <= tol)
        converged = jnp.logical_or(converged, newly_converged)

        return it + 1, z_next, lam_hat, mu_hat, zbar_next, converged

    _, z, lam, mu, _, _ = jax.lax.while_loop(cond_fn, body_fn, (it0, z, lam, mu, zbar, converged))

    y = col_scale * z
    lam = eq_scale * lam
    mu = ineq_scale * mu
    return y, lam, mu


CP_fixed_jit = jax.jit(
    CP_fixed,
    static_argnames=("safety", "knorm_iters", "knorm_seed", "max_iter", "tol", "use_ruiz", "ruiz_iters"),
)


def CP_accelerated(
    Q_diag,
    c,
    A,
    b,
    C,
    d,
    l,
    u,
    *,
    safety: float = 0.95,
    knorm_iters: int = 20,
    knorm_seed: int = 42,
    max_iter: int = 5000,
    tol: float = 1e-9,
    y0=None,
    lam0=None,
    mu0=None,
    use_ruiz: bool = False,
    ruiz_iters: int = 4,
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
    me = A.shape[1]
    mi = C.shape[1]
    dtype = jnp.result_type(Q_diag.dtype, c.dtype, A.dtype, b.dtype, C.dtype, d.dtype, l.dtype, u.dtype)

    Q_diag = Q_diag.astype(dtype)
    c = c.astype(dtype)
    A = A.astype(dtype)
    b = b.astype(dtype)
    C = C.astype(dtype)
    d = d.astype(dtype)
    l = l.astype(dtype)
    u = u.astype(dtype)

    Qs, cs, As, bs, Cs, ds, ls, us, col_scale, eq_scale, ineq_scale = _prepare_scaled_problem(
        Q_diag,
        c,
        A,
        b,
        C,
        d,
        l,
        u,
        use_ruiz=use_ruiz,
        ruiz_iters=ruiz_iters,
    )

    Lf = jnp.max(Qs, axis=1) if n != 0 else jnp.zeros((B,), dtype=dtype)
    gamma = jnp.maximum(jnp.min(Qs, axis=1), 0.0) if n != 0 else jnp.zeros((B,), dtype=dtype)

    knorm_key = jax.random.PRNGKey(knorm_seed)
    K_norm = estimate_AC_norm(As, Cs, knorm_key, iters=knorm_iters)
    L = jnp.maximum(K_norm, 1e-12)
    s = jnp.asarray(safety, dtype=dtype)

    sigma = s / L
    tau = jnp.where(gamma > 0.0, s / L, s / (Lf + L))
    theta = jnp.ones((B,), dtype=dtype)
    P = 1.0 / (1.0 + tau[:, None] * Qs)

    z = jnp.zeros((B, n), dtype=dtype) if y0 is None else jnp.asarray(y0, dtype=dtype).reshape((B, n)) / col_scale
    lam = jnp.zeros((B, me), dtype=dtype) if lam0 is None else jnp.asarray(lam0, dtype=dtype).reshape((B, me)) / jnp.where(eq_scale == 0.0, 1.0, eq_scale)
    mu = jnp.zeros((B, mi), dtype=dtype) if mu0 is None else jnp.asarray(mu0, dtype=dtype).reshape((B, mi)) / jnp.where(ineq_scale == 0.0, 1.0, ineq_scale)
    zbar = z

    converged = jnp.zeros((B,), dtype=bool)
    it0 = jnp.array(0, dtype=jnp.int32)

    def cond_fn(state):
        it, z, lam, mu, zbar, tau, sigma, theta, P, converged = state
        return jnp.logical_and(it < max_iter, jnp.logical_not(jnp.all(converged)))

    def body_fn(state):
        it, z, lam, mu, zbar, tau, sigma, theta, P, converged = state

        if me != 0:
            Azbar = jnp.einsum("bmn,bn->bm", As, zbar)
            lam_hat = lam + sigma[:, None] * (Azbar - bs)
        else:
            lam_hat = lam

        if mi != 0:
            Czbar = jnp.einsum("bmn,bn->bm", Cs, zbar)
            mu_hat = jnp.maximum(mu + sigma[:, None] * (Czbar - ds), 0.0)
        else:
            mu_hat = mu

        grad = jnp.zeros_like(z)
        if me != 0:
            grad = grad + jnp.einsum("bmn,bm->bn", As, lam_hat)
        if mi != 0:
            grad = grad + jnp.einsum("bmn,bm->bn", Cs, mu_hat)

        v = z - tau[:, None] * grad
        z_next = P * (v - tau[:, None] * cs)
        z_next = jnp.minimum(jnp.maximum(z_next, ls), us)
        zbar_next = z_next + theta[:, None] * (z_next - z)

        do_accel = gamma > 0.0
        theta_new = 1.0 / jnp.sqrt(1.0 + gamma * tau)
        tau_new = jnp.where(do_accel, theta_new * tau, tau)
        sigma_new = jnp.where(do_accel, sigma / theta_new, sigma)
        theta = jnp.where(do_accel, theta_new, theta)
        P_new = 1.0 / (1.0 + tau_new[:, None] * Qs)

        primal_res = _primal_residual(z_next, As, bs, Cs, ds, ls, us)
        gap = _primal_dual_gap(Qs, cs, As, bs, Cs, ds, ls, us, z_next, lam_hat, mu_hat)
        newly_converged = jnp.logical_and(primal_res <= tol, gap <= tol)
        converged = jnp.logical_or(converged, newly_converged)

        return it + 1, z_next, lam_hat, mu_hat, zbar_next, tau_new, sigma_new, theta, P_new, converged

    _, z, lam, mu, _, _, _, _, _, _ = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (it0, z, lam, mu, zbar, tau, sigma, theta, P, converged),
    )

    y = col_scale * z
    lam = eq_scale * lam
    mu = ineq_scale * mu
    return y, lam, mu


CP_accelerated_jit = jax.jit(
    CP_accelerated,
    static_argnames=("safety", "knorm_iters", "knorm_seed", "max_iter", "tol", "use_ruiz", "ruiz_iters"),
)
