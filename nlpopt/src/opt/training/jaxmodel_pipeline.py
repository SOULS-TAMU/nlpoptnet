from __future__ import annotations

from functools import partial
from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax

from jaxmodel import JaxNLPModel

from ..models import Backbone, State
from ..projection import solve_cp_implicit
from .config import TrainConfig

Array = jnp.ndarray


def make_subproblem_layer_from_model(
    model: JaxNLPModel,
    *,
    param_name: str = "x",
    rho: float = 1.0,
    use_diagonal_hessian: bool = True,
    diag_floor: float = 1e-8,
):
    dtype = model.dtype
    n = model.var_spec.total_size

    def _single(x: Array, y: Array):
        params = {param_name: x}
        sqp = model.sqp_subproblem_data(
            params,
            y,
            rho=rho,
            use_diagonal_hessian=use_diagonal_hessian,
            diag_floor=diag_floor,
        )
        l = sqp.l if sqp.l is not None else -jnp.inf * jnp.ones((n,), dtype=dtype)
        u = sqp.u if sqp.u is not None else jnp.inf * jnp.ones((n,), dtype=dtype)
        return (
            sqp.objective.Q_diag,
            sqp.objective.c,
            sqp.constraints.A,
            sqp.constraints.b,
            sqp.constraints.C,
            sqp.constraints.d,
            l,
            u,
        )

    def fwd_impl(x_batch: Array, y_batch: Array):
        return jax.vmap(_single, in_axes=(0, 0))(x_batch, y_batch)

    fwd_jit = jax.jit(fwd_impl)

    @jax.custom_vjp
    def layer(x_batch: Array, y_batch: Array):
        return fwd_jit(x_batch, y_batch)

    def fwd(x_batch: Array, y_batch: Array):
        return fwd_jit(x_batch, y_batch), (x_batch, y_batch)

    def bwd(res, ct):
        x_batch, y_batch = res
        _, vjp_fun = jax.vjp(fwd_impl, x_batch, y_batch)
        return vjp_fun(ct)

    layer.defvjp(fwd, bwd)
    return layer


def make_batched_objective(model: JaxNLPModel, *, param_name: str = "x"):
    def _single(x: Array, y: Array):
        return model.objective_value({param_name: x}, y)

    return jax.jit(jax.vmap(_single, in_axes=(0, 0)))


def make_batched_residuals(model: JaxNLPModel, *, param_name: str = "x"):
    def _eq_single(x: Array, y: Array):
        return model.eq_residual({param_name: x}, y)

    def _ineq_single(x: Array, y: Array):
        return model.ineq_residual({param_name: x}, y)

    return (
        jax.jit(jax.vmap(_eq_single, in_axes=(0, 0))),
        jax.jit(jax.vmap(_ineq_single, in_axes=(0, 0))),
    )


def apply_projection_layers(
    *,
    sub_layer: Callable[[Array, Array], Tuple[Array, Array, Array, Array, Array, Array, Array, Array]],
    x_batch: Array,
    y0: Array,
    lam0: Array,
    mu0: Array,
    cfg: TrainConfig,
):
    y_curr = y0
    lam_curr = lam0
    mu_curr = mu0

    for _ in range(int(cfg.k_layer)):
        Q_diag, c, A, b, C, d, l, u = sub_layer(x_batch, y_curr)
        y_curr, lam_curr, mu_curr = solve_cp_implicit(
            Q_diag,
            c,
            A,
            b,
            C,
            d,
            l,
            u,
            cfg=cfg,
            y0=y_curr,
            lam0=lam_curr,
            mu0=mu_curr,
        )

    return y_curr, lam_curr, mu_curr


def build_train_fns_from_jaxmodel(
    *,
    model_def: JaxNLPModel,
    cfg: TrainConfig,
    p: int,
    param_name: str = "x",
    rho: float = 1.0,
    use_diagonal_hessian: bool = True,
    diag_floor: float = 1e-8,
):
    example_x = jnp.zeros((p,), dtype=model_def.dtype)
    example_y = jnp.zeros((model_def.var_spec.total_size,), dtype=model_def.dtype)
    me = int(model_def.eq_residual({param_name: example_x}, example_y).shape[0])
    mi = int(model_def.ineq_residual({param_name: example_x}, example_y).shape[0])
    n = model_def.var_spec.total_size

    model = Backbone(p=p, n=n, me=me, mi=mi, hidden_size=cfg.hidden_size, hidden_dim=cfg.hidden_dim)
    sub_layer = make_subproblem_layer_from_model(
        model_def,
        param_name=param_name,
        rho=rho,
        use_diagonal_hessian=use_diagonal_hessian,
        diag_floor=diag_floor,
    )
    obj_fn = make_batched_objective(model_def, param_name=param_name)
    eq_resid_fn, ineq_resid_fn = make_batched_residuals(model_def, param_name=param_name)

    @jax.jit
    def loss_fn(params, x_batch):
        y_hat, lam_hat, mu_hat = model.apply({"params": params}, x_batch)
        y_tilde, lam_tilde, mu_tilde = apply_projection_layers(
            sub_layer=sub_layer,
            x_batch=x_batch,
            y0=y_hat,
            lam0=lam_hat,
            mu0=mu_hat,
            cfg=cfg,
        )
        obj = jnp.mean(obj_fn(x_batch, y_tilde))
        eq_penalty = jnp.asarray(0.0, dtype=model_def.dtype)
        if me > 0:
            eq_resid = eq_resid_fn(x_batch, y_tilde)
            eq_penalty = jnp.mean(jnp.square(jnp.sum(lam_tilde * eq_resid, axis=1)))
        ineq_penalty = jnp.asarray(0.0, dtype=model_def.dtype)
        if mi > 0:
            ineq_resid = ineq_resid_fn(x_batch, y_tilde)
            ineq_penalty = jnp.mean(jnp.sum(mu_tilde * jnp.maximum(ineq_resid, 0.0), axis=1))
        mse_y = jnp.mean((y_hat - y_tilde) ** 2)
        mse_lam = jnp.mean((lam_hat - lam_tilde) ** 2) if me > 0 else jnp.asarray(0.0, dtype=model_def.dtype)
        mse_mu = jnp.mean((mu_hat - mu_tilde) ** 2) if mi > 0 else jnp.asarray(0.0, dtype=model_def.dtype)
        cons = mse_y + mse_lam + mse_mu
        total = obj + eq_penalty + ineq_penalty + jnp.asarray(cfg.alpha_consistency, dtype=model_def.dtype) * cons
        metrics = {
            "loss": total,
            "obj": obj,
            "eq_penalty": eq_penalty,
            "ineq_penalty": ineq_penalty,
            "mse_y": mse_y,
            "mse_lam": mse_lam,
            "mse_mu": mse_mu,
        }
        return total, metrics

    @partial(jax.jit, donate_argnums=(0,))
    def train_step(state: State, x_batch: Array):
        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, x_batch)
        state = state.apply_gradients(grads=grads)
        return state, metrics

    @jax.jit
    def eval_step(params, x_batch: Array):
        _, metrics = loss_fn(params, x_batch)
        return metrics

    def init_state(rng: jax.random.KeyArray):
        dummy_x = jnp.zeros((cfg.batch_size, p), dtype=model_def.dtype)
        params = model.init(rng, dummy_x)["params"]
        tx = optax.adam(cfg.learning_rate)
        return State.create(apply_fn=model.apply, params=params, tx=tx)

    return model, init_state, train_step, eval_step


def build_violation_fn_from_jaxmodel(model_def: JaxNLPModel, *, cfg: TrainConfig, p: int, param_name: str = "x"):
    example_x = jnp.zeros((p,), dtype=model_def.dtype)
    example_y = jnp.zeros((model_def.var_spec.total_size,), dtype=model_def.dtype)
    me = int(model_def.eq_residual({param_name: example_x}, example_y).shape[0])
    mi = int(model_def.ineq_residual({param_name: example_x}, example_y).shape[0])
    n = model_def.var_spec.total_size

    model = Backbone(p=p, n=n, me=me, mi=mi, hidden_size=cfg.hidden_size, hidden_dim=cfg.hidden_dim)
    sub_layer = make_subproblem_layer_from_model(model_def, param_name=param_name)

    @jax.jit
    def viol_fn(params, x_batch):
        y_hat, lam_hat, mu_hat = model.apply({"params": params}, x_batch)
        y_tilde, _, _ = apply_projection_layers(
            sub_layer=sub_layer,
            x_batch=x_batch,
            y0=y_hat,
            lam0=lam_hat,
            mu0=mu_hat,
            cfg=cfg,
        )

        def _single(x, y):
            params_single = {param_name: x}
            eq = model_def.eq_residual(params_single, y)
            ineq = model_def.ineq_residual(params_single, y)
            lb = model_def.lower_bounds(params_single)
            ub = model_def.upper_bounds(params_single)
            eq_inf = jnp.max(jnp.abs(eq)) if eq.shape[0] > 0 else jnp.asarray(0.0, dtype=y.dtype)
            eq_mean = jnp.mean(jnp.abs(eq)) if eq.shape[0] > 0 else jnp.asarray(0.0, dtype=y.dtype)
            ineq_inf = jnp.max(jnp.maximum(ineq, 0.0)) if ineq.shape[0] > 0 else jnp.asarray(0.0, dtype=y.dtype)
            ineq_mean = jnp.mean(jnp.maximum(ineq, 0.0)) if ineq.shape[0] > 0 else jnp.asarray(0.0, dtype=y.dtype)
            lb_inf = jnp.max(jnp.maximum(lb - y, 0.0)) if lb is not None else jnp.asarray(0.0, dtype=y.dtype)
            ub_inf = jnp.max(jnp.maximum(y - ub, 0.0)) if ub is not None else jnp.asarray(0.0, dtype=y.dtype)
            lb_mean = jnp.mean(jnp.maximum(lb - y, 0.0)) if lb is not None else jnp.asarray(0.0, dtype=y.dtype)
            ub_mean = jnp.mean(jnp.maximum(y - ub, 0.0)) if ub is not None else jnp.asarray(0.0, dtype=y.dtype)
            return eq_inf, eq_mean, ineq_inf, ineq_mean, jnp.maximum(lb_inf, ub_inf), jnp.maximum(lb_mean, ub_mean)

        eq_vals, eq_mean_vals, ineq_vals, ineq_mean_vals, bnd_vals, bnd_mean_vals = jax.vmap(_single, in_axes=(0, 0))(x_batch, y_tilde)
        return {
            "eq_inf": jnp.max(eq_vals),
            "eq_mean": jnp.mean(eq_mean_vals),
            "ineq_inf": jnp.max(ineq_vals),
            "ineq_mean": jnp.mean(ineq_mean_vals),
            "bound_inf": jnp.max(bnd_vals),
            "bound_mean": jnp.mean(bnd_mean_vals),
        }

    return viol_fn
