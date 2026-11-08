"""Objectives resources for the NLPOptNet frontend."""

from __future__ import annotations
from jax.numpy import ndarray as Array
from nlpoptnet.api.network import Backbone
from nlpoptnet.api.formulation.compiler import JaxNLPModel
from nlpoptnet.api.configuration import TrainConfig
import jax
import jax.numpy as jnp
from nlpoptnet.api.formulation.approximation import make_subproblem_layer_from_model
from ohagi.losses.lagrangian import modified_lagrangian_loss
from nlpoptnet.api.projection import projected_model as projected_model


def nlpoptnet_loss(problem, *, alpha_consistency=10.0):
    """Adapt OHAGI's modified Lagrangian loss to NLPOPTNet metric names."""
    if alpha_consistency < 0:
        raise ValueError("alpha_consistency must be nonnegative")
    from purin.compilation import compile_problem

    c = compile_problem(problem)
    shared = modified_lagrangian_loss(
        jax.vmap(c.objective),
        jax.vmap(c.equalities),
        jax.vmap(c.inequalities),
        alpha_consistency=alpha_consistency,
    )

    def loss(outputs, batch):
        total, metrics = shared(outputs, batch)
        return total, {
            "objective": metrics["obj"],
            "equality_penalty": metrics["eq_penalty"],
            "inequality_penalty": metrics["ineq_penalty"],
            "consistency_loss": metrics["consistency"],
        }

    return loss


def make_batched_objective(model: JaxNLPModel, *, param_name: str = "x"):
    """Vectorize objective evaluation across a batch of parameters and variables."""

    def _single(x: Array, y: Array):
        return model.objective_value({param_name: x}, y)

    return jax.jit(jax.vmap(_single, in_axes=(0, 0)))


def make_batched_residuals(model: JaxNLPModel, *, param_name: str = "x"):
    """Vectorize equality and inequality residual evaluation."""

    def _eq_single(x: Array, y: Array):
        return model.eq_residual({param_name: x}, y)

    def _ineq_single(x: Array, y: Array):
        return model.ineq_residual({param_name: x}, y)

    return (
        jax.jit(jax.vmap(_eq_single, in_axes=(0, 0))),
        jax.jit(jax.vmap(_ineq_single, in_axes=(0, 0))),
    )


def build_violation(
    model_def: JaxNLPModel, *, cfg: TrainConfig, p: int, param_name: str = "x"
):
    """Build a batched violation-summary function for a trained backbone."""
    from nlpoptnet.api.projection import projected_model

    example_x = jnp.zeros((p,), dtype=model_def.dtype)
    example_y = jnp.zeros((model_def.var_spec.total_size,), dtype=model_def.dtype)
    me = int(model_def.eq_residual({param_name: example_x}, example_y).shape[0])
    mi = int(model_def.ineq_residual({param_name: example_x}, example_y).shape[0])
    n = model_def.var_spec.total_size

    model = Backbone(
        p=p, n=n, me=me, mi=mi, hidden_size=cfg.hidden_size, hidden_dim=cfg.hidden_dim
    )
    sub_layer = make_subproblem_layer_from_model(model_def, param_name=param_name)
    shared_model = projected_model(model, sub_layer, cfg)

    @jax.jit
    def viol_fn(params, x_batch):
        outputs, _ = shared_model.apply(params, {}, x_batch)
        y_tilde = outputs["projected"]

        def _single(x, y):
            params_single = {param_name: x}
            eq = model_def.eq_residual(params_single, y)
            ineq = model_def.ineq_residual(params_single, y)
            lb = model_def.lower_bounds(params_single)
            ub = model_def.upper_bounds(params_single)
            eq_inf = (
                jnp.max(jnp.abs(eq))
                if eq.shape[0] > 0
                else jnp.asarray(0.0, dtype=y.dtype)
            )
            eq_mean = (
                jnp.mean(jnp.abs(eq))
                if eq.shape[0] > 0
                else jnp.asarray(0.0, dtype=y.dtype)
            )
            ineq_inf = (
                jnp.max(jnp.maximum(ineq, 0.0))
                if ineq.shape[0] > 0
                else jnp.asarray(0.0, dtype=y.dtype)
            )
            ineq_mean = (
                jnp.mean(jnp.maximum(ineq, 0.0))
                if ineq.shape[0] > 0
                else jnp.asarray(0.0, dtype=y.dtype)
            )
            lb_inf = (
                jnp.max(jnp.maximum(lb - y, 0.0))
                if lb is not None
                else jnp.asarray(0.0, dtype=y.dtype)
            )
            ub_inf = (
                jnp.max(jnp.maximum(y - ub, 0.0))
                if ub is not None
                else jnp.asarray(0.0, dtype=y.dtype)
            )
            lb_mean = (
                jnp.mean(jnp.maximum(lb - y, 0.0))
                if lb is not None
                else jnp.asarray(0.0, dtype=y.dtype)
            )
            ub_mean = (
                jnp.mean(jnp.maximum(y - ub, 0.0))
                if ub is not None
                else jnp.asarray(0.0, dtype=y.dtype)
            )
            return (
                eq_inf,
                eq_mean,
                ineq_inf,
                ineq_mean,
                jnp.maximum(lb_inf, ub_inf),
                jnp.maximum(lb_mean, ub_mean),
            )

        eq_vals, eq_mean_vals, ineq_vals, ineq_mean_vals, bnd_vals, bnd_mean_vals = (
            jax.vmap(_single, in_axes=(0, 0))(x_batch, y_tilde)
        )
        return {
            "eq_inf": jnp.max(eq_vals),
            "eq_mean": jnp.mean(eq_mean_vals),
            "ineq_inf": jnp.max(ineq_vals),
            "ineq_mean": jnp.mean(ineq_mean_vals),
            "bound_inf": jnp.max(bnd_vals),
            "bound_mean": jnp.mean(bnd_mean_vals),
        }

    return viol_fn
