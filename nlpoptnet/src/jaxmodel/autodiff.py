"""Automatic-differentiation helpers for flat JAX objectives and constraints."""

import jax
import jax.numpy as jnp


def build_grad_y_objective(obj_flat):
    """Build the gradient of a flat objective with respect to variables."""
    return jax.grad(lambda y, p: obj_flat(p, y), argnums=0)


def build_hess_y_objective(obj_flat):
    """Build the Hessian of a flat objective with respect to variables."""
    return jax.hessian(lambda y, p: obj_flat(p, y), argnums=0)


def build_diag_hess_y_objective(grad_fun):
    """Build a diagonal-Hessian evaluator from a gradient function."""
    def diag_hess(params, y):
        n = y.shape[0]
        basis = jnp.eye(n, dtype=y.dtype)
        _, jvp_fun = jax.linearize(lambda yy: grad_fun(yy, params), y)
        cols = jax.vmap(jvp_fun)(basis)
        return jnp.diag(cols.T)
    return diag_hess


def build_jac_y_vector_fun(vec_fun):
    """Build the Jacobian of a vector-valued residual with respect to variables."""
    return jax.jacrev(lambda y, p: vec_fun(p, y), argnums=0)
