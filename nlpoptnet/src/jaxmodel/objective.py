"""Objective helpers for structured and general nonlinear models."""

from dataclasses import dataclass
from typing import Callable, Dict
import jax.numpy as jnp
from .variables import VariableSpec
from .types import ParamsDict, VarsDict, ScalarModelFun, FlatScalarFun


@dataclass
class QuadraticObjective:
    r"""Represent a quadratic objective in the flat variable vector.

    The objective is modeled as

    .. math::

        f(y) = \frac{1}{2} y^\top Q y + c^\top y + k

    where:

    - :math:`y \in \mathbf{R}^n` is the packed decision-variable vector,
    - :math:`Q \in \mathbf{R}^{n \times n}` is the quadratic matrix,
    - :math:`c \in \mathbf{R}^n` is the linear coefficient vector,
    - :math:`k \in \mathbf{R}` is a constant offset.

    This structured form is useful because the gradient and Hessian are known
    analytically:

    .. math::

        \nabla f(y) = Q y + c,\qquad \nabla^2 f(y) = Q

    Attributes
    ----------
    Q:
        Quadratic coefficient matrix.
    c:
        Linear coefficient vector.
    constant:
        Constant scalar offset.
    """

    Q: jnp.ndarray
    c: jnp.ndarray
    constant: float = 0.0

    def __post_init__(self):
        """Validate and normalize the quadratic data."""
        self.Q = jnp.asarray(self.Q)
        self.c = jnp.asarray(self.c)
        if self.Q.ndim != 2 or self.Q.shape[0] != self.Q.shape[1]:
            raise ValueError(f"Q must be square, got shape {self.Q.shape}")
        if self.c.ndim != 1 or self.c.shape[0] != self.Q.shape[0]:
            raise ValueError(
                f"c shape {self.c.shape} must match Q dimension {self.Q.shape[0]}"
            )

    def value_from_flat(self, y_flat: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the objective at a flat variable vector."""
        return 0.5 * y_flat @ self.Q @ y_flat + self.c @ y_flat + self.constant

    def grad_from_flat(self, y_flat: jnp.ndarray) -> jnp.ndarray:
        """Return the gradient with respect to the flat variable vector."""
        return self.Q @ y_flat + self.c

    def hess_from_flat(self, y_flat: jnp.ndarray) -> jnp.ndarray:
        """Return the full Hessian matrix."""
        return self.Q

    def diag_hess_from_flat(self, y_flat: jnp.ndarray) -> jnp.ndarray:
        """Return the diagonal of the Hessian."""
        return jnp.diag(self.Q)


def wrap_scalar_objective(
    var_spec: VariableSpec,
    fun: ScalarModelFun,
) -> FlatScalarFun:
    """Wrap an objective on named variables into a flat-vector objective."""
    def wrapped(params: ParamsDict, y_flat: jnp.ndarray) -> jnp.ndarray:
        vars_dict = var_spec.unpack(y_flat)
        return jnp.asarray(fun(params, vars_dict))
    return wrapped


def resolve_objective(var_spec: VariableSpec, objective_fun):
    """Normalize a user objective into flat evaluators and derivatives."""
    if isinstance(objective_fun, QuadraticObjective):
        n = var_spec.total_size
        if objective_fun.Q.shape != (n, n):
            raise ValueError(
                f"Quadratic objective Q shape must be {(n, n)}, got {objective_fun.Q.shape}"
            )
        if objective_fun.c.shape != (n,):
            raise ValueError(
                f"Quadratic objective c shape must be {(n,)}, got {objective_fun.c.shape}"
            )

        def obj_flat(params: ParamsDict, y_flat: jnp.ndarray) -> jnp.ndarray:
            return objective_fun.value_from_flat(y_flat)

        grad_fun = lambda y, params: objective_fun.grad_from_flat(y)
        hess_fun = lambda y, params: objective_fun.hess_from_flat(y)
        diag_hess_fun = lambda params, y: objective_fun.diag_hess_from_flat(y)
        return "quadratic", obj_flat, grad_fun, hess_fun, diag_hess_fun

    obj_flat = wrap_scalar_objective(var_spec, objective_fun)
    return "nonlinear", obj_flat, None, None, None
