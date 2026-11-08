"""Constraints resources for the NLPOptNet frontend."""

from __future__ import annotations
from typing import Any
from typing import Callable
from typing import Optional
from nlpoptnet.api.formulation.parameters import ParamsDict
from typing import Sequence
from typing import Tuple
from typing import Union
from nlpoptnet.api.formulation.variables import VariableSpec
from nlpoptnet.api.formulation.parameters import VectorModelFun
from dataclasses import dataclass
import jax
import jax.numpy as jnp


def _as_1d(z: jnp.ndarray) -> jnp.ndarray:
    """Normalize scalar or vector outputs into a one-dimensional array."""
    z = jnp.asarray(z)
    if z.ndim == 0:
        return z[None]
    if z.ndim != 1:
        raise ValueError(f"Expected 1D output, got shape {z.shape}")
    return z


def wrap_vector_constraint(var_spec: VariableSpec, fun: VectorModelFun):
    """Wrap a constraint on named variables into a flat-vector residual."""

    def wrapped(params: ParamsDict, y_flat: jnp.ndarray) -> jnp.ndarray:
        vars_dict = var_spec.unpack(y_flat)
        return _as_1d(fun(params, vars_dict))

    return wrapped


@dataclass
class ConstraintEntry:
    r"""Container for one residual block in the optimization model.

    Each constraint entry represents either an equality residual

    .. math::

        h(y, x) = 0

    or an inequality residual

    .. math::

        g(y, x) \le 0

    written in residual form with respect to the packed variable vector
    :math:`y` and parameter vector :math:`x`.

    The ``fun`` callable returns the residual block, while ``jac_y_fun`` stores
    an optional Jacobian with respect to :math:`y`.

    Attributes
    ----------
    name:
        Human-readable identifier for the residual block.
    kind:
        Either ``"eq"`` or ``"ineq"``.
    fun:
        Callable returning the residual vector.
    structure:
        Structural label such as ``affine``, ``quadratic``, or ``nonlinear``.
    jac_y_fun:
        Optional callable for the Jacobian with respect to variables.
    metadata:
        Optional structured metadata describing how the block was assembled.
    """

    name: str
    kind: str  # "eq" or "ineq"
    fun: callable
    structure: str = "nonlinear"
    jac_y_fun: Optional[callable] = None
    metadata: Optional[dict[str, Any]] = None


def _validate_quadratic_shapes(var_spec: VariableSpec, Q: jnp.ndarray, c: jnp.ndarray):
    """Validate the dimensions of quadratic-constraint data."""
    n = var_spec.total_size
    Q = jnp.asarray(Q)
    c = jnp.asarray(c)
    if Q.shape != (n, n):
        raise ValueError(f"Q must have shape {(n, n)}, got {Q.shape}")
    if c.shape != (n,):
        raise ValueError(f"c must have shape {(n,)}, got {c.shape}")
    return Q, c


def eq_scalar(
    fun,
    var_spec: VariableSpec,
    name: str = "eq_scalar",
    structure: str = "nonlinear",
    jac_y_fun: Optional[callable] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> ConstraintEntry:
    """Create a scalar equality constraint entry."""
    wrapped = wrap_vector_constraint(var_spec, lambda p, v: jnp.array([fun(p, v)]))
    return ConstraintEntry(
        name=name,
        kind="eq",
        fun=wrapped,
        structure=structure,
        jac_y_fun=jac_y_fun,
        metadata=metadata,
    )


def ineq_scalar(
    fun,
    var_spec: VariableSpec,
    name: str = "ineq_scalar",
    structure: str = "nonlinear",
    jac_y_fun: Optional[callable] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> ConstraintEntry:
    """Create a scalar inequality constraint entry."""
    wrapped = wrap_vector_constraint(var_spec, lambda p, v: jnp.array([fun(p, v)]))
    return ConstraintEntry(
        name=name,
        kind="ineq",
        fun=wrapped,
        structure=structure,
        jac_y_fun=jac_y_fun,
        metadata=metadata,
    )


def eq_block(
    fun,
    var_spec: VariableSpec,
    name: str = "eq_block",
    structure: str = "nonlinear",
    jac_y_fun: Optional[callable] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> ConstraintEntry:
    """Create a vector-valued equality constraint block."""
    wrapped = wrap_vector_constraint(var_spec, fun)
    return ConstraintEntry(
        name=name,
        kind="eq",
        fun=wrapped,
        structure=structure,
        jac_y_fun=jac_y_fun,
        metadata=metadata,
    )


def ineq_block(
    fun,
    var_spec: VariableSpec,
    name: str = "ineq_block",
    structure: str = "nonlinear",
    jac_y_fun: Optional[callable] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> ConstraintEntry:
    """Create a vector-valued inequality constraint block."""
    wrapped = wrap_vector_constraint(var_spec, fun)
    return ConstraintEntry(
        name=name,
        kind="ineq",
        fun=wrapped,
        structure=structure,
        jac_y_fun=jac_y_fun,
        metadata=metadata,
    )


def quadratic_eq_scalar(
    var_spec: VariableSpec,
    Q: jnp.ndarray,
    c: jnp.ndarray,
    rhs_const: Union[float, jnp.ndarray] = 0.0,
    x_coeff: Optional[jnp.ndarray] = None,
    x_name: Optional[str] = None,
    name: str = "quadratic_eq",
) -> ConstraintEntry:
    """Create a scalar quadratic equality constraint."""
    Q, c = _validate_quadratic_shapes(var_spec, Q, c)
    Qsym = 0.5 * (Q + Q.T)
    rhs_const = jnp.asarray(rhs_const)

    if (x_coeff is None) != (x_name is None):
        raise ValueError("x_coeff and x_name must be provided together.")

    def fun(params: ParamsDict, y_flat: jnp.ndarray) -> jnp.ndarray:
        rhs = rhs_const
        if x_coeff is not None:
            rhs = rhs + jnp.asarray(x_coeff) @ jnp.ravel(params[x_name])
        val = 0.5 * y_flat @ Qsym @ y_flat + c @ y_flat - rhs
        return jnp.array([val])

    def jac_y_fun(params: ParamsDict, y_flat: jnp.ndarray) -> jnp.ndarray:
        return (Qsym @ y_flat + c)[None, :]

    return ConstraintEntry(
        name=name,
        kind="eq",
        fun=fun,
        structure="quadratic",
        jac_y_fun=jac_y_fun,
        metadata={
            "type": "quadratic_scalar",
            "Q": Qsym,
            "c": c,
            "rhs_const": rhs_const,
            "x_coeff": x_coeff,
            "x_name": x_name,
        },
    )


def quadratic_ineq_scalar(
    var_spec: VariableSpec,
    Q: jnp.ndarray,
    c: jnp.ndarray,
    rhs_const: Union[float, jnp.ndarray] = 0.0,
    x_coeff: Optional[jnp.ndarray] = None,
    x_name: Optional[str] = None,
    name: str = "quadratic_ineq",
) -> ConstraintEntry:
    """Create a scalar quadratic inequality constraint."""
    Q, c = _validate_quadratic_shapes(var_spec, Q, c)
    Qsym = 0.5 * (Q + Q.T)
    rhs_const = jnp.asarray(rhs_const)

    if (x_coeff is None) != (x_name is None):
        raise ValueError("x_coeff and x_name must be provided together.")

    def fun(params: ParamsDict, y_flat: jnp.ndarray) -> jnp.ndarray:
        rhs = rhs_const
        if x_coeff is not None:
            rhs = rhs + jnp.asarray(x_coeff) @ jnp.ravel(params[x_name])
        val = 0.5 * y_flat @ Qsym @ y_flat + c @ y_flat - rhs
        return jnp.array([val])

    def jac_y_fun(params: ParamsDict, y_flat: jnp.ndarray) -> jnp.ndarray:
        return (Qsym @ y_flat + c)[None, :]

    return ConstraintEntry(
        name=name,
        kind="ineq",
        fun=fun,
        structure="quadratic",
        jac_y_fun=jac_y_fun,
        metadata={
            "type": "quadratic_scalar",
            "Q": Qsym,
            "c": c,
            "rhs_const": rhs_const,
            "x_coeff": x_coeff,
            "x_name": x_name,
        },
    )


def aggregate_constraints(entries: Sequence[ConstraintEntry], dtype=jnp.float64):
    """Concatenate multiple residual blocks into one vector-valued function."""

    def agg(params: ParamsDict, y_flat: jnp.ndarray) -> jnp.ndarray:
        if len(entries) == 0:
            return jnp.zeros((0,), dtype=dtype)
        vals = [e.fun(params, y_flat) for e in entries]
        return jnp.concatenate([_as_1d(v) for v in vals], axis=0)

    return agg


def aggregate_constraint_jacobian(
    entries: Sequence[ConstraintEntry],
    n_vars: int,
    dtype=jnp.float64,
):
    """Aggregate per-constraint Jacobians into one dense Jacobian matrix."""
    jac_parts = []
    for entry in entries:
        if entry.jac_y_fun is not None:
            jac_parts.append(lambda y, p, jf=entry.jac_y_fun: jf(p, y))
        else:
            jac_parts.append(jax.jacrev(lambda y, p, f=entry.fun: f(p, y), argnums=0))

    def agg_jac(y: jnp.ndarray, params: ParamsDict) -> jnp.ndarray:
        if len(jac_parts) == 0:
            return jnp.zeros((0, n_vars), dtype=dtype)
        mats = [jp(y, params) for jp in jac_parts]
        return jnp.concatenate(mats, axis=0)

    return agg_jac


def _build_affine_jacobian_matrix(
    var_spec: VariableSpec,
    y_blocks: Sequence[Tuple[jnp.ndarray, str]],
) -> jnp.ndarray:
    """Assemble a dense Jacobian from named affine variable blocks."""
    n = var_spec.total_size
    if len(y_blocks) == 0:
        return jnp.zeros((0, n), dtype=jnp.float64)

    m = int(jnp.asarray(y_blocks[0][0]).shape[0])
    full = jnp.zeros((m, n), dtype=jnp.asarray(y_blocks[0][0]).dtype)

    for mat, yname in y_blocks:
        block = jnp.asarray(mat)
        sl = var_spec.slices[yname]
        full = full.at[:, sl].add(block)
    return full


def affine_eq_from_parts(
    var_spec: VariableSpec,
    y_blocks: Sequence[Tuple[jnp.ndarray, str]],
    rhs_const: Optional[jnp.ndarray] = None,
    x_blocks: Optional[Sequence[Tuple[jnp.ndarray, str]]] = None,
    name: str = "affine_eq",
):
    """Build an affine equality block from named variable and parameter pieces."""
    rhs_const = 0.0 if rhs_const is None else rhs_const
    x_blocks = [] if x_blocks is None else list(x_blocks)

    def fun(params, vars):
        lhs = 0.0
        for A, yname in y_blocks:
            lhs = lhs + A @ jnp.ravel(vars[yname])

        rhs = jnp.asarray(rhs_const)
        for D, xname in x_blocks:
            rhs = rhs + D @ jnp.ravel(params[xname])

        return lhs - rhs

    jac_matrix = _build_affine_jacobian_matrix(var_spec, y_blocks)
    jac_y_fun = lambda params, y, J=jac_matrix: J

    return eq_block(
        fun,
        var_spec,
        name=name,
        structure="affine",
        jac_y_fun=jac_y_fun,
        metadata={
            "type": "affine_block",
            "jac_matrix": jac_matrix,
            "rhs_const": jnp.asarray(rhs_const),
            "x_blocks": list(x_blocks),
        },
    )


def affine_ineq_from_parts(
    var_spec: VariableSpec,
    y_blocks: Sequence[Tuple[jnp.ndarray, str]],
    rhs_const: Optional[jnp.ndarray] = None,
    x_blocks: Optional[Sequence[Tuple[jnp.ndarray, str]]] = None,
    name: str = "affine_ineq",
):
    """Build an affine inequality block from named variable and parameter pieces."""
    rhs_const = 0.0 if rhs_const is None else rhs_const
    x_blocks = [] if x_blocks is None else list(x_blocks)

    def fun(params, vars):
        lhs = 0.0
        for C, yname in y_blocks:
            lhs = lhs + C @ jnp.ravel(vars[yname])

        rhs = jnp.asarray(rhs_const)
        for E, xname in x_blocks:
            rhs = rhs + E @ jnp.ravel(params[xname])

        return lhs - rhs

    jac_matrix = _build_affine_jacobian_matrix(var_spec, y_blocks)
    jac_y_fun = lambda params, y, J=jac_matrix: J

    return ineq_block(
        fun,
        var_spec,
        name=name,
        structure="affine",
        jac_y_fun=jac_y_fun,
        metadata={
            "type": "affine_block",
            "jac_matrix": jac_matrix,
            "rhs_const": jnp.asarray(rhs_const),
            "x_blocks": list(x_blocks),
        },
    )


def nonlinear_eq_from_parts(
    var_spec: VariableSpec,
    linear_y_terms: Optional[Sequence[Tuple[jnp.ndarray, str]]] = None,
    nonlinear_y_terms: Optional[Sequence[Tuple[jnp.ndarray, Callable, str]]] = None,
    rhs_const: Optional[jnp.ndarray] = None,
    x_blocks: Optional[Sequence[Tuple[jnp.ndarray, str]]] = None,
    x_direct_name: Optional[str] = None,
    name: str = "nonlinear_eq",
):
    """Build a nonlinear equality block from linear and transformed terms."""
    linear_y_terms = [] if linear_y_terms is None else list(linear_y_terms)
    nonlinear_y_terms = [] if nonlinear_y_terms is None else list(nonlinear_y_terms)
    x_blocks = [] if x_blocks is None else list(x_blocks)
    rhs_const = 0.0 if rhs_const is None else rhs_const

    def fun(params, vars):
        lhs = 0.0
        for A, yname in linear_y_terms:
            lhs = lhs + A @ jnp.ravel(vars[yname])

        for B, phi, yname in nonlinear_y_terms:
            lhs = lhs + B @ phi(jnp.ravel(vars[yname]))

        rhs = jnp.asarray(rhs_const)
        for D, xname in x_blocks:
            rhs = rhs + D @ jnp.ravel(params[xname])

        if x_direct_name is not None:
            rhs = rhs + jnp.ravel(params[x_direct_name])

        return lhs - rhs

    return eq_block(fun, var_spec, name=name)


def nonlinear_ineq_from_parts(
    var_spec: VariableSpec,
    linear_y_terms: Optional[Sequence[Tuple[jnp.ndarray, str]]] = None,
    nonlinear_y_terms: Optional[Sequence[Tuple[jnp.ndarray, Callable, str]]] = None,
    rhs_const: Optional[jnp.ndarray] = None,
    x_blocks: Optional[Sequence[Tuple[jnp.ndarray, str]]] = None,
    name: str = "nonlinear_ineq",
):
    """Build a nonlinear inequality block from linear and transformed terms."""
    linear_y_terms = [] if linear_y_terms is None else list(linear_y_terms)
    nonlinear_y_terms = [] if nonlinear_y_terms is None else list(nonlinear_y_terms)
    x_blocks = [] if x_blocks is None else list(x_blocks)
    rhs_const = 0.0 if rhs_const is None else rhs_const

    def fun(params, vars):
        lhs = 0.0
        for C, yname in linear_y_terms:
            lhs = lhs + C @ jnp.ravel(vars[yname])

        for F, phi, yname in nonlinear_y_terms:
            lhs = lhs + F @ phi(jnp.ravel(vars[yname]))

        rhs = jnp.asarray(rhs_const)
        for E, xname in x_blocks:
            rhs = rhs + E @ jnp.ravel(params[xname])

        return lhs - rhs

    return ineq_block(fun, var_spec, name=name)
