"""Approximation resources for the NLPOptNet frontend."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nlpoptnet.api.formulation.compiler import JaxNLPModel

from jax.numpy import ndarray as Array
from nlpoptnet.api.formulation.parameters import FlatScalarFun
from typing import NamedTuple
from typing import Optional
from nlpoptnet.api.formulation.parameters import ParamsDict
from nlpoptnet.api.formulation.parameters import ScalarModelFun
from nlpoptnet.api.formulation.variables import VariableSpec
from dataclasses import dataclass
import jax
import jax.numpy as jnp


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


class LinearizationData(NamedTuple):
    """Affine approximation data for equality and inequality residuals."""

    A: jnp.ndarray
    b: jnp.ndarray
    C: jnp.ndarray
    d: jnp.ndarray
    h_val: jnp.ndarray
    g_val: jnp.ndarray


class QuadraticObjectiveData(NamedTuple):
    """Quadratic approximation data for a local objective model."""

    grad_f: jnp.ndarray
    Q: jnp.ndarray
    Q_diag: jnp.ndarray
    c: jnp.ndarray


class SQPSubproblemData(NamedTuple):
    """Structured data needed to solve a local SQP subproblem."""

    objective: QuadraticObjectiveData
    constraints: LinearizationData
    l: Optional[jnp.ndarray]
    u: Optional[jnp.ndarray]


def linearize_constraints_data(eq_fun, ineq_fun, jac_eq, jac_ineq, params, y):
    """Return matrix data for affine residual linearizations at ``y``."""
    A = jac_eq(y, params)
    h = eq_fun(params, y)
    b = A @ y - h

    C = jac_ineq(y, params)
    g = ineq_fun(params, y)
    d = C @ y - g

    return LinearizationData(A=A, b=b, C=C, d=d, h_val=h, g_val=g)


def linearize_constraints(eq_fun, ineq_fun, jac_eq, jac_ineq, params, y):
    """Return affine constraint linearizations as a dictionary."""
    data = linearize_constraints_data(
        eq_fun=eq_fun,
        ineq_fun=ineq_fun,
        jac_eq=jac_eq,
        jac_ineq=jac_ineq,
        params=params,
        y=y,
    )
    return dict(data._asdict())


def quadraticize_objective_data(
    grad_fun,
    hess_fun,
    diag_hess_fun,
    params,
    y,
    rho: float = 1.0,
    use_diagonal_hessian: bool = True,
    diag_floor: Optional[float] = None,
):
    """Return the quadratic approximation data for the objective at ``y``."""
    grad = grad_fun(y, params)

    if use_diagonal_hessian:
        qdiag = diag_hess_fun(params, y)
        if diag_floor is not None:
            qdiag = jnp.maximum(qdiag, diag_floor)
        Q = jnp.diag(rho * qdiag)
    else:
        Q = rho * hess_fun(y, params)

    c = grad - Q @ y

    return QuadraticObjectiveData(grad_f=grad, Q=Q, Q_diag=jnp.diag(Q), c=c)


def quadraticize_objective(
    grad_fun,
    hess_fun,
    diag_hess_fun,
    params,
    y,
    rho: float = 1.0,
    use_diagonal_hessian: bool = True,
    diag_floor: Optional[float] = None,
):
    """Return the quadratic approximation of the objective as a dictionary."""
    data = quadraticize_objective_data(
        grad_fun=grad_fun,
        hess_fun=hess_fun,
        diag_hess_fun=diag_hess_fun,
        params=params,
        y=y,
        rho=rho,
        use_diagonal_hessian=use_diagonal_hessian,
        diag_floor=diag_floor,
    )
    return dict(data._asdict())


def build_sqp_subproblem_data(
    eq_fun,
    ineq_fun,
    jac_eq,
    jac_ineq,
    grad_fun,
    hess_fun,
    diag_hess_fun,
    lower_fun,
    upper_fun,
    params,
    y,
    rho: float = 1.0,
    use_diagonal_hessian: bool = True,
    diag_floor: Optional[float] = None,
):
    """Build a structured SQP subproblem approximation at ``y``."""
    qdata = quadraticize_objective_data(
        grad_fun=grad_fun,
        hess_fun=hess_fun,
        diag_hess_fun=diag_hess_fun,
        params=params,
        y=y,
        rho=rho,
        use_diagonal_hessian=use_diagonal_hessian,
        diag_floor=diag_floor,
    )

    cdata = linearize_constraints_data(
        eq_fun=eq_fun,
        ineq_fun=ineq_fun,
        jac_eq=jac_eq,
        jac_ineq=jac_ineq,
        params=params,
        y=y,
    )

    return SQPSubproblemData(
        objective=qdata,
        constraints=cdata,
        l=None if lower_fun is None else lower_fun(params),
        u=None if upper_fun is None else upper_fun(params),
    )


def build_sqp_data(
    eq_fun,
    ineq_fun,
    jac_eq,
    jac_ineq,
    grad_fun,
    hess_fun,
    diag_hess_fun,
    lower_fun,
    upper_fun,
    params,
    y,
    rho: float = 1.0,
    use_diagonal_hessian: bool = True,
    diag_floor: Optional[float] = None,
):
    """Build SQP approximation data as a flat dictionary."""
    data = build_sqp_subproblem_data(
        eq_fun=eq_fun,
        ineq_fun=ineq_fun,
        jac_eq=jac_eq,
        jac_ineq=jac_ineq,
        grad_fun=grad_fun,
        hess_fun=hess_fun,
        diag_hess_fun=diag_hess_fun,
        lower_fun=lower_fun,
        upper_fun=upper_fun,
        params=params,
        y=y,
        rho=rho,
        use_diagonal_hessian=use_diagonal_hessian,
        diag_floor=diag_floor,
    )
    return {
        **dict(data.objective._asdict()),
        **dict(data.constraints._asdict()),
        "l": data.l,
        "u": data.u,
    }


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


def make_subproblem_layer_from_model(
    model: JaxNLPModel,
    *,
    param_name: str = "x",
    rho: float = 1.0,
    use_diagonal_hessian: bool = True,
    diag_floor: float = 1e-8,
):
    """Create a batched layer that extracts local SQP subproblem data."""
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
