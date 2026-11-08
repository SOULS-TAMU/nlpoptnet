"""Builder resources for the NLPOptNet frontend."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nlpoptnet.api.model import NLPOptNet

from nlpoptnet.api.formulation.bounds import BoundSpec
from typing import Callable
from nlpoptnet.api.formulation.expressions import Constant
from nlpoptnet.api.formulation.constraints import ConstraintEntry
from nlpoptnet.api.formulation.expressions import Expression
from typing import Iterable
from nlpoptnet.api.formulation.compiler import JaxNLPModel
from typing import Optional
from nlpoptnet.api.formulation.parameters import ParameterSpec
from pathlib import Path
from nlpoptnet.api.formulation.approximation import QuadraticObjective
from typing import Sequence
from typing import Union
from nlpoptnet.api.formulation.variables import VariableBuilder
from nlpoptnet.api.formulation.variables import VariableSpec
from nlpoptnet.api.formulation.expressions import VectorExpression
from nlpoptnet.api.formulation.expressions import _AffineParamScalar
from nlpoptnet.api.formulation.expressions import _AffineParamVector
from nlpoptnet.api.formulation.expressions import _ScalarBound
from nlpoptnet.api.formulation.expressions import _VectorBound
from nlpoptnet.api.configuration import _coerce_names
from nlpoptnet.api.configuration import _normalize_problem_type
from nlpoptnet.api.formulation.constraints import affine_eq_from_parts
from nlpoptnet.api.formulation.constraints import affine_ineq_from_parts
from dataclasses import dataclass
from nlpoptnet.api.formulation.constraints import eq_block
from nlpoptnet.api.formulation.constraints import eq_scalar
from dataclasses import field
from nlpoptnet.api.formulation.constraints import ineq_block
from nlpoptnet.api.formulation.constraints import ineq_scalar
import inspect
import jax.numpy as jnp
import numpy as np
from nlpoptnet.api.formulation.constraints import quadratic_eq_scalar
from nlpoptnet.api.formulation.constraints import quadratic_ineq_scalar
from ohagi._utils.serialization import resolve_path


@dataclass
class NLPBuilder:
    """Incrementally assemble a :class:`JaxNLPModel` from its components."""

    var_spec: VariableSpec
    parameter_spec: Optional[ParameterSpec] = None
    dtype: jnp.dtype = jnp.float64
    _objective_fun: Optional[Callable] = None
    _constraints: list[ConstraintEntry] = field(default_factory=list)
    _bounds: BoundSpec = field(default_factory=BoundSpec)

    def set_objective(self, objective_fun) -> "NLPBuilder":
        """Set the scalar objective callable or structured objective."""
        self._objective_fun = objective_fun
        return self

    def add_constraint(self, constraint: ConstraintEntry) -> "NLPBuilder":
        """Add one constraint entry to the model."""
        self._constraints.append(constraint)
        return self

    def add_constraints(self, constraints: Sequence[ConstraintEntry]) -> "NLPBuilder":
        """Add multiple constraint entries to the model."""
        self._constraints.extend(constraints)
        return self

    def add_eq_scalar(self, fun, name: str = "eq_scalar") -> "NLPBuilder":
        """Add a scalar equality residual."""
        return self.add_constraint(eq_scalar(fun, self.var_spec, name=name))

    def add_ineq_scalar(self, fun, name: str = "ineq_scalar") -> "NLPBuilder":
        """Add a scalar inequality residual."""
        return self.add_constraint(ineq_scalar(fun, self.var_spec, name=name))

    def add_eq_block(self, fun, name: str = "eq_block") -> "NLPBuilder":
        """Add a vector-valued equality block."""
        return self.add_constraint(eq_block(fun, self.var_spec, name=name))

    def add_ineq_block(self, fun, name: str = "ineq_block") -> "NLPBuilder":
        """Add a vector-valued inequality block."""
        return self.add_constraint(ineq_block(fun, self.var_spec, name=name))

    def set_bounds(
        self,
        lower_fun=None,
        upper_fun=None,
    ) -> "NLPBuilder":
        """Attach callable lower and upper bounds."""
        self._bounds = BoundSpec(lower_fun=lower_fun, upper_fun=upper_fun)
        return self

    def build(self, jit_compile: bool = True) -> JaxNLPModel:
        """Finalize and construct the :class:`JaxNLPModel`."""
        if self._objective_fun is None:
            raise ValueError("Objective function must be set before build().")

        return JaxNLPModel(
            var_spec=self.var_spec,
            objective_fun=self._objective_fun,
            constraints=self._constraints,
            bounds=self._bounds,
            parameter_spec=self.parameter_spec,
            dtype=self.dtype,
            jit_compile=jit_compile,
        )


@dataclass
class _AffineBound:
    """Store an affine parameter-to-bound mapping for one variable block."""

    param_name: str
    M: jnp.ndarray
    c: jnp.ndarray


class HighLevelNLPBuilder:
    r"""High-level modeling interface for parametric nonlinear programs.

    This builder assembles problems of the form

    .. math::

        \begin{aligned}
            \min_y \quad & f(y, x) \\
            \text{s.t.}\quad & h(y, x) = 0, \\
            & g(y, x) \le 0, \\
            & \ell(x) \le y \le u(x),
        \end{aligned}

    where :math:`x` denotes the parameter vector and :math:`y` denotes the
    decision variables.

    It provides a more convenient interface than the lower-level
    :class:`NLPBuilder` for assembling:

    - parameter blocks,
    - variable blocks,
    - objectives,
    - affine, quadratic, and nonlinear constraints,
    - affine lower and upper bounds.
    """

    def __init__(self, dtype=jnp.float64):
        """Initialize an empty high-level model builder."""
        self.dtype = dtype
        self._vb = VariableBuilder()
        self._param_sizes: dict[str, int] = {}
        self._objective_fun = None
        self._constraints = []
        self._lower_bounds: dict[str, _AffineBound] = {}
        self._upper_bounds: dict[str, _AffineBound] = {}

    def add_variable(self, name: str, size: int) -> "HighLevelNLPBuilder":
        """Register a named variable block."""
        self._vb.add_vector(name, size)
        return self

    def add_parameter(self, name: str, size: int) -> "HighLevelNLPBuilder":
        """Register a named parameter block."""
        if name in self._param_sizes:
            raise ValueError(f"Parameter '{name}' already exists")
        self._param_sizes[name] = int(size)
        return self

    def set_objective(self, objective_fun) -> "HighLevelNLPBuilder":
        """Set a callable scalar objective."""
        self._objective_fun = objective_fun
        return self

    def set_quadratic_objective(
        self,
        Q: jnp.ndarray,
        c: jnp.ndarray,
        constant: float = 0.0,
    ) -> "HighLevelNLPBuilder":
        """Set a structured quadratic objective."""
        self._objective_fun = QuadraticObjective(Q=Q, c=c, constant=constant)
        return self

    def add_affine_equality(
        self,
        *,
        var_name: str,
        A: jnp.ndarray,
        rhs_const: Union[jnp.ndarray, float] = 0.0,
        param_terms: Optional[list[tuple[jnp.ndarray, str]]] = None,
        name: str = "affine_eq",
    ) -> "HighLevelNLPBuilder":
        """Add an affine equality residual block."""
        param_terms = [] if param_terms is None else param_terms
        self._constraints.append(
            (
                "affine_eq",
                dict(
                    var_name=var_name,
                    A=A,
                    rhs_const=rhs_const,
                    param_terms=param_terms,
                    name=name,
                ),
            )
        )
        return self

    def add_affine_inequality(
        self,
        *,
        var_name: str,
        C: jnp.ndarray,
        rhs_const: Union[jnp.ndarray, float] = 0.0,
        param_terms: Optional[list[tuple[jnp.ndarray, str]]] = None,
        name: str = "affine_ineq",
    ) -> "HighLevelNLPBuilder":
        """Add an affine inequality residual block."""
        param_terms = [] if param_terms is None else param_terms
        self._constraints.append(
            (
                "affine_ineq",
                dict(
                    var_name=var_name,
                    C=C,
                    rhs_const=rhs_const,
                    param_terms=param_terms,
                    name=name,
                ),
            )
        )
        return self

    def add_nonlinear_equality(
        self,
        fun: Callable,
        name: str = "nonlinear_eq",
    ) -> "HighLevelNLPBuilder":
        """Add a nonlinear equality block."""
        self._constraints.append(("eq_block", dict(fun=fun, name=name)))
        return self

    def add_nonlinear_inequality(
        self,
        fun: Callable,
        name: str = "nonlinear_ineq",
    ) -> "HighLevelNLPBuilder":
        """Add a nonlinear inequality block."""
        self._constraints.append(("ineq_block", dict(fun=fun, name=name)))
        return self

    def add_quadratic_equality(
        self,
        *,
        Q: jnp.ndarray,
        c: jnp.ndarray,
        rhs_const: Union[jnp.ndarray, float] = 0.0,
        x_coeff: Optional[jnp.ndarray] = None,
        x_name: Optional[str] = None,
        name: str = "quadratic_eq",
    ) -> "HighLevelNLPBuilder":
        """Add a scalar quadratic equality residual."""
        self._constraints.append(
            (
                "quadratic_eq",
                dict(
                    Q=Q,
                    c=c,
                    rhs_const=rhs_const,
                    x_coeff=x_coeff,
                    x_name=x_name,
                    name=name,
                ),
            )
        )
        return self

    def add_quadratic_inequality(
        self,
        *,
        Q: jnp.ndarray,
        c: jnp.ndarray,
        rhs_const: Union[jnp.ndarray, float] = 0.0,
        x_coeff: Optional[jnp.ndarray] = None,
        x_name: Optional[str] = None,
        name: str = "quadratic_ineq",
    ) -> "HighLevelNLPBuilder":
        """Add a scalar quadratic inequality residual."""
        self._constraints.append(
            (
                "quadratic_ineq",
                dict(
                    Q=Q,
                    c=c,
                    rhs_const=rhs_const,
                    x_coeff=x_coeff,
                    x_name=x_name,
                    name=name,
                ),
            )
        )
        return self

    def set_affine_lower_bound(
        self,
        *,
        var_name: str,
        param_name: str,
        M: jnp.ndarray,
        c: jnp.ndarray,
    ) -> "HighLevelNLPBuilder":
        """Set an affine lower bound for a variable block."""
        self._lower_bounds[var_name] = _AffineBound(
            param_name=param_name, M=jnp.asarray(M), c=jnp.asarray(c)
        )
        return self

    def set_affine_upper_bound(
        self,
        *,
        var_name: str,
        param_name: str,
        M: jnp.ndarray,
        c: jnp.ndarray,
    ) -> "HighLevelNLPBuilder":
        """Set an affine upper bound for a variable block."""
        self._upper_bounds[var_name] = _AffineBound(
            param_name=param_name, M=jnp.asarray(M), c=jnp.asarray(c)
        )
        return self

    def _build_parameter_spec(
        self, example_params: dict[str, jnp.ndarray]
    ) -> ParameterSpec:
        """Validate example parameters and create a parameter specification."""
        missing = [k for k in self._param_sizes if k not in example_params]
        if missing:
            raise KeyError(f"Missing parameter examples for: {missing}")

        shapes = {}
        for name, size in self._param_sizes.items():
            shape = tuple(jnp.asarray(example_params[name]).shape)
            if shape != (size,):
                raise ValueError(
                    f"Parameter '{name}' expected shape {(size,)}, got {shape}"
                )
            shapes[name] = shape
        return ParameterSpec(names=list(self._param_sizes.keys()), shapes=shapes)

    def _build_bounds(self, var_spec: VariableSpec):
        """Build dense bound callables for all variable blocks."""
        if len(self._lower_bounds) == 0 and len(self._upper_bounds) == 0:
            return None, None

        for var_name, bnd in {**self._lower_bounds, **self._upper_bounds}.items():
            if var_name not in var_spec.shapes:
                raise KeyError(f"Unknown variable block '{var_name}' in bounds")
            n = var_spec.sizes[var_name]
            if bnd.M.shape[0] != n:
                raise ValueError(
                    f"Bound matrix for '{var_name}' must have {n} rows, got {bnd.M.shape[0]}"
                )
            if bnd.c.shape != (n,):
                raise ValueError(
                    f"Bound vector for '{var_name}' must have shape {(n,)}, got {bnd.c.shape}"
                )

        def lower_fun(params):
            blocks = {}
            for name in var_spec.names:
                n = var_spec.sizes[name]
                if name in self._lower_bounds:
                    bnd = self._lower_bounds[name]
                    blocks[name] = bnd.M @ jnp.ravel(params[bnd.param_name]) + bnd.c
                else:
                    blocks[name] = -jnp.inf * jnp.ones((n,), dtype=self.dtype)
            return var_spec.pack(blocks, dtype=self.dtype)

        def upper_fun(params):
            blocks = {}
            for name in var_spec.names:
                n = var_spec.sizes[name]
                if name in self._upper_bounds:
                    bnd = self._upper_bounds[name]
                    blocks[name] = bnd.M @ jnp.ravel(params[bnd.param_name]) + bnd.c
                else:
                    blocks[name] = jnp.inf * jnp.ones((n,), dtype=self.dtype)
            return var_spec.pack(blocks, dtype=self.dtype)

        return lower_fun, upper_fun

    def build(
        self, *, example_params: dict[str, jnp.ndarray], jit_compile: bool = True
    ):
        """Construct the final :class:`JaxNLPModel`."""
        if self._objective_fun is None:
            raise ValueError("Objective must be set before build().")

        var_spec = self._vb.build()
        parameter_spec = self._build_parameter_spec(example_params)
        lower_fun, upper_fun = self._build_bounds(var_spec)

        builder = NLPBuilder(
            var_spec=var_spec, parameter_spec=parameter_spec, dtype=self.dtype
        )
        builder.set_objective(self._objective_fun)

        for kind, payload in self._constraints:
            if kind == "affine_eq":
                builder.add_constraint(
                    affine_eq_from_parts(
                        var_spec=var_spec,
                        y_blocks=[(jnp.asarray(payload["A"]), payload["var_name"])],
                        rhs_const=payload["rhs_const"],
                        x_blocks=payload["param_terms"],
                        name=payload["name"],
                    )
                )
            elif kind == "affine_ineq":
                builder.add_constraint(
                    affine_ineq_from_parts(
                        var_spec=var_spec,
                        y_blocks=[(jnp.asarray(payload["C"]), payload["var_name"])],
                        rhs_const=payload["rhs_const"],
                        x_blocks=payload["param_terms"],
                        name=payload["name"],
                    )
                )
            elif kind == "eq_block":
                builder.add_constraint(
                    eq_block(payload["fun"], var_spec, name=payload["name"])
                )
            elif kind == "ineq_block":
                builder.add_constraint(
                    ineq_block(payload["fun"], var_spec, name=payload["name"])
                )
            elif kind == "quadratic_eq":
                builder.add_constraint(
                    quadratic_eq_scalar(
                        var_spec=var_spec,
                        Q=jnp.asarray(payload["Q"]),
                        c=jnp.asarray(payload["c"]),
                        rhs_const=payload["rhs_const"],
                        x_coeff=None
                        if payload["x_coeff"] is None
                        else jnp.asarray(payload["x_coeff"]),
                        x_name=payload["x_name"],
                        name=payload["name"],
                    )
                )
            elif kind == "quadratic_ineq":
                builder.add_constraint(
                    quadratic_ineq_scalar(
                        var_spec=var_spec,
                        Q=jnp.asarray(payload["Q"]),
                        c=jnp.asarray(payload["c"]),
                        rhs_const=payload["rhs_const"],
                        x_coeff=None
                        if payload["x_coeff"] is None
                        else jnp.asarray(payload["x_coeff"]),
                        x_name=payload["x_name"],
                        name=payload["name"],
                    )
                )
            else:
                raise ValueError(f"Unknown constraint kind: {kind}")

        if lower_fun is not None or upper_fun is not None:
            builder.set_bounds(lower_fun=lower_fun, upper_fun=upper_fun)

        return builder.build(jit_compile=jit_compile)


def add_parameter(self, names: str | Iterable[str]):
    """Register one or more parameter symbols and return the namespace."""
    for name in _coerce_names(names):
        if name in self.parameter_names or name in self.variable_names:
            raise ValueError(f"Duplicate symbol name '{name}'.")
        self.parameter_names.append(name)
    return self.parameter


def add_variable(self, names: str | Iterable[str]):
    """Register one or more decision-variable symbols and return the namespace."""
    for name in _coerce_names(names):
        if name in self.parameter_names or name in self.variable_names:
            raise ValueError(f"Duplicate symbol name '{name}'.")
        self.variable_names.append(name)
    return self.variable


def objective(self, value) -> "NLPOptNet":
    """Set the scalar objective from an expression or callable."""
    if callable(value):
        signature = inspect.signature(value)
        if len(signature.parameters) == 0:
            resolved = value()
            if isinstance(resolved, Expression):
                self._objective_expr = resolved
                self._objective_callable = None
                return self
        self._objective_callable = value
        self._objective_expr = None
        return self
    self._objective_expr = self._as_scalar_expr(value)
    self._objective_callable = None
    return self


def matrix(self, values) -> Constant:
    """Register a constant matrix and return its wrapper."""
    return self._register_constant(values)


def vector(self, values) -> Constant:
    """Register a constant vector and return its wrapper."""
    return self._register_constant(values)


def tensor(self, values) -> Constant:
    """Register a constant tensor and return its wrapper."""
    return self._register_constant(values)


def extract(self, path: str | Path) -> dict[str, Constant]:
    """Load constants from a ``problem.npz`` file into the model."""
    target = resolve_path(path)
    if not target.exists():
        raise FileNotFoundError(f"problem.npz not found: {target}")
    loaded = np.load(target, allow_pickle=False)
    extracted: dict[str, Constant] = {}
    for key in loaded.files:
        const = self._register_constant(loaded[key], name=key)
        extracted[key] = const
        if not hasattr(self, key):
            setattr(self, key, const)
    self._extracted_problem_path = str(target)
    if "problem_type" in extracted and self.problem_type is None:
        try:
            raw_type = str(np.asarray(extracted["problem_type"].value).item())
        except Exception:
            raw_type = str(np.asarray(extracted["problem_type"].value).reshape(-1)[0])
        self.problem_type = _normalize_problem_type(raw_type)
        if self.name == "nlpoptnet":
            self.name = str(self.problem_type)
    print(f"Loaded problem constants: {sorted(extracted.keys())}")
    return extracted


def lin(self, matrix, expr):
    """Form an affine expression ``A @ expr``."""
    const = self._ensure_constant(matrix)
    vector = self._as_vector_expr(expr)
    arr = np.asarray(const.value, dtype=np.float64)
    if arr.ndim == 1:
        affine = None
        if vector.affine_param is not None:
            affine = _AffineParamScalar(
                coeff=arr @ vector.affine_param.matrix,
                const=float(arr @ vector.affine_param.const),
            )
        return Expression(
            self,
            lambda ctx, a=arr, vec=vector: jnp.asarray(a) @ jnp.asarray(vec.eval(ctx)),
            f"lin({const.name}, {vector.text})",
            affine_param=affine,
        )
    if arr.ndim == 2:
        affine = None
        if vector.affine_param is not None:
            affine = _AffineParamVector(
                matrix=arr @ vector.affine_param.matrix,
                const=arr @ vector.affine_param.const,
            )
        return VectorExpression(
            self,
            lambda ctx, a=arr, vec=vector: jnp.asarray(a) @ jnp.asarray(vec.eval(ctx)),
            f"lin({const.name}, {vector.text})",
            size=arr.shape[0],
            affine_param=affine,
        )
    raise ValueError("lin(...) expects a 1D or 2D matrix/vector.")


def batch_lin(self, matrix, expr):
    """Form a batched affine vector expression."""
    const = self._ensure_constant(matrix)
    vector = self._as_vector_expr(expr)
    arr = np.asarray(const.value, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("batch_lin(...) expects a 2D matrix.")
    affine = None
    if vector.affine_param is not None:
        affine = _AffineParamVector(
            matrix=arr @ vector.affine_param.matrix,
            const=arr @ vector.affine_param.const,
        )
    return VectorExpression(
        self,
        lambda ctx, a=arr, vec=vector: jnp.asarray(a) @ jnp.asarray(vec.eval(ctx)),
        f"batch_lin({const.name}, {vector.text})",
        size=arr.shape[0],
        affine_param=affine,
    )


def quad(self, matrix, expr):
    """Form a quadratic scalar expression ``expr^T Q expr``."""
    const = self._ensure_constant(matrix)
    vector = self._as_vector_expr(expr)
    arr = np.asarray(const.value, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("quad(...) expects a 2D matrix.")
    return Expression(
        self,
        lambda ctx, a=arr, vec=vector: (
            jnp.ravel(vec.eval(ctx)) @ (jnp.asarray(a) @ jnp.ravel(vec.eval(ctx)))
        ),
        f"quad({const.name}, {vector.text})",
    )


def batch_quad(self, tensor, expr):
    """Form batched quadratic expressions from a rank-3 tensor."""
    const = self._ensure_constant(tensor)
    vector = self._as_vector_expr(expr)
    arr = np.asarray(const.value, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError("batch_quad(...) expects a 3D tensor.")
    return VectorExpression(
        self,
        lambda ctx, a=arr, vec=vector: jnp.einsum(
            "mij,i,j->m",
            jnp.asarray(a),
            jnp.ravel(vec.eval(ctx)),
            jnp.ravel(vec.eval(ctx)),
        ),
        f"batch_quad({const.name}, {vector.text})",
        size=arr.shape[0],
    )


def batch_exp(self, expr):
    """Apply elementwise exponential to a vector expression."""
    vector = self._as_vector_expr(expr)
    return VectorExpression(
        self,
        lambda ctx, vec=vector: jnp.exp(jnp.asarray(vec.eval(ctx))),
        f"batch_exp({vector.text})",
        size=vector.size,
    )


def sin(self, expr):
    """Apply elementwise sine to a scalar or vector expression."""
    return self._elementwise(expr, jnp.sin, "sin")


def cos(self, expr):
    """Apply elementwise cosine to a scalar or vector expression."""
    return self._elementwise(expr, jnp.cos, "cos")


def exp(self, expr):
    """Apply elementwise exponential to a scalar or vector expression."""
    return self._elementwise(expr, jnp.exp, "exp")


def log(self, expr):
    """Apply elementwise logarithm to a scalar or vector expression."""
    return self._elementwise(expr, jnp.log, "log")


def sqrt(self, expr):
    """Apply elementwise square root to a scalar or vector expression."""
    return self._elementwise(expr, jnp.sqrt, "sqrt")


def abs(self, expr):
    """Apply elementwise absolute value to a scalar or vector expression."""
    return self._elementwise(expr, jnp.abs, "abs")


def _build_box_bounds(self, *, n_x: int, n_y: int):
    # Default: unbounded variables
    lower_M = np.zeros((n_y, n_x), dtype=np.float64)
    lower_c = np.full((n_y,), -np.inf, dtype=np.float64)

    upper_M = np.zeros((n_y, n_x), dtype=np.float64)
    upper_c = np.full((n_y,), np.inf, dtype=np.float64)

    for entry in self.constraints.box.items:
        if isinstance(entry, _VectorBound):
            if entry.lower is not None:
                lower_M = np.asarray(entry.lower.matrix, dtype=np.float64)
                lower_c = np.asarray(entry.lower.const, dtype=np.float64)

            if entry.upper is not None:
                upper_M = np.asarray(entry.upper.matrix, dtype=np.float64)
                upper_c = np.asarray(entry.upper.const, dtype=np.float64)

        elif isinstance(entry, _ScalarBound):
            if entry.lower is not None:
                lower_M[entry.index] = np.asarray(entry.lower.coeff, dtype=np.float64)
                lower_c[entry.index] = float(entry.lower.const)

            if entry.upper is not None:
                upper_M[entry.index] = np.asarray(entry.upper.coeff, dtype=np.float64)
                upper_c[entry.index] = float(entry.upper.const)

    return lower_M, lower_c, upper_M, upper_c
