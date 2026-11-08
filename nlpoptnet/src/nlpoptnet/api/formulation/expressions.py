"""Expressions resources for the NLPOptNet frontend."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nlpoptnet.api.model import NLPOptNet

from typing import Any
from typing import Callable
from dataclasses import dataclass
import inspect
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class _AffineParamScalar:
    coeff: np.ndarray
    const: float


@dataclass(frozen=True)
class _AffineParamVector:
    matrix: np.ndarray
    const: np.ndarray


@dataclass(frozen=True)
class Constraint:
    """Symbolic equality or inequality formed from two scalar expressions."""

    lhs: Any
    rhs: Any
    kind: str
    op: str
    text: str

    def residual(self, ctx):
        """Evaluate the signed constraint residual under the expression context."""
        if self.kind == "eq":
            return self.lhs.eval(ctx) - self.rhs.eval(ctx)
        if self.op == "<=":
            return self.lhs.eval(ctx) - self.rhs.eval(ctx)
        return self.rhs.eval(ctx) - self.lhs.eval(ctx)


@dataclass(frozen=True)
class _BlockConstraint:
    fn: Callable
    kind: str
    text: str
    serializable: bool = False


@dataclass(frozen=True)
class _ScalarBound:
    index: int
    lower: _AffineParamScalar | None = None
    upper: _AffineParamScalar | None = None


@dataclass(frozen=True)
class _VectorBound:
    lower: _AffineParamVector | None = None
    upper: _AffineParamVector | None = None


@dataclass(frozen=True)
class _EvalContext:
    y: jnp.ndarray
    x: jnp.ndarray
    variable_index: dict[str, int]
    parameter_index: dict[str, int]

    def value(self, kind: str, name: str):
        """Resolve a named symbol from the current parameter and variable context."""
        if kind == "variable":
            return self.y[self.variable_index[name]]
        if kind == "parameter":
            return self.x[self.parameter_index[name]]
        raise KeyError(f"Unknown symbol kind '{kind}'.")


class Expression:
    """Scalar symbolic expression tracked by :class:`NLPOptNet`."""

    def __init__(
        self,
        owner: "NLPOptNet",
        fn: Callable,
        text: str,
        *,
        affine_param: _AffineParamScalar | None = None,
        ref_kind: str | None = None,
        ref_name: str | None = None,
    ) -> None:
        self._owner = owner
        self._fn = fn
        self.text = str(text)
        self.affine_param = affine_param
        self.ref_kind = ref_kind
        self.ref_name = ref_name

    def eval(self, ctx):
        """Evaluate this symbolic scalar expression under the supplied context."""
        return self._fn(ctx)

    def _binary(self, other, op, symbol: str) -> "Expression":
        rhs = self._owner._as_scalar_expr(other)
        affine = self._owner._combine_scalar_affine(
            self.affine_param, rhs.affine_param, symbol
        )
        return Expression(
            self._owner,
            lambda ctx, lhs=self, rhs=rhs: op(lhs.eval(ctx), rhs.eval(ctx)),
            f"({self.text} {symbol} {rhs.text})",
            affine_param=affine,
        )

    def _rbinary(self, other, op, symbol: str) -> "Expression":
        lhs = self._owner._as_scalar_expr(other)
        affine = self._owner._combine_scalar_affine(
            lhs.affine_param, self.affine_param, symbol
        )
        return Expression(
            self._owner,
            lambda ctx, lhs=lhs, rhs=self: op(lhs.eval(ctx), rhs.eval(ctx)),
            f"({lhs.text} {symbol} {self.text})",
            affine_param=affine,
        )

    def __add__(self, other):
        return self._binary(other, lambda a, b: a + b, "+")

    def __radd__(self, other):
        return self._rbinary(other, lambda a, b: a + b, "+")

    def __sub__(self, other):
        return self._binary(other, lambda a, b: a - b, "-")

    def __rsub__(self, other):
        return self._rbinary(other, lambda a, b: a - b, "-")

    def __mul__(self, other):
        rhs = self._owner._as_scalar_expr(other)
        affine = None
        if self.affine_param is not None and rhs.affine_param is not None:
            affine = None
        elif (
            self.affine_param is not None
            and rhs.affine_param is None
            and rhs.text.replace(".", "", 1).replace("-", "", 1).isdigit()
        ):
            affine = _AffineParamScalar(
                coeff=float(rhs.text) * self.affine_param.coeff,
                const=float(rhs.text) * self.affine_param.const,
            )
        elif (
            rhs.affine_param is not None
            and self.affine_param is None
            and self.text.replace(".", "", 1).replace("-", "", 1).isdigit()
        ):
            affine = _AffineParamScalar(
                coeff=float(self.text) * rhs.affine_param.coeff,
                const=float(self.text) * rhs.affine_param.const,
            )
        return Expression(
            self._owner,
            lambda ctx, lhs=self, rhs=rhs: lhs.eval(ctx) * rhs.eval(ctx),
            f"({self.text} * {rhs.text})",
            affine_param=affine,
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        rhs = self._owner._as_scalar_expr(other)
        return Expression(
            self._owner,
            lambda ctx, lhs=self, rhs=rhs: lhs.eval(ctx) / rhs.eval(ctx),
            f"({self.text} / {rhs.text})",
        )

    def __rtruediv__(self, other):
        lhs = self._owner._as_scalar_expr(other)
        return Expression(
            self._owner,
            lambda ctx, lhs=lhs, rhs=self: lhs.eval(ctx) / rhs.eval(ctx),
            f"({lhs.text} / {self.text})",
        )

    def __pow__(self, other):
        rhs = self._owner._as_scalar_expr(other)
        return Expression(
            self._owner,
            lambda ctx, lhs=self, rhs=rhs: lhs.eval(ctx) ** rhs.eval(ctx),
            f"({self.text} ** {rhs.text})",
        )

    def __neg__(self):
        affine = None
        if self.affine_param is not None:
            affine = _AffineParamScalar(
                coeff=-self.affine_param.coeff, const=-self.affine_param.const
            )
        return Expression(
            self._owner,
            lambda ctx, expr=self: -expr.eval(ctx),
            f"(-{self.text})",
            affine_param=affine,
        )

    def __le__(self, other):
        rhs = self._owner._as_scalar_expr(other)
        return Constraint(self, rhs, "ineq", "<=", f"{self.text} <= {rhs.text}")

    def __ge__(self, other):
        rhs = self._owner._as_scalar_expr(other)
        return Constraint(self, rhs, "ineq", ">=", f"{self.text} >= {rhs.text}")

    def __eq__(self, other):  # type: ignore[override]
        rhs = self._owner._as_scalar_expr(other)
        return Constraint(self, rhs, "eq", "==", f"{self.text} == {rhs.text}")


class VectorExpression:
    """Vector-valued symbolic expression tracked by :class:`NLPOptNet`."""

    def __init__(
        self,
        owner: "NLPOptNet",
        fn: Callable,
        text: str,
        *,
        size: int,
        affine_param: _AffineParamVector | None = None,
        ref_kind: str | None = None,
    ) -> None:
        self._owner = owner
        self._fn = fn
        self.text = str(text)
        self.size = int(size)
        self.affine_param = affine_param
        self.ref_kind = ref_kind

    def eval(self, ctx):
        """Evaluate and stack the vector's scalar expressions."""
        return self._fn(ctx)

    def _binary(self, other, op, symbol: str) -> "VectorExpression":
        rhs = self._owner._as_vector_expr(other, size=self.size)
        affine = self._owner._combine_vector_affine(
            self.affine_param, rhs.affine_param, symbol
        )
        return VectorExpression(
            self._owner,
            lambda ctx, lhs=self, rhs=rhs: op(lhs.eval(ctx), rhs.eval(ctx)),
            f"({self.text} {symbol} {rhs.text})",
            size=self.size,
            affine_param=affine,
        )

    def _rbinary(self, other, op, symbol: str) -> "VectorExpression":
        lhs = self._owner._as_vector_expr(other, size=self.size)
        affine = self._owner._combine_vector_affine(
            lhs.affine_param, self.affine_param, symbol
        )
        return VectorExpression(
            self._owner,
            lambda ctx, lhs=lhs, rhs=self: op(lhs.eval(ctx), rhs.eval(ctx)),
            f"({lhs.text} {symbol} {self.text})",
            size=self.size,
            affine_param=affine,
        )

    def __add__(self, other):
        return self._binary(other, lambda a, b: a + b, "+")

    def __radd__(self, other):
        return self._rbinary(other, lambda a, b: a + b, "+")

    def __sub__(self, other):
        return self._binary(other, lambda a, b: a - b, "-")

    def __rsub__(self, other):
        return self._rbinary(other, lambda a, b: a - b, "-")

    def __mul__(self, other):
        rhs = self._owner._as_scalar_expr(other)
        affine = None
        if (
            self.affine_param is not None
            and rhs.affine_param is None
            and rhs.text.replace(".", "", 1).replace("-", "", 1).isdigit()
        ):
            scalar = float(rhs.text)
            affine = _AffineParamVector(
                matrix=scalar * self.affine_param.matrix,
                const=scalar * self.affine_param.const,
            )
        return VectorExpression(
            self._owner,
            lambda ctx, lhs=self, rhs=rhs: lhs.eval(ctx) * rhs.eval(ctx),
            f"({self.text} * {rhs.text})",
            size=self.size,
            affine_param=affine,
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        rhs = self._owner._as_scalar_expr(other)
        return VectorExpression(
            self._owner,
            lambda ctx, lhs=self, rhs=rhs: lhs.eval(ctx) / rhs.eval(ctx),
            f"({self.text} / {rhs.text})",
            size=self.size,
        )

    def __neg__(self):
        affine = None
        if self.affine_param is not None:
            affine = _AffineParamVector(
                matrix=-self.affine_param.matrix,
                const=-self.affine_param.const,
            )
        return VectorExpression(
            self._owner,
            lambda ctx, expr=self: -expr.eval(ctx),
            f"(-{self.text})",
            size=self.size,
            affine_param=affine,
        )

    def __le__(self, other):
        rhs = self._owner._as_vector_expr(other, size=self.size)
        return Constraint(self, rhs, "ineq", "<=", f"{self.text} <= {rhs.text}")

    def __ge__(self, other):
        rhs = self._owner._as_vector_expr(other, size=self.size)
        return Constraint(self, rhs, "ineq", ">=", f"{self.text} >= {rhs.text}")

    def __eq__(self, other):  # type: ignore[override]
        rhs = self._owner._as_vector_expr(other, size=self.size)
        return Constraint(self, rhs, "eq", "==", f"{self.text} == {rhs.text}")


class Constant:
    """Named constant array registered inside an :class:`NLPOptNet` model."""

    def __init__(self, owner: "NLPOptNet", name: str, value: Any) -> None:
        self._owner = owner
        self.name = str(name)
        self.value = np.asarray(value)

    def __array__(self, dtype=None):
        if dtype is None:
            return np.asarray(self.value)
        return np.asarray(self.value, dtype=dtype)

    def __matmul__(self, other):
        return np.asarray(self.value) @ other

    def __rmatmul__(self, other):
        return other @ np.asarray(self.value)

    def _expr(self):
        if self.value.ndim == 0:
            return self._owner._constant_to_scalar_expr(self)
        if self.value.ndim == 1:
            return self._owner._constant_to_vector_expr(self)
        raise TypeError(
            f"Constant '{self.name}' is not directly usable as a scalar/vector expression."
        )

    def __add__(self, other):
        return self._expr() + other

    def __radd__(self, other):
        return other + self._expr()

    def __sub__(self, other):
        return self._expr() - other

    def __rsub__(self, other):
        return other - self._expr()

    def __mul__(self, other):
        return self._expr() * other

    def __rmul__(self, other):
        return other * self._expr()

    def __truediv__(self, other):
        return self._expr() / other

    def __rtruediv__(self, other):
        return other / self._expr()

    def __neg__(self):
        return -self._expr()

    def __le__(self, other):
        return self._expr() <= other

    def __ge__(self, other):
        return self._expr() >= other

    def __eq__(self, other):  # type: ignore[override]
        return self._expr() == other

    def __repr__(self) -> str:
        return f"Constant(name={self.name!r}, shape={tuple(self.value.shape)!r})"


class _SymbolNamespace:
    """Ordered symbolic parameter or variable namespace with vector access."""

    def __init__(self, owner: "NLPOptNet", kind: str) -> None:
        self._owner = owner
        self._kind = kind

    def __getattr__(self, name: str) -> Expression:
        return self[name]

    def __getitem__(self, name: str) -> Expression:
        self._owner._check_symbol(self._kind, name)
        if self._kind == "parameter":
            idx = self._owner.parameter_names.index(name)
            coeff = np.zeros((len(self._owner.parameter_names),), dtype=np.float64)
            coeff[idx] = 1.0
            affine = _AffineParamScalar(coeff=coeff, const=0.0)
        else:
            affine = None
        return Expression(
            self._owner,
            lambda ctx, kind=self._kind, n=name: ctx.value(kind, n),
            str(name),
            affine_param=affine,
            ref_kind=self._kind,
            ref_name=str(name),
        )

    def vector(self) -> VectorExpression:
        """Create a symbolic vector from selected namespace entries."""
        names = (
            self._owner.parameter_names
            if self._kind == "parameter"
            else self._owner.variable_names
        )
        if self._kind == "parameter":
            affine = _AffineParamVector(
                matrix=np.eye(len(names), dtype=np.float64),
                const=np.zeros((len(names),), dtype=np.float64),
            )
        else:
            affine = None
        token = "x" if self._kind == "parameter" else "y"
        return VectorExpression(
            self._owner,
            lambda ctx, kind=self._kind, ordered=list(names): jnp.asarray(
                [ctx.value(kind, name) for name in ordered],
                dtype=ctx.x.dtype if kind == "parameter" else ctx.y.dtype,
            ),
            token,
            size=len(names),
            affine_param=affine,
            ref_kind=f"{self._kind}_vector",
        )


class _ConstraintGroup:
    def __init__(self, owner: "NLPOptNet", kind: str) -> None:
        self._owner = owner
        self.kind = str(kind)
        self.items: list[Any] = []

    def add(self, *items) -> "_ConstraintGroup":
        """Register one or more constraints in this equality/inequality group."""
        for item in items:
            normalized = self._normalize_item(item)
            self.items.append(normalized)
        return self

    def _normalize_item(self, item):
        if isinstance(item, Constraint):
            if item.kind != self.kind:
                raise ValueError(f"Expected a {self.kind} constraint, got {item.kind}.")
            return item
        if isinstance(item, VectorExpression):
            zero = self._owner._zero_vector(item.size)
            op = "==" if self.kind == "eq" else "<="
            return Constraint(
                item, zero, self.kind, op, f"{item.text} {op} {zero.text}"
            )
        if isinstance(item, Expression):
            zero = self._owner._zero_scalar()
            op = "==" if self.kind == "eq" else "<="
            return Constraint(
                item, zero, self.kind, op, f"{item.text} {op} {zero.text}"
            )
        if callable(item):
            signature = inspect.signature(item)
            if len(signature.parameters) == 0:
                return self._normalize_item(item())
            return _BlockConstraint(
                fn=item,
                kind=self.kind,
                text=getattr(item, "__name__", f"{self.kind}_block"),
                serializable=False,
            )
        raise TypeError(
            f"Unsupported constraint item for {self.kind}: {type(item).__name__}"
        )


class _BoxConstraintGroup:
    """Collector translating symbolic bound constraints into variable box metadata."""

    def __init__(self, owner: "NLPOptNet") -> None:
        self._owner = owner
        self.items: list[Any] = []

    def add(self, *items, var=None, lower=None, upper=None) -> "_BoxConstraintGroup":
        """Register box bounds from the supplied symbolic constraints."""
        if var is not None or lower is not None or upper is not None:
            self.items.append(self._from_explicit(var=var, lower=lower, upper=upper))
        for item in items:
            self.items.append(self._from_constraint(item))
        return self

    def _from_explicit(self, *, var, lower, upper):
        if var is None:
            raise ValueError("box.add(var=..., lower=..., upper=...) requires var.")
        var_expr = self._owner._as_vector_expr(var)
        if var_expr.ref_kind != "variable_vector":
            raise ValueError(
                "box.add(..., var=...) expects the variable namespace returned by add_variable(...)."
            )
        lower_affine = (
            None
            if lower is None
            else self._owner._as_affine_vector(lower, size=var_expr.size)
        )
        upper_affine = (
            None
            if upper is None
            else self._owner._as_affine_vector(upper, size=var_expr.size)
        )
        return _VectorBound(lower=lower_affine, upper=upper_affine)

    def _from_constraint(self, item):
        if not isinstance(item, Constraint) or item.kind != "ineq":
            raise TypeError(
                "constraints.box.add(...) expects bound inequalities or explicit var/lower/upper arguments."
            )

        if item.lhs.ref_kind == "variable":
            index = self._owner.variable_names.index(str(item.lhs.ref_name))
            other = self._owner._as_scalar_expr(item.rhs)
            if other.affine_param is None:
                raise ValueError("Box bounds must be affine in parameters.")
            if item.op == "<=":
                return _ScalarBound(index=index, upper=other.affine_param)
            return _ScalarBound(index=index, lower=other.affine_param)

        if item.rhs.ref_kind == "variable":
            index = self._owner.variable_names.index(str(item.rhs.ref_name))
            other = self._owner._as_scalar_expr(item.lhs)
            if other.affine_param is None:
                raise ValueError("Box bounds must be affine in parameters.")
            if item.op == "<=":
                return _ScalarBound(index=index, lower=other.affine_param)
            return _ScalarBound(index=index, upper=other.affine_param)

        raise ValueError(
            "Box constraints must compare a single variable against an affine parameter expression."
        )


class _ConstraintManager:
    def __init__(self, owner: "NLPOptNet") -> None:
        self.equality = _ConstraintGroup(owner, "eq")
        self.inequality = _ConstraintGroup(owner, "ineq")
        self.box = _BoxConstraintGroup(owner)


def _elementwise(self, expr, fn, name: str):
    if isinstance(expr, (VectorExpression, _SymbolNamespace)):
        vector = self._as_vector_expr(expr)
        return VectorExpression(
            self,
            lambda ctx, e=vector: fn(e.eval(ctx)),
            f"{name}({vector.text})",
            size=vector.size,
        )
    scalar = self._as_scalar_expr(expr)
    return Expression(
        self,
        lambda ctx, e=scalar: fn(e.eval(ctx)),
        f"{name}({scalar.text})",
    )


def _register_constant(self, value, *, name: str | None = None) -> Constant:
    if name is None:
        name = f"_const_{self._constant_counter}"
        self._constant_counter += 1
    arr = np.asarray(value)
    self._constants[str(name)] = arr
    return Constant(self, str(name), arr)


def _ensure_constant(self, value) -> Constant:
    if isinstance(value, Constant):
        return value
    return self._register_constant(value)


def _constant_to_scalar_expr(self, const: Constant) -> Expression:
    arr = np.asarray(const.value)
    if arr.ndim != 0:
        raise ValueError(f"Constant '{const.name}' is not scalar.")
    return Expression(
        self,
        lambda _ctx, value=float(arr): jnp.asarray(value),
        const.name,
        affine_param=_AffineParamScalar(
            coeff=np.zeros((len(self.parameter_names),), dtype=np.float64),
            const=float(arr),
        ),
    )


def _constant_to_vector_expr(self, const: Constant) -> VectorExpression:
    arr = np.asarray(const.value, dtype=np.float64).reshape(-1)
    return VectorExpression(
        self,
        lambda _ctx, value=arr: jnp.asarray(value),
        const.name,
        size=arr.size,
        affine_param=_AffineParamVector(
            matrix=np.zeros((arr.size, len(self.parameter_names)), dtype=np.float64),
            const=arr,
        ),
    )


def _zero_scalar(self) -> Expression:
    return Expression(
        self,
        lambda _ctx: jnp.asarray(0.0),
        "0.0",
        affine_param=_AffineParamScalar(
            coeff=np.zeros((len(self.parameter_names),), dtype=np.float64),
            const=0.0,
        ),
    )


def _zero_vector(self, size: int) -> VectorExpression:
    zeros = np.zeros((int(size),), dtype=np.float64)
    return VectorExpression(
        self,
        lambda _ctx, value=zeros: jnp.asarray(value),
        f"jnp.zeros(({int(size)},))",
        size=int(size),
        affine_param=_AffineParamVector(
            matrix=np.zeros((int(size), len(self.parameter_names)), dtype=np.float64),
            const=zeros,
        ),
    )


def _as_scalar_expr(self, value) -> Expression:
    if isinstance(value, Expression):
        return value
    if isinstance(value, Constant):
        return self._constant_to_scalar_expr(value)
    arr = np.asarray(value)
    if arr.ndim == 0:
        return Expression(
            self,
            lambda _ctx, scalar=float(arr): jnp.asarray(scalar),
            repr(float(arr)),
            affine_param=_AffineParamScalar(
                coeff=np.zeros((len(self.parameter_names),), dtype=np.float64),
                const=float(arr),
            ),
        )
    raise TypeError(f"Expected a scalar expression, got {type(value).__name__}.")


def _as_vector_expr(self, value, *, size: int | None = None) -> VectorExpression:
    if isinstance(value, VectorExpression):
        if size is not None and value.size != int(size):
            raise ValueError(
                f"Vector shape mismatch: expected size {size}, got {value.size}."
            )
        return value
    if isinstance(value, _SymbolNamespace):
        vector = value.vector()
        if size is not None and vector.size != int(size):
            raise ValueError(
                f"Vector shape mismatch: expected size {size}, got {vector.size}."
            )
        return vector
    if isinstance(value, Expression):
        if size is None:
            raise TypeError(
                "Cannot broadcast a scalar expression to a vector without a target size."
            )
        return VectorExpression(
            self,
            lambda ctx, expr=value, count=int(size): jnp.broadcast_to(
                expr.eval(ctx), (count,)
            ),
            value.text,
            size=int(size),
            affine_param=None
            if value.affine_param is None
            else _AffineParamVector(
                matrix=np.broadcast_to(
                    value.affine_param.coeff[None, :],
                    (int(size), len(self.parameter_names)),
                ).copy(),
                const=np.full(
                    (int(size),), float(value.affine_param.const), dtype=np.float64
                ),
            ),
        )
    if isinstance(value, Constant):
        vector = self._constant_to_vector_expr(value)
        if size is not None and vector.size != int(size):
            if vector.size == 1:
                return self._as_vector_expr(
                    self._as_scalar_expr(float(vector.affine_param.const[0])), size=size
                )
            raise ValueError(
                f"Vector shape mismatch: expected size {size}, got {vector.size}."
            )
        return vector
    arr = np.asarray(value)
    if arr.ndim == 0:
        if size is None:
            raise TypeError(
                "Cannot broadcast a scalar constant to a vector without a target size."
            )
        return self._as_vector_expr(self._as_scalar_expr(float(arr)), size=size)
    if arr.ndim == 1:
        const = self._register_constant(arr)
        vector = self._constant_to_vector_expr(const)
        if size is not None and vector.size != int(size):
            raise ValueError(
                f"Vector shape mismatch: expected size {size}, got {vector.size}."
            )
        return vector
    raise TypeError(f"Expected a vector expression, got {type(value).__name__}.")


def _as_affine_vector(self, value, *, size: int) -> _AffineParamVector:
    vector = self._as_vector_expr(value, size=size)
    if vector.affine_param is None:
        raise ValueError("Expected an affine parameter expression.")
    return vector.affine_param


def _combine_scalar_affine(
    self,
    lhs: _AffineParamScalar | None,
    rhs: _AffineParamScalar | None,
    symbol: str,
) -> _AffineParamScalar | None:
    if lhs is None or rhs is None:
        return None
    if symbol == "+":
        return _AffineParamScalar(
            coeff=lhs.coeff + rhs.coeff, const=lhs.const + rhs.const
        )
    if symbol == "-":
        return _AffineParamScalar(
            coeff=lhs.coeff - rhs.coeff, const=lhs.const - rhs.const
        )
    return None


def _combine_vector_affine(
    self,
    lhs: _AffineParamVector | None,
    rhs: _AffineParamVector | None,
    symbol: str,
) -> _AffineParamVector | None:
    if lhs is None or rhs is None:
        return None
    if symbol == "+":
        return _AffineParamVector(
            matrix=lhs.matrix + rhs.matrix, const=lhs.const + rhs.const
        )
    if symbol == "-":
        return _AffineParamVector(
            matrix=lhs.matrix - rhs.matrix, const=lhs.const - rhs.const
        )
    return None


def _check_symbol(self, kind: str, name: str) -> None:
    pool = self.parameter_names if kind == "parameter" else self.variable_names
    if name not in pool:
        raise AttributeError(f"Unknown {kind} '{name}'.")
