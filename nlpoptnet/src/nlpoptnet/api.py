from __future__ import annotations

import copy
import inspect
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import jax
import jax.numpy as jnp
import numpy as np

from jaxmodel import HighLevelNLPBuilder
from opt.models.backbone import Backbone
from opt.training import (
    apply_projection_layers,
    build_epoch_fns,
    build_train_fns_from_jaxmodel,
    build_violation_fn_from_jaxmodel,
    cfg_from_dict,
    make_batched_objective,
    make_fixed_batches,
    make_subproblem_layer_from_model,
    warmup_compile,
)
from .sampling import (
    hit_and_run_samples,
    load_csv_matrix,
    sample_box,
    split_train_val,
    write_csv_matrix,
)
from .serialization import build_model_from_problem_spec
from .utils import json_safe, resolve_dtype, resolve_path, timestamp, write_json

jax.config.update("jax_enable_x64", True)


_METRIC_KEYS = ("loss", "obj", "mse_y", "mse_lam", "mse_mu")
_TYPE_ALIASES = {"nonconvx": "nonconvex"}
_DEFAULT_CONFIG = {
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "train_frac": 0.8,
    "hidden_size": 64,
    "hidden_layers": 2,
    "seed": 42,
    "dtype": "float64",
    "print_every": 10,
    "verbose": True,
    "alpha_consistency": 1.0,
    "cp_iters": 5000,
    "cp_tol": 1e-6,
    "IS_FIXED": True,
    "stepsize": "auto",
    "safety": 0.90,
    "knorm_iters": 20,
    "knorm_seed": 42,
    "adjoint_iters": 50,
    "use_ruiz": True,
    "ruiz_iters": 4,
    "k_layer": 1,
    "device": "auto",
    "jit_warmup": True,
    "num_samples": 1000,
    "y_bound": 10.0,
}


def _normalize_problem_type(problem_type: str | None) -> str | None:
    if problem_type is None:
        return None
    normalized = _TYPE_ALIASES.get(str(problem_type).strip().lower(), str(problem_type).strip().lower())
    return normalized


def _coerce_names(names: str | Iterable[str]) -> list[str]:
    if isinstance(names, str):
        out = [names]
    else:
        out = [str(name) for name in names]
    if not out:
        raise ValueError("At least one name is required.")
    if len(set(out)) != len(out):
        raise ValueError(f"Names must be unique, got {out}.")
    return out


def _metrics_tuple_to_dict(values) -> dict[str, float]:
    return {key: float(value) for key, value in zip(_METRIC_KEYS, values)}


def _block_until_ready(tree):
    return jax.tree_util.tree_map(
        lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
        tree,
    )


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
    lhs: Any
    rhs: Any
    kind: str
    op: str
    text: str

    def residual(self, ctx):
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
        if kind == "variable":
            return self.y[self.variable_index[name]]
        if kind == "parameter":
            return self.x[self.parameter_index[name]]
        raise KeyError(f"Unknown symbol kind '{kind}'.")


@dataclass
class _BuildState:
    model: Any
    cfg: Any
    train_config: dict[str, Any]
    dtype: Any
    X: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    train_batches: Any
    val_batches: Any
    backbone: Any
    init_state_fn: Callable
    train_step_fn: Callable
    eval_step_fn: Callable
    train_epoch_fn: Callable
    eval_epoch_fn: Callable
    violation_fn: Callable
    objective_fn: Callable
    sub_layer: Callable


@dataclass
class _InferenceState:
    model: Any
    cfg: Any
    train_config: dict[str, Any]
    params: Any
    backbone: Any
    sub_layer: Callable
    n_x: int
    output_dir: str | None


class Expression:
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
        return self._fn(ctx)

    def _binary(self, other, op, symbol: str) -> "Expression":
        rhs = self._owner._as_scalar_expr(other)
        affine = self._owner._combine_scalar_affine(self.affine_param, rhs.affine_param, symbol)
        return Expression(
            self._owner,
            lambda ctx, lhs=self, rhs=rhs: op(lhs.eval(ctx), rhs.eval(ctx)),
            f"({self.text} {symbol} {rhs.text})",
            affine_param=affine,
        )

    def _rbinary(self, other, op, symbol: str) -> "Expression":
        lhs = self._owner._as_scalar_expr(other)
        affine = self._owner._combine_scalar_affine(lhs.affine_param, self.affine_param, symbol)
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
        elif self.affine_param is not None and rhs.affine_param is None and rhs.text.replace(".", "", 1).replace("-", "", 1).isdigit():
            affine = _AffineParamScalar(
                coeff=float(rhs.text) * self.affine_param.coeff,
                const=float(rhs.text) * self.affine_param.const,
            )
        elif rhs.affine_param is not None and self.affine_param is None and self.text.replace(".", "", 1).replace("-", "", 1).isdigit():
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
            affine = _AffineParamScalar(coeff=-self.affine_param.coeff, const=-self.affine_param.const)
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
        return self._fn(ctx)

    def _binary(self, other, op, symbol: str) -> "VectorExpression":
        rhs = self._owner._as_vector_expr(other, size=self.size)
        affine = self._owner._combine_vector_affine(self.affine_param, rhs.affine_param, symbol)
        return VectorExpression(
            self._owner,
            lambda ctx, lhs=self, rhs=rhs: op(lhs.eval(ctx), rhs.eval(ctx)),
            f"({self.text} {symbol} {rhs.text})",
            size=self.size,
            affine_param=affine,
        )

    def _rbinary(self, other, op, symbol: str) -> "VectorExpression":
        lhs = self._owner._as_vector_expr(other, size=self.size)
        affine = self._owner._combine_vector_affine(lhs.affine_param, self.affine_param, symbol)
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
        if self.affine_param is not None and rhs.affine_param is None and rhs.text.replace(".", "", 1).replace("-", "", 1).isdigit():
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
        raise TypeError(f"Constant '{self.name}' is not directly usable as a scalar/vector expression.")

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
        names = self._owner.parameter_names if self._kind == "parameter" else self._owner.variable_names
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
            return Constraint(item, zero, self.kind, op, f"{item.text} {op} {zero.text}")
        if isinstance(item, Expression):
            zero = self._owner._zero_scalar()
            op = "==" if self.kind == "eq" else "<="
            return Constraint(item, zero, self.kind, op, f"{item.text} {op} {zero.text}")
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
        raise TypeError(f"Unsupported constraint item for {self.kind}: {type(item).__name__}")


class _BoxConstraintGroup:
    def __init__(self, owner: "NLPOptNet") -> None:
        self._owner = owner
        self.items: list[Any] = []

    def add(self, *items, var=None, lower=None, upper=None) -> "_BoxConstraintGroup":
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
            raise ValueError("box.add(..., var=...) expects the variable namespace returned by add_variable(...).")
        lower_affine = None if lower is None else self._owner._as_affine_vector(lower, size=var_expr.size)
        upper_affine = None if upper is None else self._owner._as_affine_vector(upper, size=var_expr.size)
        return _VectorBound(lower=lower_affine, upper=upper_affine)

    def _from_constraint(self, item):
        if not isinstance(item, Constraint) or item.kind != "ineq":
            raise TypeError("constraints.box.add(...) expects bound inequalities or explicit var/lower/upper arguments.")

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

        raise ValueError("Box constraints must compare a single variable against an affine parameter expression.")


class _ConstraintManager:
    def __init__(self, owner: "NLPOptNet") -> None:
        self.equality = _ConstraintGroup(owner, "eq")
        self.inequality = _ConstraintGroup(owner, "ineq")
        self.box = _BoxConstraintGroup(owner)


class NLPOptNet:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        type: str | None = None,
        name: str | None = None,
    ) -> None:
        self.problem_type = _normalize_problem_type(type)
        self.name = str(name or (self.problem_type or "nlpoptnet"))
        self.config = copy.deepcopy(_DEFAULT_CONFIG)
        if config is not None:
            self.config.update(copy.deepcopy(dict(config)))

        self.parameter_names: list[str] = []
        self.variable_names: list[str] = []
        self.parameter = _SymbolNamespace(self, "parameter")
        self.variable = _SymbolNamespace(self, "variable")
        self.parameters = self.parameter
        self.variables = self.variable
        self.constraints = _ConstraintManager(self)

        self._objective_expr: Expression | None = None
        self._objective_callable: Callable | None = None
        self._constants: dict[str, np.ndarray] = {}
        self._constant_counter = 0
        self._dataset_spec: dict[str, Any] | None = None
        self._region_spec: dict[str, Any] | None = None
        self._build_state: _BuildState | None = None
        self._inference_state: _InferenceState | None = None
        self._metadata_path: str | None = None
        self._extracted_problem_path: str | None = None

    def add_parameter(self, names: str | Iterable[str]):
        for name in _coerce_names(names):
            if name in self.parameter_names or name in self.variable_names:
                raise ValueError(f"Duplicate symbol name '{name}'.")
            self.parameter_names.append(name)
        return self.parameter

    def add_variable(self, names: str | Iterable[str]):
        for name in _coerce_names(names):
            if name in self.parameter_names or name in self.variable_names:
                raise ValueError(f"Duplicate symbol name '{name}'.")
            self.variable_names.append(name)
        return self.variable

    def objective(self, value) -> "NLPOptNet":
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
        return self._register_constant(values)

    def vector(self, values) -> Constant:
        return self._register_constant(values)

    def tensor(self, values) -> Constant:
        return self._register_constant(values)

    def extract(self, path: str | Path) -> dict[str, Constant]:
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

    def dataset(self, *, parameters: str | Path) -> "NLPOptNet":
        self._dataset_spec = {"parameters": str(parameters)}
        self._region_spec = None
        return self

    def simplex(self, *constraints, M=None, num_samples: int | None = None) -> "NLPOptNet":
        if M is not None and constraints:
            raise ValueError("Provide either simplex constraints or M, not both.")
        self._dataset_spec = None
        if M is not None:
            self._region_spec = {
                "type": "simplex_matrix",
                "matrix": self._resolve_polytope_matrix(M),
                "num_samples": int(self._resolve_num_samples(num_samples)),
            }
            return self
        if not constraints:
            raise ValueError("simplex(...) requires linear parameter constraints or M.")
        normalized_constraints: list[Constraint] = []
        for constraint in constraints:
            normalized = self._normalize_region_constraint(constraint)
            if isinstance(normalized, list):
                normalized_constraints.extend(normalized)
            else:
                normalized_constraints.append(normalized)
        self._region_spec = {
            "type": "simplex_constraints",
            "constraints": normalized_constraints,
            "num_samples": int(self._resolve_num_samples(num_samples)),
        }
        return self

    def box(self, *, lower=None, upper=None, num_samples: int | None = None) -> "NLPOptNet":
        self._dataset_spec = None
        n_x = len(self.parameter_names)
        if n_x <= 0:
            raise ValueError("Call add_parameter(...) before defining a parameter box.")

        if lower is None and "x_L" in self._constants:
            lower = self._constants["x_L"]
        if upper is None and "x_U" in self._constants:
            upper = self._constants["x_U"]
        if lower is None or upper is None:
            raise ValueError("box(...) needs lower and upper bounds, or extracted x_L/x_U constants.")

        low = np.asarray(lower if not isinstance(lower, Constant) else lower.value, dtype=np.float64).reshape(-1)
        high = np.asarray(upper if not isinstance(upper, Constant) else upper.value, dtype=np.float64).reshape(-1)
        if low.size == 1:
            low = np.full((n_x,), float(low[0]), dtype=np.float64)
        if high.size == 1:
            high = np.full((n_x,), float(high[0]), dtype=np.float64)
        if low.shape != (n_x,) or high.shape != (n_x,):
            raise ValueError(f"box bounds must each have shape ({n_x},).")
        self._region_spec = {
            "type": "box",
            "lower": low,
            "upper": high,
            "num_samples": int(self._resolve_num_samples(num_samples)),
        }
        return self

    def parameter_region(self, type: str):
        normalized = str(type).strip().lower()
        if normalized == "data":
            if self._dataset_spec is None:
                raise ValueError("parameter_region(type='data') requires model.dataset(...).")
        elif normalized == "simplex":
            if self._region_spec is None or not str(self._region_spec.get("type", "")).startswith("simplex"):
                raise ValueError("parameter_region(type='simplex') requires model.simplex(...).")
        elif normalized == "box":
            if self._region_spec is None or self._region_spec.get("type") != "box":
                self.box()
        else:
            raise ValueError("parameter_region(type=...) must be one of data, simplex, or box.")
        return self

    def lin(self, matrix, expr):
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
        const = self._ensure_constant(matrix)
        vector = self._as_vector_expr(expr)
        arr = np.asarray(const.value, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("quad(...) expects a 2D matrix.")
        return Expression(
            self,
            lambda ctx, a=arr, vec=vector: jnp.ravel(vec.eval(ctx)) @ (jnp.asarray(a) @ jnp.ravel(vec.eval(ctx))),
            f"quad({const.name}, {vector.text})",
        )

    def batch_quad(self, tensor, expr):
        const = self._ensure_constant(tensor)
        vector = self._as_vector_expr(expr)
        arr = np.asarray(const.value, dtype=np.float64)
        if arr.ndim != 3:
            raise ValueError("batch_quad(...) expects a 3D tensor.")
        return VectorExpression(
            self,
            lambda ctx, a=arr, vec=vector: jnp.einsum("mij,i,j->m", jnp.asarray(a), jnp.ravel(vec.eval(ctx)), jnp.ravel(vec.eval(ctx))),
            f"batch_quad({const.name}, {vector.text})",
            size=arr.shape[0],
        )

    def batch_exp(self, expr):
        vector = self._as_vector_expr(expr)
        return VectorExpression(
            self,
            lambda ctx, vec=vector: jnp.exp(jnp.asarray(vec.eval(ctx))),
            f"batch_exp({vector.text})",
            size=vector.size,
        )

    def sin(self, expr):
        return self._elementwise(expr, jnp.sin, "sin")

    def cos(self, expr):
        return self._elementwise(expr, jnp.cos, "cos")

    def exp(self, expr):
        return self._elementwise(expr, jnp.exp, "exp")

    def log(self, expr):
        return self._elementwise(expr, jnp.log, "log")

    def sqrt(self, expr):
        return self._elementwise(expr, jnp.sqrt, "sqrt")

    def abs(self, expr):
        return self._elementwise(expr, jnp.abs, "abs")

    def build(self) -> "NLPOptNet":
        if self._objective_expr is None and self._objective_callable is None:
            raise ValueError("Set the objective before build().")
        if not self.parameter_names:
            raise ValueError("Add at least one parameter.")
        if not self.variable_names:
            raise ValueError("Add at least one variable.")

        n_x = len(self.parameter_names)
        n_y = len(self.variable_names)
        dtype = resolve_dtype(str(self.config["dtype"]))
        lower_M, lower_c, upper_M, upper_c = self._build_box_bounds(n_x=n_x, n_y=n_y)

        def make_ctx(params, vars_dict):
            return _EvalContext(
                y=jnp.ravel(vars_dict["y"]),
                x=jnp.ravel(params["x"]),
                variable_index={name: idx for idx, name in enumerate(self.variable_names)},
                parameter_index={name: idx for idx, name in enumerate(self.parameter_names)},
            )

        if self._objective_callable is not None:
            objective_fun = self._objective_callable
        else:
            objective_expr = self._objective_expr
            assert objective_expr is not None

            def objective_fun(params, vars_dict):
                return objective_expr.eval(make_ctx(params, vars_dict))

        builder = (
            HighLevelNLPBuilder(dtype=dtype)
            .add_parameter("x", n_x)
            .add_variable("y", n_y)
            .set_objective(objective_fun)
            .set_affine_lower_bound(var_name="y", param_name="x", M=jnp.asarray(lower_M, dtype=dtype), c=jnp.asarray(lower_c, dtype=dtype))
            .set_affine_upper_bound(var_name="y", param_name="x", M=jnp.asarray(upper_M, dtype=dtype), c=jnp.asarray(upper_c, dtype=dtype))
        )

        eq_entries = list(self.constraints.equality.items)
        ineq_entries = list(self.constraints.inequality.items)

        eq_constraints = [entry for entry in eq_entries if isinstance(entry, Constraint)]
        eq_blocks = [entry for entry in eq_entries if isinstance(entry, _BlockConstraint)]
        ineq_constraints = [entry for entry in ineq_entries if isinstance(entry, Constraint)]
        ineq_blocks = [entry for entry in ineq_entries if isinstance(entry, _BlockConstraint)]

        if eq_constraints or eq_blocks:

            def eq_block(params, vars_dict):
                ctx = make_ctx(params, vars_dict)
                pieces = [jnp.ravel(entry.residual(ctx)) for entry in eq_constraints]
                pieces.extend(jnp.ravel(entry.fn(params, vars_dict)) for entry in eq_blocks)
                return jnp.concatenate(pieces, axis=0) if pieces else jnp.zeros((0,), dtype=dtype)

            builder = builder.add_nonlinear_equality(eq_block, name="nlpoptnet_eq_block")

        if ineq_constraints or ineq_blocks:

            def ineq_block(params, vars_dict):
                ctx = make_ctx(params, vars_dict)
                pieces = [jnp.ravel(entry.residual(ctx)) for entry in ineq_constraints]
                pieces.extend(jnp.ravel(entry.fn(params, vars_dict)) for entry in ineq_blocks)
                return jnp.concatenate(pieces, axis=0) if pieces else jnp.zeros((0,), dtype=dtype)

            builder = builder.add_nonlinear_inequality(ineq_block, name="nlpoptnet_ineq_block")

        model = builder.build(example_params={"x": jnp.zeros((n_x,), dtype=dtype)}, jit_compile=True)
        X = self._load_or_sample_parameters(n_x=n_x)
        train_idx, val_idx = split_train_val(X, train_frac=float(self.config["train_frac"]), seed=int(self.config["seed"]))
        effective_batch = int(min(int(self.config["batch_size"]), len(train_idx), len(val_idx)))
        train_cfg = self._train_cfg_dict(effective_batch)
        cfg = cfg_from_dict(train_cfg)

        X_train = jnp.asarray(X[train_idx], dtype=dtype)
        X_val = jnp.asarray(X[val_idx], dtype=dtype)
        train_batches = make_fixed_batches(X_train, cfg.batch_size)
        val_batches = make_fixed_batches(X_val, cfg.batch_size)
        if train_batches.shape[0] == 0 or val_batches.shape[0] == 0:
            raise ValueError("Not enough samples for the configured batch size after the train/validation split.")

        backbone, init_state_fn, train_step_fn, eval_step_fn = build_train_fns_from_jaxmodel(
            model_def=model,
            cfg=cfg,
            p=n_x,
        )
        train_epoch_fn, eval_epoch_fn = build_epoch_fns(train_step_fn, eval_step_fn)
        state = init_state_fn(jax.random.PRNGKey(int(cfg.seed)))
        if bool(self.config.get("jit_warmup", True)):
            state = warmup_compile(
                cfg=cfg,
                state=state,
                train_step_fn=train_step_fn,
                eval_step_fn=eval_step_fn,
                p=n_x,
                dtype=dtype,
                device=None,
            )

        self._build_state = _BuildState(
            model=model,
            cfg=cfg,
            train_config=train_cfg,
            dtype=dtype,
            X=X,
            train_idx=train_idx,
            val_idx=val_idx,
            train_batches=train_batches,
            val_batches=val_batches,
            backbone=backbone,
            init_state_fn=lambda: state,
            train_step_fn=train_step_fn,
            eval_step_fn=eval_step_fn,
            train_epoch_fn=train_epoch_fn,
            eval_epoch_fn=eval_epoch_fn,
            violation_fn=build_violation_fn_from_jaxmodel(model, cfg=cfg, p=n_x),
            objective_fn=make_batched_objective(model),
            sub_layer=make_subproblem_layer_from_model(model),
        )
        return self

    def optimize(self) -> dict[str, Any]:
        if self._build_state is None:
            self.build()
        assert self._build_state is not None

        build_state = self._build_state
        state = build_state.init_state_fn()
        history: list[dict[str, Any]] = []
        start = time.perf_counter()

        for epoch in range(1, int(build_state.cfg.epochs) + 1):
            state, train_tuple = build_state.train_epoch_fn(state, build_state.train_batches)
            val_tuple = build_state.eval_epoch_fn(state.params, build_state.val_batches)
            train_metrics = _metrics_tuple_to_dict(train_tuple)
            val_metrics = _metrics_tuple_to_dict(val_tuple)
            record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_objective": train_metrics["obj"],
                "val_objective": val_metrics["obj"],
                "train_mse_y": train_metrics["mse_y"],
                "val_mse_y": val_metrics["mse_y"],
                "train_mse_lam": train_metrics["mse_lam"],
                "val_mse_lam": val_metrics["mse_lam"],
                "train_mse_mu": train_metrics["mse_mu"],
                "val_mse_mu": val_metrics["mse_mu"],
            }
            history.append(record)
            if bool(self.config.get("verbose", True)) and (
                epoch == 1
                or epoch == int(build_state.cfg.epochs)
                or epoch % max(1, int(self.config.get("print_every", 10))) == 0
            ):
                print(
                    f"[{epoch:04d}/{int(build_state.cfg.epochs):04d}] "
                    f"train_loss={record['train_loss']:.6e} "
                    f"val_loss={record['val_loss']:.6e} "
                    f"train_obj={record['train_objective']:.6e} "
                    f"val_obj={record['val_objective']:.6e}"
                )

        training_wall_time = time.perf_counter() - start
        X_all = jnp.asarray(build_state.X, dtype=build_state.dtype)
        predictions = self._project_with_state(
            params=state.params,
            backbone=build_state.backbone,
            sub_layer=build_state.sub_layer,
            cfg=build_state.cfg,
            X=X_all,
        )
        objective_values = np.asarray(build_state.objective_fn(X_all, predictions), dtype=np.float64)
        violations = build_state.violation_fn(state.params, X_all)
        summary = {
            "framework": "nlpoptnet",
            "problem_type": self.problem_type,
            "num_samples": int(build_state.X.shape[0]),
            "train_samples": int(build_state.train_idx.shape[0]),
            "val_samples": int(build_state.val_idx.shape[0]),
            "epochs": int(build_state.cfg.epochs),
            "batch_size": int(build_state.cfg.batch_size),
            "objective_mean": float(np.mean(objective_values)),
            "objective_min": float(np.min(objective_values)),
            "objective_max": float(np.max(objective_values)),
            "eq_inf": float(np.asarray(violations["eq_inf"])),
            "eq_mean": float(np.asarray(violations["eq_mean"])),
            "ineq_inf": float(np.asarray(violations["ineq_inf"])),
            "ineq_mean": float(np.asarray(violations["ineq_mean"])),
            "bound_inf": float(np.asarray(violations["bound_inf"])),
            "bound_mean": float(np.asarray(violations["bound_mean"])),
            "training_wall_time_sec": float(training_wall_time),
            "final_train_loss": float(history[-1]["train_loss"]),
            "final_val_loss": float(history[-1]["val_loss"]),
            "final_train_objective": float(history[-1]["train_objective"]),
            "final_val_objective": float(history[-1]["val_objective"]),
        }

        run_dir = Path.cwd() / f"{self.name}_{timestamp()}"
        run_dir.mkdir(parents=True, exist_ok=False)
        parameters_path = run_dir / "parameters.csv"
        predictions_path = run_dir / "predicted_variables.csv"
        history_path = run_dir / "history.csv"
        weights_path = run_dir / "model_weights.npz"
        summary_path = run_dir / "summary.json"
        metadata_path = run_dir / "metadata.json"
        constants_path = run_dir / "problem_constants.npz"

        write_csv_matrix(parameters_path, build_state.X, headers=self.parameter_names)
        write_csv_matrix(predictions_path, np.asarray(predictions, dtype=np.float64), headers=self.variable_names)
        self._write_history_csv(history_path, history)
        np.savez(weights_path, params=np.array(jax.device_get(state.params), dtype=object))
        np.savez(constants_path, **self._constants)
        if self._extracted_problem_path is not None:
            shutil.copy2(resolve_path(self._extracted_problem_path), run_dir / "problem.npz")
        write_json(summary_path, summary)

        lower_M, lower_c, upper_M, upper_c = self._build_box_bounds(
            n_x=len(self.parameter_names),
            n_y=len(self.variable_names),
        )
        metadata = {
            "format": "nlpoptnet-metadata-v1",
            "version": "0.2.0",
            "model_name": self.name,
            "created_at": timestamp(),
            "problem_type": self.problem_type,
            "config": self._metadata_config(build_state.train_config),
            "problem": {
                "serializable": self._problem_is_serializable(),
                "parameter_names": list(self.parameter_names),
                "variable_names": list(self.variable_names),
                "objective_text": None if self._objective_expr is None else self._objective_expr.text,
                "equality_texts": [entry.text for entry in self.constraints.equality.items if isinstance(entry, Constraint)],
                "inequality_texts": [entry.text for entry in self.constraints.inequality.items if isinstance(entry, Constraint)],
                "bounds": {
                    "lower_M": lower_M.tolist(),
                    "lower_c": lower_c.tolist(),
                    "upper_M": upper_M.tolist(),
                    "upper_c": upper_c.tolist(),
                },
                "has_callable_objective": self._objective_callable is not None,
                "has_callable_blocks": any(isinstance(entry, _BlockConstraint) for entry in self.constraints.equality.items + self.constraints.inequality.items),
            },
            "artifacts": {
                "parameters": parameters_path.name,
                "predicted_variables": predictions_path.name,
                "history": history_path.name,
                "weights": weights_path.name,
                "summary": summary_path.name,
                "constants": constants_path.name,
                "problem_npz": "problem.npz" if self._extracted_problem_path is not None else None,
            },
        }
        write_json(metadata_path, metadata)

        self._metadata_path = str(metadata_path)
        self._inference_state = _InferenceState(
            model=build_state.model,
            cfg=build_state.cfg,
            train_config=build_state.train_config,
            params=state.params,
            backbone=build_state.backbone,
            sub_layer=build_state.sub_layer,
            n_x=len(self.parameter_names),
            output_dir=str(run_dir),
        )
        return {
            "output_dir": str(run_dir),
            "metadata_path": str(metadata_path),
            "summary": summary,
            "history": history,
        }

    def load(self, metadata_path: str | Path) -> "NLPOptNet":
        metadata_file = resolve_path(metadata_path)
        with open(metadata_file, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("format") != "nlpoptnet-metadata-v1":
            raise ValueError(f"Unsupported metadata format in {metadata_file}.")
        if not bool(payload["problem"]["serializable"]):
            raise RuntimeError(
                "This run was created from non-serializable Python callables and cannot be reloaded automatically."
            )

        self.name = str(payload["model_name"])
        self.problem_type = _normalize_problem_type(payload.get("problem_type"))
        self.config = copy.deepcopy(_DEFAULT_CONFIG)
        self.config.update(copy.deepcopy(dict(payload.get("config", {}))))
        self.parameter_names = list(payload["problem"]["parameter_names"])
        self.variable_names = list(payload["problem"]["variable_names"])
        self.parameter = _SymbolNamespace(self, "parameter")
        self.variable = _SymbolNamespace(self, "variable")
        self.parameters = self.parameter
        self.variables = self.variable
        self.constraints = _ConstraintManager(self)

        constants_file = metadata_file.parent / payload["artifacts"]["constants"]
        constants_npz = np.load(constants_file, allow_pickle=False)
        self._constants = {str(key): np.asarray(constants_npz[key]) for key in constants_npz.files}
        for key, value in self._constants.items():
            if not hasattr(self, key):
                setattr(self, key, Constant(self, key, value))

        dtype = resolve_dtype(str(self.config["dtype"]))
        model = build_model_from_problem_spec(payload["problem"], constants=self._constants, dtype=dtype)
        n_x = len(self.parameter_names)
        y0 = jnp.zeros((model.var_spec.total_size,), dtype=dtype)
        x0 = jnp.zeros((n_x,), dtype=dtype)
        me = int(model.eq_residual({"x": x0}, y0).shape[0])
        mi = int(model.ineq_residual({"x": x0}, y0).shape[0])
        backbone = Backbone(
            p=n_x,
            n=len(self.variable_names),
            me=me,
            mi=mi,
            hidden_size=int(self.config["hidden_size"]),
            hidden_dim=int(self.config["hidden_layers"]),
        )
        train_cfg = self._train_cfg_dict(batch_size=int(self.config["batch_size"]))
        cfg = cfg_from_dict(train_cfg)
        weights_file = metadata_file.parent / payload["artifacts"]["weights"]
        params = np.load(weights_file, allow_pickle=True)["params"].item()

        self._metadata_path = str(metadata_file)
        self._inference_state = _InferenceState(
            model=model,
            cfg=cfg,
            train_config=train_cfg,
            params=params,
            backbone=backbone,
            sub_layer=make_subproblem_layer_from_model(model),
            n_x=n_x,
            output_dir=str(metadata_file.parent),
        )
        return self

    def predict(self, values) -> np.ndarray:
        if self._inference_state is None:
            raise RuntimeError("Please train or load the model before calling predict().")
        data = np.asarray(values, dtype=np.float64)
        squeeze = data.ndim == 1
        if squeeze:
            data = data.reshape(1, -1)
        if data.ndim != 2 or data.shape[1] != self._inference_state.n_x:
            raise ValueError(
                f"predict expected a 1D or 2D array with {self._inference_state.n_x} parameter values."
            )
        x_batch = jnp.asarray(data, dtype=resolve_dtype(str(self._inference_state.train_config["dtype"])))
        projected = self._project_with_state(
            params=self._inference_state.params,
            backbone=self._inference_state.backbone,
            sub_layer=self._inference_state.sub_layer,
            cfg=self._inference_state.cfg,
            X=x_batch,
        )
        out = np.asarray(projected, dtype=np.float64)
        return out[0] if squeeze else out

    def _project_with_state(self, *, params, backbone, sub_layer, cfg, X):
        y_hat, lam_hat, mu_hat = backbone.apply({"params": params}, X)
        y_proj, _, _ = apply_projection_layers(
            sub_layer=sub_layer,
            x_batch=X,
            y0=y_hat,
            lam0=lam_hat,
            mu0=mu_hat,
            cfg=cfg,
        )
        _block_until_ready(y_proj)
        return y_proj

    def _metadata_config(self, train_config: dict[str, Any]) -> dict[str, Any]:
        out = dict(train_config)
        out["hidden_layers"] = int(self.config["hidden_layers"])
        out["verbose"] = bool(self.config.get("verbose", True))
        out["num_samples"] = int(self.config.get("num_samples", 1000))
        return out

    def _problem_is_serializable(self) -> bool:
        if self._objective_expr is None:
            return False
        if self._objective_callable is not None:
            return False
        if any(isinstance(entry, _BlockConstraint) for entry in self.constraints.equality.items):
            return False
        if any(isinstance(entry, _BlockConstraint) for entry in self.constraints.inequality.items):
            return False
        return True

    def _write_history_csv(self, path: Path, history: list[dict[str, Any]]) -> None:
        if not history:
            return
        write_csv_matrix(
            path,
            np.asarray([[row[key] for key in history[0].keys()] for row in history], dtype=np.float64),
            headers=list(history[0].keys()),
        )

    def _train_cfg_dict(self, batch_size: int) -> dict[str, Any]:
        return {
            "batch_size": int(batch_size),
            "epochs": int(self.config["epochs"]),
            "learning_rate": float(self.config["learning_rate"]),
            "alpha_consistency": float(self.config.get("alpha_consistency", 1.0)),
            "train_frac": float(self.config["train_frac"]),
            "hidden_size": int(self.config["hidden_size"]),
            "hidden_dim": int(self.config["hidden_layers"]),
            "cp_iters": int(self.config.get("cp_iters", 5000)),
            "cp_tol": float(self.config.get("cp_tol", 1e-6)),
            "IS_FIXED": bool(self.config.get("IS_FIXED", True)),
            "stepsize": str(self.config.get("stepsize", "auto")),
            "safety": float(self.config.get("safety", 0.90)),
            "knorm_iters": int(self.config.get("knorm_iters", 20)),
            "knorm_seed": int(self.config.get("knorm_seed", 42)),
            "seed": int(self.config["seed"]),
            "adjoint_iters": int(self.config.get("adjoint_iters", 50)),
            "use_ruiz": bool(self.config.get("use_ruiz", True)),
            "ruiz_iters": int(self.config.get("ruiz_iters", 4)),
            "k_layer": int(self.config.get("k_layer", 1)),
            "dtype": str(self.config["dtype"]),
            "device": str(self.config.get("device", "auto")),
            "jit_warmup": bool(self.config.get("jit_warmup", True)),
        }

    def _load_or_sample_parameters(self, *, n_x: int) -> np.ndarray:
        if self._dataset_spec is not None:
            return load_csv_matrix(self._dataset_spec["parameters"], self.parameter_names)
        if self._region_spec is None:
            if "x_L" in self._constants and "x_U" in self._constants:
                self.box()
            else:
                raise ValueError("Provide parameters via model.dataset(...), model.simplex(...), or model.box(...).")
        assert self._region_spec is not None

        region_type = str(self._region_spec["type"])
        if region_type == "box":
            return sample_box(
                self._region_spec["lower"],
                self._region_spec["upper"],
                num_samples=int(self._region_spec["num_samples"]),
                seed=int(self.config["seed"]),
            )
        if region_type == "simplex_matrix":
            A, b = self._region_spec["matrix"]
            return hit_and_run_samples(A, b, num_samples=int(self._region_spec["num_samples"]), seed=int(self.config["seed"]))
        if region_type == "simplex_constraints":
            A_rows = []
            b_rows = []
            zeros = jnp.zeros((len(self.variable_names),), dtype=jnp.float64)
            param_index = {name: idx for idx, name in enumerate(self.parameter_names)}
            ctx0 = _EvalContext(
                y=zeros,
                x=jnp.zeros((n_x,), dtype=jnp.float64),
                variable_index={name: idx for idx, name in enumerate(self.variable_names)},
                parameter_index=param_index,
            )
            for constraint in self._region_spec["constraints"]:
                residual0 = float(np.asarray(constraint.residual(ctx0)))

                def residual_fn(x_vec):
                    ctx = _EvalContext(
                        y=zeros,
                        x=jnp.asarray(x_vec, dtype=jnp.float64),
                        variable_index={name: idx for idx, name in enumerate(self.variable_names)},
                        parameter_index=param_index,
                    )
                    return jnp.asarray(constraint.residual(ctx), dtype=jnp.float64)

                grad = np.asarray(jax.jacobian(residual_fn)(jnp.zeros((n_x,), dtype=jnp.float64)), dtype=np.float64).reshape(-1)
                A_rows.append(grad)
                b_rows.append(-residual0)
            A = np.asarray(A_rows, dtype=np.float64)
            b = np.asarray(b_rows, dtype=np.float64)
            return hit_and_run_samples(A, b, num_samples=int(self._region_spec["num_samples"]), seed=int(self.config["seed"]))
        raise ValueError(f"Unsupported parameter region type '{region_type}'.")

    def _normalize_region_constraint(self, constraint):
        if isinstance(constraint, Constraint):
            if constraint.kind == "eq":
                return [
                    Constraint(constraint.lhs, constraint.rhs, "ineq", "<=", f"{constraint.lhs.text} <= {constraint.rhs.text}"),
                    Constraint(constraint.lhs, constraint.rhs, "ineq", ">=", f"{constraint.lhs.text} >= {constraint.rhs.text}"),
                ]
            if constraint.kind != "ineq":
                raise ValueError("simplex(...) only accepts equality or inequality constraints.")
            return constraint
        if callable(constraint):
            signature = inspect.signature(constraint)
            if len(signature.parameters) == 0:
                return self._normalize_region_constraint(constraint())
        raise TypeError("simplex(...) expects parameter constraints built from the symbolic API.")

    def _resolve_polytope_matrix(self, matrix_source):
        if isinstance(matrix_source, (str, Path)):
            path = resolve_path(matrix_source)
            arrays = np.load(path, allow_pickle=False)
            if "A" in arrays and "b" in arrays:
                return np.asarray(arrays["A"], dtype=np.float64), np.asarray(arrays["b"], dtype=np.float64).reshape(-1)
            if "M" in arrays:
                M = np.asarray(arrays["M"], dtype=np.float64)
                if "b" in arrays:
                    return M, np.asarray(arrays["b"], dtype=np.float64).reshape(-1)
                return M, np.zeros((M.shape[0],), dtype=np.float64)
            raise ValueError("simplex(M=...) npz files must contain either (A, b) or M.")
        const = self._ensure_constant(matrix_source)
        matrix = np.asarray(const.value, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("simplex(M=...) expects a 2D matrix or an npz file containing A/b.")
        return matrix, np.zeros((matrix.shape[0],), dtype=np.float64)

    def _resolve_num_samples(self, explicit: int | None) -> int:
        if explicit is not None:
            return int(explicit)
        return int(self.config.get("num_samples", 1000))

    def _build_box_bounds(self, *, n_x: int, n_y: int):
        default_radius = float(self.config.get("y_bound", 10.0))
        lower_M = np.zeros((n_y, n_x), dtype=np.float64)
        lower_c = -default_radius * np.ones((n_y,), dtype=np.float64)
        upper_M = np.zeros((n_y, n_x), dtype=np.float64)
        upper_c = default_radius * np.ones((n_y,), dtype=np.float64)

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
                raise ValueError(f"Vector shape mismatch: expected size {size}, got {value.size}.")
            return value
        if isinstance(value, _SymbolNamespace):
            vector = value.vector()
            if size is not None and vector.size != int(size):
                raise ValueError(f"Vector shape mismatch: expected size {size}, got {vector.size}.")
            return vector
        if isinstance(value, Expression):
            if size is None:
                raise TypeError("Cannot broadcast a scalar expression to a vector without a target size.")
            return VectorExpression(
                self,
                lambda ctx, expr=value, count=int(size): jnp.broadcast_to(expr.eval(ctx), (count,)),
                value.text,
                size=int(size),
                affine_param=None if value.affine_param is None else _AffineParamVector(
                    matrix=np.broadcast_to(value.affine_param.coeff[None, :], (int(size), len(self.parameter_names))).copy(),
                    const=np.full((int(size),), float(value.affine_param.const), dtype=np.float64),
                ),
            )
        if isinstance(value, Constant):
            vector = self._constant_to_vector_expr(value)
            if size is not None and vector.size != int(size):
                if vector.size == 1:
                    return self._as_vector_expr(self._as_scalar_expr(float(vector.affine_param.const[0])), size=size)
                raise ValueError(f"Vector shape mismatch: expected size {size}, got {vector.size}.")
            return vector
        arr = np.asarray(value)
        if arr.ndim == 0:
            if size is None:
                raise TypeError("Cannot broadcast a scalar constant to a vector without a target size.")
            return self._as_vector_expr(self._as_scalar_expr(float(arr)), size=size)
        if arr.ndim == 1:
            const = self._register_constant(arr)
            vector = self._constant_to_vector_expr(const)
            if size is not None and vector.size != int(size):
                raise ValueError(f"Vector shape mismatch: expected size {size}, got {vector.size}.")
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
            return _AffineParamScalar(coeff=lhs.coeff + rhs.coeff, const=lhs.const + rhs.const)
        if symbol == "-":
            return _AffineParamScalar(coeff=lhs.coeff - rhs.coeff, const=lhs.const - rhs.const)
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
            return _AffineParamVector(matrix=lhs.matrix + rhs.matrix, const=lhs.const + rhs.const)
        if symbol == "-":
            return _AffineParamVector(matrix=lhs.matrix - rhs.matrix, const=lhs.const - rhs.const)
        return None

    def _check_symbol(self, kind: str, name: str) -> None:
        pool = self.parameter_names if kind == "parameter" else self.variable_names
        if name not in pool:
            raise AttributeError(f"Unknown {kind} '{name}'.")


ProblemBuilder = NLPOptNet
