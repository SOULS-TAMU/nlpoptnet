"""Data resources for the NLPOptNet frontend."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nlpoptnet.api.model import NLPOptNet

from nlpoptnet.api.formulation.expressions import Constant
from nlpoptnet.api.formulation.expressions import Constraint
from pathlib import Path
from nlpoptnet.api.formulation.expressions import _EvalContext
from ohagi.data.sampling import hit_and_run_samples
import inspect
import jax
import jax.numpy as jnp
from ohagi.data.sampling import load_csv_matrix
import numpy as np
from ohagi._utils.serialization import resolve_path
from ohagi.data.sampling import sample_box


def dataset(self, *, parameters: str | Path) -> "NLPOptNet":
    """Use a CSV file of parameter samples as the dataset source."""
    self._dataset_spec = {"parameters": str(parameters)}
    self._region_spec = None
    return self


def simplex(self, *constraints, M=None, num_samples: int | None = None) -> "NLPOptNet":
    """Sample parameters from a simplex or more general polytope region."""
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
    """Sample parameters from an axis-aligned box."""
    self._dataset_spec = None
    n_x = len(self.parameter_names)
    if n_x <= 0:
        raise ValueError("Call add_parameter(...) before defining a parameter box.")

    if lower is None and "x_L" in self._constants:
        lower = self._constants["x_L"]
    if upper is None and "x_U" in self._constants:
        upper = self._constants["x_U"]
    if lower is None or upper is None:
        raise ValueError(
            "box(...) needs lower and upper bounds, or extracted x_L/x_U constants."
        )

    low = np.asarray(
        lower if not isinstance(lower, Constant) else lower.value, dtype=np.float64
    ).reshape(-1)
    high = np.asarray(
        upper if not isinstance(upper, Constant) else upper.value, dtype=np.float64
    ).reshape(-1)
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
    """Validate or activate a parameter-region mode."""
    normalized = str(type).strip().lower()
    if normalized == "data":
        if self._dataset_spec is None:
            raise ValueError(
                "parameter_region(type='data') requires model.dataset(...)."
            )
    elif normalized == "simplex":
        if self._region_spec is None or not str(
            self._region_spec.get("type", "")
        ).startswith("simplex"):
            raise ValueError(
                "parameter_region(type='simplex') requires model.simplex(...)."
            )
    elif normalized == "box":
        if self._region_spec is None or self._region_spec.get("type") != "box":
            self.box()
    else:
        raise ValueError(
            "parameter_region(type=...) must be one of data, simplex, or box."
        )
    return self


def _load_or_sample_parameters(self, *, n_x: int) -> np.ndarray:
    """Load parameter observations or sample them from the configured feasible region."""
    if self._dataset_spec is not None:
        return load_csv_matrix(self._dataset_spec["parameters"], self.parameter_names)
    if self._region_spec is None:
        if "x_L" in self._constants and "x_U" in self._constants:
            self.box()
        else:
            raise ValueError(
                "Provide parameters via model.dataset(...), model.simplex(...), or model.box(...)."
            )
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
        return hit_and_run_samples(
            A,
            b,
            num_samples=int(self._region_spec["num_samples"]),
            seed=int(self.config["seed"]),
        )
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
                    variable_index={
                        name: idx for idx, name in enumerate(self.variable_names)
                    },
                    parameter_index=param_index,
                )
                return jnp.asarray(constraint.residual(ctx), dtype=jnp.float64)

            grad = np.asarray(
                jax.jacobian(residual_fn)(jnp.zeros((n_x,), dtype=jnp.float64)),
                dtype=np.float64,
            ).reshape(-1)
            A_rows.append(grad)
            b_rows.append(-residual0)
        A = np.asarray(A_rows, dtype=np.float64)
        b = np.asarray(b_rows, dtype=np.float64)
        return hit_and_run_samples(
            A,
            b,
            num_samples=int(self._region_spec["num_samples"]),
            seed=int(self.config["seed"]),
        )
    raise ValueError(f"Unsupported parameter region type '{region_type}'.")


def _normalize_region_constraint(self, constraint):
    if isinstance(constraint, Constraint):
        if constraint.kind == "eq":
            return [
                Constraint(
                    constraint.lhs,
                    constraint.rhs,
                    "ineq",
                    "<=",
                    f"{constraint.lhs.text} <= {constraint.rhs.text}",
                ),
                Constraint(
                    constraint.lhs,
                    constraint.rhs,
                    "ineq",
                    ">=",
                    f"{constraint.lhs.text} >= {constraint.rhs.text}",
                ),
            ]
        if constraint.kind != "ineq":
            raise ValueError(
                "simplex(...) only accepts equality or inequality constraints."
            )
        return constraint
    if callable(constraint):
        signature = inspect.signature(constraint)
        if len(signature.parameters) == 0:
            return self._normalize_region_constraint(constraint())
    raise TypeError(
        "simplex(...) expects parameter constraints built from the symbolic API."
    )


def _resolve_polytope_matrix(self, matrix_source):
    if isinstance(matrix_source, (str, Path)):
        path = resolve_path(matrix_source)
        arrays = np.load(path, allow_pickle=False)
        if "A" in arrays and "b" in arrays:
            return np.asarray(arrays["A"], dtype=np.float64), np.asarray(
                arrays["b"], dtype=np.float64
            ).reshape(-1)
        if "M" in arrays:
            M = np.asarray(arrays["M"], dtype=np.float64)
            if "b" in arrays:
                return M, np.asarray(arrays["b"], dtype=np.float64).reshape(-1)
            return M, np.zeros((M.shape[0],), dtype=np.float64)
        raise ValueError("simplex(M=...) npz files must contain either (A, b) or M.")
    const = self._ensure_constant(matrix_source)
    matrix = np.asarray(const.value, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(
            "simplex(M=...) expects a 2D matrix or an npz file containing A/b."
        )
    return matrix, np.zeros((matrix.shape[0],), dtype=np.float64)


def _resolve_num_samples(self, explicit: int | None) -> int:
    if explicit is not None:
        return int(explicit)
    return int(self.config.get("num_samples", 1000))
