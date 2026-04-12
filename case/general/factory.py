from __future__ import annotations

import copy
import hashlib
import importlib.util
import sys
from dataclasses import asdict, is_dataclass
import inspect
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from jaxmodel import JaxNLPModel
from solgen import SolGenModel, SolveResult, explain_cvxpy_support
from scripts.misc.inequality_multipliers import default_ineq_multipliers

jax.config.update("jax_enable_x64", True)

DATASET_KIND = "general"
PARAM_NAME = "x"


def default_case_dir() -> Path:
    return Path(__file__).resolve().parent


def model_definition_path(case_dir: Path | None = None) -> Path:
    root = Path(case_dir) if case_dir is not None else default_case_dir()
    return root / "model_definition.py"


def normalize_problem_type(problem_type: str) -> str:
    normalized = str(problem_type).strip().lower()
    if normalized != DATASET_KIND:
        raise ValueError(f"case/general only supports type='{DATASET_KIND}', got '{problem_type}'.")
    return normalized


def _file_sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:12]


def _module_name_for_path(path: Path) -> str:
    return f"_nlpopt_general_model_{_file_sha1(path)}"


def load_model_module(case_dir: Path | None = None):
    path = model_definition_path(case_dir)
    if not path.exists():
        raise FileNotFoundError(f"Expected general model definition at {path}")

    module_name = _module_name_for_path(path)
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load model definition module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _coerce_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    if is_dataclass(value):
        return copy.deepcopy(asdict(value))
    raise TypeError("Expected a mapping or dataclass-compatible problem payload.")


def _param_name(module) -> str:
    return str(getattr(module, "PARAM_NAME", PARAM_NAME))


def _infer_shape(model: JaxNLPModel, *, n_x: int, param_name: str) -> dict[str, int]:
    y0 = jnp.zeros((model.var_spec.total_size,), dtype=model.dtype)
    x0 = jnp.zeros((n_x,), dtype=model.dtype)
    params = {param_name: x0}
    return {
        "n_x": int(n_x),
        "n_y": int(model.var_spec.total_size),
        "n_eq": int(model.eq_residual(params, y0).shape[0]),
        "n_ineq": int(model.ineq_residual(params, y0).shape[0]),
    }


def build_problem_model(*, dtype=jnp.float64, case_dir: Path | None = None) -> JaxNLPModel:
    module = load_model_module(case_dir)
    if not hasattr(module, "build_model"):
        raise AttributeError(f"{model_definition_path(case_dir)} must define build_model(..., *, dtype).")
    if _param_name(module) != PARAM_NAME:
        raise ValueError(
            f"The general case currently expects the parameter name '{PARAM_NAME}'. "
            f"Found '{_param_name(module)}' in model_definition.py."
        )
    signature = inspect.signature(module.build_model)
    if "model_cfg" in signature.parameters:
        model = module.build_model(model_cfg={}, dtype=dtype)
    else:
        model = module.build_model(dtype=dtype)
    if not isinstance(model, JaxNLPModel):
        raise TypeError("build_model(...) must return a JaxNLPModel instance.")
    return model


def build_problem_model_from_data(
    problem_data: Mapping[str, Any],
    *,
    dtype=jnp.float64,
    case_dir: Path | None = None,
) -> JaxNLPModel:
    _coerce_payload(problem_data)
    return build_problem_model(dtype=dtype, case_dir=case_dir)


def _normalize_solver_name(name: object) -> str | None:
    text = str(name).strip()
    if text == "" or text.lower() == "auto":
        return None
    return text.upper()


def _normalize_solve_output(result: Any, *, n_ineq: int) -> dict[str, Any]:
    if isinstance(result, SolveResult):
        return {
            "status": str(result.status),
            "y": np.asarray(result.y, dtype=np.float64),
            "objective": float(result.objective),
            "mu": np.asarray(result.mu, dtype=np.float64) if result.mu is not None else default_ineq_multipliers(n_ineq),
        }
    if isinstance(result, Mapping):
        status = str(result.get("status", "unknown"))
        y = result.get("y")
        if y is None:
            raise ValueError("Custom solve_instance(...) must return a y value.")
        objective = result.get("objective")
        mu = result.get("mu")
        return {
            "status": status,
            "y": np.asarray(y, dtype=np.float64),
            "objective": None if objective is None else float(objective),
            "mu": default_ineq_multipliers(n_ineq) if mu is None else np.asarray(mu, dtype=np.float64),
        }
    raise TypeError("solve_instance(...) must return either solgen.SolveResult or a dict-like payload.")


class GeneralProblemGenerator:
    def __init__(
        self,
        data_cfg: Mapping[str, Any],
        *,
        case_dir: Path | None = None,
    ) -> None:
        self.case_dir = Path(case_dir) if case_dir is not None else default_case_dir()
        self.data_cfg = copy.deepcopy(dict(data_cfg))
        self.module = load_model_module(self.case_dir)
        self.param_name = _param_name(self.module)

        x_l = np.asarray(self.data_cfg["x_L"], dtype=np.float64)
        x_u = np.asarray(self.data_cfg["x_U"], dtype=np.float64)
        if x_l.shape != x_u.shape:
            raise ValueError("x_L and x_U must have the same shape.")
        self.n_x = int(x_l.shape[0])
        self.x_L = x_l
        self.x_U = x_u

        self.model = build_problem_model(dtype=jnp.float64, case_dir=self.case_dir)
        dims = _infer_shape(self.model, n_x=self.n_x, param_name=self.param_name)
        self.n_y = dims["n_y"]
        self.n_eq = dims["n_eq"]
        self.n_ineq = dims["n_ineq"]
        self.solver = str(self.data_cfg.get("solver", "auto"))
        self.requested_solver = self.solver
        self._sample_hook = getattr(self.module, "sample_parameters", None)
        self._solve_hook = getattr(self.module, "solve_instance", None)
        self._solgen: SolGenModel | None = None

    def sample_parameters(self, num_samples: int) -> np.ndarray:
        num_samples = int(num_samples)
        rng = np.random.default_rng(int(self.data_cfg["seed"]))
        if self._sample_hook is not None:
            sample_signature = inspect.signature(self._sample_hook)
            sample_kwargs = {
                "num_samples": num_samples,
                "rng": rng,
                "data_cfg": copy.deepcopy(self.data_cfg),
            }
            if "model_cfg" in sample_signature.parameters:
                sample_kwargs["model_cfg"] = {}
            xs = self._sample_hook(**sample_kwargs)
            xs = np.asarray(xs, dtype=np.float64)
        else:
            xs = rng.uniform(self.x_L, self.x_U, size=(num_samples, self.n_x))
        if xs.shape != (num_samples, self.n_x):
            raise ValueError(f"sample_parameters(...) must return shape ({num_samples}, {self.n_x}).")
        return xs

    def solve_for_x(self, x: np.ndarray) -> dict[str, Any]:
        params = {self.param_name: jnp.asarray(x, dtype=self.model.dtype)}
        if self._solve_hook is not None:
            solve_signature = inspect.signature(self._solve_hook)
            solve_kwargs = {
                "model": self.model,
                "params": params,
                "solver": self.solver,
            }
            if "model_cfg" in solve_signature.parameters:
                solve_kwargs["model_cfg"] = {}
            result = self._solve_hook(**solve_kwargs)
            return _normalize_solve_output(result, n_ineq=self.n_ineq)

        if self._solgen is None:
            self._solgen = SolGenModel(self.model)
        if not self._solgen.supports_direct_cvxpy():
            raise NotImplementedError(
                "This general model is not directly solvable by solgen/CVXPY. "
                + explain_cvxpy_support(self.model)
                + " Add solve_instance(model, params, solver) to model_definition.py "
                + "only if you need reference solutions."
            )

        result = self._solgen.solve(params, solver=_normalize_solver_name(self.solver))
        return _normalize_solve_output(result, n_ineq=self.n_ineq)

    def get_problem_data(self) -> dict[str, Any]:
        return {
            "type": DATASET_KIND,
            "module_path": str(model_definition_path(self.case_dir)),
            "module_hash": _file_sha1(model_definition_path(self.case_dir)),
            "n_x": int(self.n_x),
            "n_y": int(self.n_y),
            "n_eq": int(self.n_eq),
            "n_ineq": int(self.n_ineq),
        }


def build_problem_generator(
    data_cfg: Mapping[str, Any],
    *,
    case_dir: Path | None = None,
) -> GeneralProblemGenerator:
    payload = dict(data_cfg)
    normalize_problem_type(str(payload["type"]))
    return GeneralProblemGenerator(payload, case_dir=case_dir)
