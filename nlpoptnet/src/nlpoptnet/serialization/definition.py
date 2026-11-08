"""Definition resources for the NLPOptNet frontend."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nlpoptnet.api.model import NLPOptNet

from typing import Any
from nlpoptnet.api.network import Backbone
from typing import Callable
from nlpoptnet.api.formulation.expressions import Constant
from nlpoptnet.api.formulation.builder import HighLevelNLPBuilder
from pathlib import Path
from nlpoptnet.api.formulation.expressions import _BlockConstraint
from nlpoptnet.api.formulation.expressions import _ConstraintManager
from nlpoptnet.api.configuration import _DEFAULT_CONFIG
from nlpoptnet.api.configuration import _InferenceState
from nlpoptnet.api.configuration import _ScalingState
from nlpoptnet.api.formulation.expressions import _SymbolNamespace
from nlpoptnet.api.configuration import _apply_config_aliases
from nlpoptnet.api.configuration import _normalize_problem_type
from nlpoptnet.api.configuration import cfg_from_dict
import copy
from datetime import datetime as datetime
from nlpoptnet.api.formulation.constraints import eq_block as eq_block
from nlpoptnet.api.formulation.constraints import ineq_block as ineq_block
import jax.numpy as jnp
from nlpoptnet.api.formulation.approximation import make_subproblem_layer_from_model
import numpy as np
from purin.execution.runtime import resolve_jax_dtype as resolve_dtype
from ohagi._utils.serialization import resolve_path


def load(
    self, metadata_path: str | Path, *, verbose: bool | None = None
) -> "NLPOptNet":
    """Load a previously saved serializable model run from metadata."""
    metadata_file = resolve_path(metadata_path)
    from ohagi.serialization import read_model, load_parameters

    payload = read_model(metadata_file, framework="nlpoptnet")
    if not bool(payload["problem"]["serializable"]):
        raise RuntimeError(
            "This run was created from non-serializable Python callables and cannot be reloaded automatically."
        )

    self.name = str(payload["model_name"])
    self.problem_type = _normalize_problem_type(payload.get("problem_type"))
    self.config = copy.deepcopy(_DEFAULT_CONFIG)
    self.config.update(copy.deepcopy(dict(payload.get("config", {}))))
    _apply_config_aliases(self.config)
    self.parameter_names = list(payload["problem"]["parameter_names"])
    self.variable_names = list(payload["problem"]["variable_names"])
    self.parameter = _SymbolNamespace(self, "parameter")
    self.variable = _SymbolNamespace(self, "variable")
    self.parameters = self.parameter
    self.variables = self.variable
    self.constraints = _ConstraintManager(self)
    scaling_spec = payload["problem"].get("scaling", {})
    self.scaling = _ScalingState()
    self.scaling.enabled = bool(scaling_spec.get("enabled", False))

    D_p = scaling_spec.get("D_p")
    D_v = scaling_spec.get("D_v")
    D_eq = scaling_spec.get("D_eq")
    D_ineq = scaling_spec.get("D_ineq")

    self.scaling.D_p = None if D_p is None else np.asarray(D_p, dtype=np.float64)
    self.scaling.D_v = None if D_v is None else np.asarray(D_v, dtype=np.float64)
    self.scaling.D_obj = float(scaling_spec.get("D_obj", 1.0))
    self.scaling.D_eq = None if D_eq is None else np.asarray(D_eq, dtype=np.float64)
    self.scaling.D_ineq = (
        None if D_ineq is None else np.asarray(D_ineq, dtype=np.float64)
    )

    self.scaling.mode = str(scaling_spec.get("mode", "fast"))
    self.scaling.use_jacobian_scaling = bool(
        scaling_spec.get("use_jacobian_scaling", False)
    )
    self.scaling.probe_points = int(scaling_spec.get("probe_points", 8))
    self.scaling.objective_percentile = float(
        scaling_spec.get("objective_percentile", 95.0)
    )
    self.scaling.constraint_percentile = float(
        scaling_spec.get("constraint_percentile", 95.0)
    )
    self.scaling.eps = float(scaling_spec.get("eps", 1e-12))

    if self.scaling.enabled:
        if self.scaling.D_p is None:
            self.scaling.D_p = np.ones((len(self.parameter_names),), dtype=np.float64)
        if self.scaling.D_v is None:
            self.scaling.D_v = np.ones((len(self.variable_names),), dtype=np.float64)

    constants_file = metadata_file.parent / payload["artifacts"]["constants"]
    constants_npz = np.load(constants_file, allow_pickle=False)
    self._constants = {
        str(key): np.asarray(constants_npz[key]) for key in constants_npz.files
    }
    for key, value in self._constants.items():
        if not hasattr(self, key):
            setattr(self, key, Constant(self, key, value))

    dtype = resolve_dtype(str(self.config["dtype"]))
    model = build_model_from_problem_spec(
        payload["problem"],
        constants=self._constants,
        dtype=dtype,
    )
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
    params = load_parameters(metadata_file, payload)

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
    if verbose is None:
        verbose = bool(self.config.get("verbose", True))

    if verbose:
        from datetime import datetime

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print("\n" + "=" * 120)
        print("✅ Model Loaded Successfully")
        print("-" * 120)
        print(f"Model Name   : {self.name}")
        print(f"Location     : {metadata_file.parent}")
        print(f"Time         : {now}")
        print("=" * 120 + "\n")

    return self


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
    if any(
        isinstance(entry, _BlockConstraint) for entry in self.constraints.equality.items
    ):
        return False
    if any(
        isinstance(entry, _BlockConstraint)
        for entry in self.constraints.inequality.items
    ):
        return False
    return True


def parse_constraint_text(text: str) -> tuple[str, str]:
    """Convert serialized constraint text into residual form and constraint kind."""
    payload = str(text).strip()
    if "==" in payload:
        left, right = payload.split("==", 1)
        return f"({left.strip()}) - ({right.strip()})", "eq"
    if "<=" in payload:
        left, right = payload.split("<=", 1)
        return f"({left.strip()}) - ({right.strip()})", "ineq"
    if ">=" in payload:
        left, right = payload.split(">=", 1)
        return f"({right.strip()}) - ({left.strip()})", "ineq"
    raise ValueError(f"Unsupported constraint format: {text}")


def _safe_env(constants: dict[str, Any]):
    """Build the restricted expression environment used during reload."""

    def lin(A, z):
        return jnp.asarray(A) @ jnp.asarray(z)

    def batch_lin(A, z):
        return jnp.asarray(A) @ jnp.asarray(z)

    def quad(Q, z):
        z_vec = jnp.ravel(jnp.asarray(z))
        return z_vec @ (jnp.asarray(Q) @ z_vec)

    def batch_quad(Qs, z):
        z_vec = jnp.ravel(jnp.asarray(z))
        return jnp.einsum("mij,i,j->m", jnp.asarray(Qs), z_vec, z_vec)

    def batch_exp(z):
        return jnp.exp(jnp.asarray(z))

    env = {
        "jnp": jnp,
        "np": jnp,
        "sin": jnp.sin,
        "cos": jnp.cos,
        "tan": jnp.tan,
        "exp": jnp.exp,
        "log": jnp.log,
        "sqrt": jnp.sqrt,
        "abs": jnp.abs,
        "maximum": jnp.maximum,
        "minimum": jnp.minimum,
        "pi": jnp.pi,
        "lin": lin,
        "batch_lin": batch_lin,
        "quad": quad,
        "batch_quad": batch_quad,
        "batch_exp": batch_exp,
    }
    for name, value in constants.items():
        array = np.asarray(value)
        if array.dtype.kind in {"b", "i", "u", "f", "c"}:
            env[str(name)] = jnp.asarray(array)
        else:
            env[str(name)] = value
    return env


def make_scalar_eval_fn(
    expr: str,
    *,
    parameter_names: list[str],
    variable_names: list[str],
    constants: dict[str, Any],
) -> Callable:
    """Build a scalar evaluator from serialized expression text."""
    base_env = _safe_env(constants)

    def fn(y, x):
        env = dict(base_env)
        x_vec = jnp.ravel(jnp.asarray(x))
        y_vec = jnp.ravel(jnp.asarray(y))
        env["x"] = x_vec
        env["y"] = y_vec
        for idx, name in enumerate(parameter_names):
            env[str(name)] = x_vec[idx]
        for idx, name in enumerate(variable_names):
            env[str(name)] = y_vec[idx]
        return eval(expr, {"__builtins__": {}}, env)

    return fn


def build_model_from_problem_spec(
    problem_spec: dict[str, Any],
    *,
    constants: dict[str, Any],
    dtype=jnp.float64,
):
    """Rebuild a serializable nonlinear problem from saved metadata."""
    parameter_names = list(problem_spec["parameter_names"])
    variable_names = list(problem_spec["variable_names"])
    objective_text = str(problem_spec["objective_text"])
    equality_texts = list(problem_spec.get("equality_texts", []))
    inequality_texts = list(problem_spec.get("inequality_texts", []))
    lower_M = jnp.asarray(problem_spec["bounds"]["lower_M"], dtype=dtype)
    lower_c = jnp.asarray(problem_spec["bounds"]["lower_c"], dtype=dtype)
    upper_M = jnp.asarray(problem_spec["bounds"]["upper_M"], dtype=dtype)
    upper_c = jnp.asarray(problem_spec["bounds"]["upper_c"], dtype=dtype)

    scaling = problem_spec.get("scaling", {})
    scaling_enabled = bool(scaling.get("enabled", False))

    D_p = jnp.asarray(
        scaling.get("D_p", [1.0] * len(parameter_names)),
        dtype=dtype,
    )
    D_v = jnp.asarray(
        scaling.get("D_v", [1.0] * len(variable_names)),
        dtype=dtype,
    )
    D_obj = jnp.asarray(float(scaling.get("D_obj", 1.0)), dtype=dtype)

    D_eq_raw = scaling.get("D_eq")
    D_ineq_raw = scaling.get("D_ineq")
    D_eq = None if D_eq_raw is None else jnp.asarray(D_eq_raw, dtype=dtype).reshape(-1)
    D_ineq = (
        None if D_ineq_raw is None else jnp.asarray(D_ineq_raw, dtype=dtype).reshape(-1)
    )

    objective_fn = make_scalar_eval_fn(
        objective_text,
        parameter_names=parameter_names,
        variable_names=variable_names,
        constants=constants,
    )

    def objective(params, vars_dict):
        x_scaled = params["x"]
        y_scaled = vars_dict["y"]

        if scaling_enabled:
            x_eval = D_p * x_scaled
            y_eval = D_v * y_scaled
        else:
            x_eval = x_scaled
            y_eval = y_scaled

        return objective_fn(y_eval, x_eval) / D_obj

    builder = (
        HighLevelNLPBuilder(dtype=dtype)
        .add_parameter("x", len(parameter_names))
        .add_variable("y", len(variable_names))
        .set_objective(objective)
        .set_affine_lower_bound(var_name="y", param_name="x", M=lower_M, c=lower_c)
        .set_affine_upper_bound(var_name="y", param_name="x", M=upper_M, c=upper_c)
    )

    if equality_texts:
        eq_fns = [
            make_scalar_eval_fn(
                parse_constraint_text(text)[0],
                parameter_names=parameter_names,
                variable_names=variable_names,
                constants=constants,
            )
            for text in equality_texts
        ]

        def eq_block(params, vars_dict):
            x_scaled = params["x"]
            y_scaled = vars_dict["y"]

            if scaling_enabled:
                x_vec = D_p * x_scaled
                y_vec = D_v * y_scaled
            else:
                x_vec = x_scaled
                y_vec = y_scaled

            out = jnp.concatenate(
                [jnp.ravel(fn(y_vec, x_vec)) for fn in eq_fns], axis=0
            )

            if scaling_enabled and D_eq is not None:
                out = out / D_eq

            return out

        builder = builder.add_nonlinear_equality(eq_block, name="serialized_eq_block")

    if inequality_texts:
        ineq_fns = [
            make_scalar_eval_fn(
                parse_constraint_text(text)[0],
                parameter_names=parameter_names,
                variable_names=variable_names,
                constants=constants,
            )
            for text in inequality_texts
        ]

        def ineq_block(params, vars_dict):
            x_scaled = params["x"]
            y_scaled = vars_dict["y"]

            if scaling_enabled:
                x_vec = D_p * x_scaled
                y_vec = D_v * y_scaled
            else:
                x_vec = x_scaled
                y_vec = y_scaled

            out = jnp.concatenate(
                [jnp.ravel(fn(y_vec, x_vec)) for fn in ineq_fns], axis=0
            )

            if scaling_enabled and D_ineq is not None:
                out = out / D_ineq

            return out

        builder = builder.add_nonlinear_inequality(
            ineq_block, name="serialized_ineq_block"
        )

    return builder.build(
        example_params={"x": jnp.zeros((len(parameter_names),), dtype=dtype)},
        jit_compile=True,
    )
