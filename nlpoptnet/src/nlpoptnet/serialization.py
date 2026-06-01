"""Serialization helpers for rebuilding symbolic NLPOptNet problems."""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from jaxmodel import HighLevelNLPBuilder

jax.config.update("jax_enable_x64", True)


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
    """Rebuild a serializable jaxmodel problem from saved metadata."""
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
    D_ineq = None if D_ineq_raw is None else jnp.asarray(D_ineq_raw, dtype=dtype).reshape(-1)

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

            out = jnp.concatenate([jnp.ravel(fn(y_vec, x_vec)) for fn in eq_fns], axis=0)

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

            out = jnp.concatenate([jnp.ravel(fn(y_vec, x_vec)) for fn in ineq_fns], axis=0)

            if scaling_enabled and D_ineq is not None:
                out = out / D_ineq

            return out

        builder = builder.add_nonlinear_inequality(ineq_block, name="serialized_ineq_block")

    return builder.build(example_params={"x": jnp.zeros((len(parameter_names),), dtype=dtype)}, jit_compile=True)
