"""Scaled callable problems and their serialized symbolic counterparts."""

import jax.numpy as jnp
import numpy as np
import pytest

from nlpoptnet import NLPOptNet
from nlpoptnet.serialization.definition import (
    build_model_from_problem_spec,
    make_scalar_eval_fn,
    parse_constraint_text,
)


@pytest.mark.parametrize("use_scaling", [False, True])
def test_callable_blocks_receive_original_coordinates(use_scaling):
    model = NLPOptNet(
        type="nlp",
        config={
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 1e-3,
            "hidden_size": 4,
            "hidden_layers": 1,
            "jit_warmup": False,
        },
    )
    p = model.add_parameter("p")
    y = model.add_variable("u")
    model.objective(
        lambda params, values: (jnp.ravel(values["y"])[0] - params["x"][0]) ** 2
    )
    model.constraints.equality.add(
        y.u - p.p,
        lambda params, values: values["y"] + params["x"],
    )
    model.constraints.inequality.add(
        y.u**2 - 4,
        lambda params, values: values["y"] - 3 * params["x"],
    )
    model.box(lower=1, upper=2, num_samples=10)
    if use_scaling:
        model.scale(mode="jacobian", probe_points=2, use_jacobian_scaling=True)
    model.build()
    assert not model._problem_is_serializable()
    state = model._build_state
    dp, dv = model.scaling.D_p, model.scaling.D_v
    x = {"x": jnp.asarray([2.0] / dp)}
    at = jnp.asarray([1.0] / dv)
    assert float(
        state.model.objective_value(x, at)
    ) * model.scaling.D_obj == pytest.approx(1)
    for kind, expected in [("eq", [-1, 3]), ("ineq", [-3, -5])]:
        residual = getattr(state.model, f"{kind}_residual")(x, at)
        factor = getattr(model.scaling, f"D_{kind}")
        np.testing.assert_allclose(
            residual * (1 if factor is None else factor), expected
        )


def test_saved_expression_helpers():
    params = dict(parameter_names=["p"], variable_names=["u", "v"])
    constants = {
        "A": [[2, 0], [0, 3]],
        "Q": np.eye(2),
        "Qs": [np.eye(2)],
        "label": "qp",
    }
    x, y = [2], [3, 4]
    for text, expected in [
        ("lin(A, y)", [6, 12]),
        ("batch_lin(A, y)", [6, 12]),
        ("quad(Q, y)", 25),
        ("batch_quad(Qs, y)", [25]),
        ("batch_exp(y)", np.exp(y)),
        ("label", "qp"),
    ]:
        value = make_scalar_eval_fn(text, **params, constants=constants)(y, x)
        np.testing.assert_equal(value, expected) if isinstance(
            expected, str
        ) else np.testing.assert_allclose(value, expected)
    assert parse_constraint_text("u >= p") == ("(p) - (u)", "ineq")
    with pytest.raises(ValueError, match="Unsupported constraint"):
        parse_constraint_text("u < p")


@pytest.mark.parametrize("with_blocks", [False, True])
def test_rebuilt_scaled_spec(with_blocks):
    spec = {
        "parameter_names": ["p"],
        "variable_names": ["u"],
        "objective_text": "(u-p)**2",
        "bounds": {"lower_M": [[0]], "lower_c": [0], "upper_M": [[0]], "upper_c": [5]},
        "scaling": {
            "enabled": True,
            "D_p": [2],
            "D_v": [4],
            "D_obj": 2,
            "D_eq": [4] if with_blocks else None,
            "D_ineq": [2] if with_blocks else None,
        },
        "equality_texts": ["u == p"] if with_blocks else [],
        "inequality_texts": ["u >= p"] if with_blocks else [],
    }
    model = build_model_from_problem_spec(spec, constants={})
    x = {"x": jnp.array([1.5])}
    y = jnp.array([0.25])
    np.testing.assert_allclose(model.objective_value(x, y), 2)
    np.testing.assert_allclose(model.eq_residual(x, y), [-0.5] if with_blocks else [])
    np.testing.assert_allclose(model.ineq_residual(x, y), [1] if with_blocks else [])


def test_build_validates_required_definition():
    model = NLPOptNet()
    with pytest.raises(ValueError, match="Set the objective"):
        model.build()
    model.objective(2)
    with pytest.raises(ValueError, match="at least one parameter"):
        model.build()
    p = model.add_parameter("p")
    with pytest.raises(ValueError, match="at least one variable"):
        model.build()
    y = model.add_variable("u")
    model.objective(lambda: (y.u - p.p) ** 2)
    assert model._problem_is_serializable()
    model.constraints.equality.add(lambda params, values: values["y"] - params["x"])
    assert not model._problem_is_serializable()
    with pytest.raises(ValueError, match="Missing required config"):
        model.build()


def test_unbounded_symbolic_problem_uses_default_variable_scale():
    model = NLPOptNet(
        type="qp",
        config={
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 1e-3,
            "hidden_size": 4,
            "hidden_layers": 1,
            "jit_warmup": False,
        },
    )
    model.add_parameter("p")
    y = model.add_variable("u")
    model.objective(3 * y.u + 2)
    model.box(lower=1, upper=2, num_samples=10)
    model.scale(probe_points=1)
    model.build()
    np.testing.assert_allclose(model.scaling.D_v, [1])
    assert model.scaling.D_obj == pytest.approx(3)
    assert model.scaling.D_eq is None
    assert model.scaling.D_ineq is None


def test_invalid_manual_scaling_size():
    model = NLPOptNet()
    with pytest.raises(ValueError, match="must have shape"):
        model._positive_scale([1, 2], size=3)
    np.testing.assert_allclose(model._positive_scale([-2, 0, np.nan]), [2, 1, 1])
