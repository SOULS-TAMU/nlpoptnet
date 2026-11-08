"""Verify algebra, variable layout and parameter sampling against known values."""

import jax.numpy as jnp
import numpy as np
import pytest

from nlpoptnet import NLPOptNet
from nlpoptnet.api.formulation.expressions import _EvalContext
from nlpoptnet.api.formulation.parameters import ParameterSpec
from nlpoptnet.api.formulation.variables import VariableBuilder


def test_symbolic_affine_and_quadratic_operations(tmp_path):
    model = NLPOptNet()
    params = model.add_parameter(["p", "q"])
    variables = model.add_variable(["u", "v"])
    ctx = _EvalContext(
        jnp.array([2.0, 3.0]), jnp.array([4.0, 5.0]), {"u": 0, "v": 1}, {"p": 0, "q": 1}
    )
    x = params.vector()
    y = variables.vector()
    file = tmp_path / "coefficients.npz"
    np.savez(file, A=np.array([[2.0, 1.0], [0.0, 3.0]]))
    assert "A" in model.extract(file)
    a = model.A
    np.testing.assert_allclose(model.lin(a, x).eval(ctx), [13.0, 15.0])
    np.testing.assert_allclose(model.batch_lin(a, y).eval(ctx), [7.0, 9.0])
    np.testing.assert_allclose(model.lin([2.0, 1.0], x).eval(ctx), 13.0)
    np.testing.assert_allclose(model.quad(np.eye(2), y).eval(ctx), 13.0)
    np.testing.assert_allclose(
        model.batch_quad([np.eye(2), 2 * np.eye(2)], y).eval(ctx), [13.0, 26.0]
    )
    np.testing.assert_allclose(model.batch_exp(y).eval(ctx), np.exp([2, 3]))
    scalar = -((params.p + 1) / 5) + (2 * variables.u - 1)
    np.testing.assert_allclose(scalar.eval(ctx), 2.0)
    np.testing.assert_allclose(((x + y) * 2 - y).eval(ctx), [10, 13])


@pytest.mark.parametrize(
    ("operator", "expected"),
    [
        ("sin", np.sin(2)),
        ("cos", np.cos(2)),
        ("exp", np.exp(2)),
        ("log", np.log(2)),
        ("sqrt", np.sqrt(2)),
        ("abs", 2),
    ],
)
def test_symbolic_elementwise_operations(operator, expected):
    model = NLPOptNet()
    variable = model.add_variable("u")
    ctx = _EvalContext(jnp.array([2.0]), jnp.zeros(0), {"u": 0}, {})
    np.testing.assert_allclose(getattr(model, operator)(variable.u).eval(ctx), expected)


def test_variable_blocks_roundtrip_and_shape_errors():
    builder = VariableBuilder().add_scalar("a").add_vector("b", 2)
    builder.add_matrix("c", (1, 2)).add_tensor("d", (1, 1, 2))
    values = {"a": 1.0, "b": [2.0, 3.0], "c": [[4.0, 5.0]], "d": [[[6.0, 7.0]]]}
    layout = builder.build()
    flat = layout.pack(values)
    np.testing.assert_array_equal(flat, np.arange(1.0, 8.0))
    for name, value in layout.unpack(flat).items():
        np.testing.assert_array_equal(value, values[name])
    with pytest.raises(KeyError, match="Missing variable"):
        layout.pack({})
    with pytest.raises(ValueError, match="expected size"):
        layout.pack(dict(values, b=[0.0]))
    with pytest.raises(ValueError, match="Expected flat vector"):
        layout.unpack(flat[None, :])
    with pytest.raises(ValueError, match="Expected vector length"):
        layout.unpack(flat[:-1])
    with pytest.raises(ValueError, match="already exists"):
        builder.add_scalar("a")


def test_parameter_shapes_are_enforced():
    spec = ParameterSpec.from_example({"x": [1.0, 2.0], "p": 2.0})
    spec.validate({"x": [2.0, 3.0], "p": 0.0})
    with pytest.raises(KeyError, match="Missing parameter"):
        spec.validate({"x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="expected shape"):
        spec.validate({"x": [1.0], "p": 0.0})


def test_symbolic_simplex_sampling():
    model = NLPOptNet(config={"seed": 42})
    p = model.add_parameter(["p", "q"])
    model.simplex(p.p >= 0, p.q >= 0, p.p + p.q <= 1, num_samples=16)
    samples = model._load_or_sample_parameters(n_x=2)
    assert samples.shape == (16, 2)
    assert np.all(samples >= -1e-10)
    assert np.all(samples.sum(axis=1) <= 1 + 1e-10)
    np.testing.assert_array_equal(samples, model._load_or_sample_parameters(n_x=2))


@pytest.mark.parametrize("automatic", [False, True])
def test_scaling_preserves_physical_coordinates(automatic):
    model = NLPOptNet(
        config={
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 1e-3,
            "hidden_size": 8,
            "hidden_layers": 1,
            "device": "cpu",
        }
    )
    p = model.add_parameter("p")
    y = model.add_variable("y")
    model.objective((y.y - p.p) ** 2)
    model.constraints.equality.add(y.y == p.p)
    model.constraints.inequality.add(y.y <= 5)
    model.constraints.box.add(y.y >= 0, y.y <= 10)
    model.box(lower=1, upper=3, num_samples=20)
    if automatic:
        model.scale(probe_points=3)
    else:
        model.scale(
            parameter=[2], variable=[4], objective=2, equality=[4], inequality=[4]
        )
    model.build()
    for scale in (model.scaling.D_p, model.scaling.D_v):
        assert scale.shape == (1,) and np.all(np.isfinite(scale)) and np.all(scale > 0)
    x_original = np.array([[2.0]])
    np.testing.assert_allclose(
        model._scaled_parameters(x_original) * model.scaling.D_p, x_original
    )
    np.testing.assert_allclose(
        model._unscale_variables(x_original / model.scaling.D_v), x_original
    )
    if not automatic:
        np.testing.assert_allclose(model.scaling.D_p, [2])
        np.testing.assert_allclose(model.scaling.D_v, [4])
