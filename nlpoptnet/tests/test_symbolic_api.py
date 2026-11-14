"""Evaluate symbolic operators and affine bounds against explicit arithmetic."""

import jax.numpy as jnp
import numpy as np
import pytest

from nlpoptnet import NLPOptNet
from nlpoptnet.api.formulation.expressions import _EvalContext


def test_symbolic_arithmetic_and_constant_dispatch():
    model = NLPOptNet()
    p = model.add_parameter(["a", "b"])
    y = model.add_variable(["u", "v"])
    ctx = _EvalContext(
        jnp.array([4.0, 5.0]), jnp.array([2.0, 3.0]), {"u": 0, "v": 1}, {"a": 0, "b": 1}
    )
    scalar = model.vector(2.0)
    vector = model.vector([1.0, 2.0])
    matrix = model.matrix([[1.0, 0.0], [0.0, 2.0]])
    np.testing.assert_equal(np.asarray(scalar), 2)
    np.testing.assert_allclose(matrix @ np.array([4, 5]), [4, 10])
    np.testing.assert_allclose(np.array([4, 5]) @ matrix, [4, 10])
    np.testing.assert_equal(np.asarray(vector), [1, 2])
    assert scalar.name in repr(scalar)
    for expr, expected in [
        (3 + p.a, 5),
        (3 - p.a, 1),
        (12 / p.b, 4),
        (-p.a, -2),
        (scalar + y.u, 6),
        (y.u + scalar, 6),
        (scalar - y.u, -2),
        (y.u - scalar, 2),
        (scalar * y.u, 8),
        (y.u * scalar, 8),
        (scalar / y.u, 0.5),
        (y.u / scalar, 2),
        (-scalar, -2),
        (1 + p.vector(), [3, 4]),
        (10 - p.vector(), [8, 7]),
        (2 * p.vector(), [4, 6]),
        (-p.vector(), [-2, -3]),
        (y.vector() / 2, [2, 2.5]),
        (-y.vector(), [-4, -5]),
        (model.abs(y.vector() - vector), [3, 3]),
        (p.vector() - p.vector(), [0, 0]),
        (p.vector() + p.vector(), [4, 6]),
    ]:
        np.testing.assert_allclose(expr.eval(ctx), expected)
    for constraint, expected in [
        (y.vector() <= vector, [3, 3]),
        (y.vector() >= vector, [-3, -3]),
        (y.vector() == 4, [0, 1]),
        (scalar <= y.u, -2),
        (scalar >= y.u, 2),
        (scalar == y.u, -2),
    ]:
        np.testing.assert_allclose(constraint.residual(ctx), expected)


def test_vector_bounds_and_symbolic_constraint_shorthand():
    model = NLPOptNet()
    p = model.add_parameter(["a", "b"])
    y = model.add_variable(["u", "v"])
    model.constraints.box.add(var=y, lower=p.a, upper=model.vector([8]))
    model.constraints.box.add(p.b <= y.u, p.b + 4 >= y.v)
    lower_M, lower_c, upper_M, upper_c = model._build_box_bounds(n_x=2, n_y=2)
    x = np.array([2, 3])
    np.testing.assert_allclose(lower_M @ x + lower_c, [3, 2])
    np.testing.assert_allclose(upper_M @ x + upper_c, [8, 7])
    model.constraints.equality.add(lambda: y.vector() - p.vector())
    model.constraints.inequality.add(y.u - p.a)
    ctx = _EvalContext(
        jnp.array([4.0, 5.0]), jnp.array([2.0, 3.0]), {"u": 0, "v": 1}, {"a": 0, "b": 1}
    )
    np.testing.assert_allclose(
        model.constraints.equality.items[0].residual(ctx), [2, 2]
    )
    np.testing.assert_allclose(model.constraints.inequality.items[0].residual(ctx), 2)


def test_symbolic_rejects_invalid_types_and_shapes(tmp_path):
    model = NLPOptNet()
    p = model.add_parameter(["a"])
    y = model.add_variable(["u", "v"])
    with pytest.raises(AttributeError, match="Unknown variable"):
        _ = y.missing
    with pytest.raises(FileNotFoundError, match="problem.npz"):
        model.extract(tmp_path / "missing.npz")
    with pytest.raises(ValueError, match="Expected a eq"):
        model.constraints.equality.add(y.u <= 1)
    with pytest.raises(TypeError, match="Unsupported constraint"):
        model.constraints.inequality.add(object())
    with pytest.raises(TypeError, match="bound inequalities"):
        model.constraints.box.add(y.u == 2)
    with pytest.raises(ValueError, match="requires var"):
        model.constraints.box.add(lower=0)
    with pytest.raises(ValueError, match="variable namespace"):
        model.constraints.box.add(var=p, lower=0)
    with pytest.raises(ValueError, match="affine"):
        model.constraints.box.add(var=y, lower=y.u)
    with pytest.raises(ValueError, match="affine"):
        model.constraints.box.add(p.a**2 <= y.v)
    with pytest.raises(ValueError, match="affine"):
        model.constraints.box.add(y.u <= p.a**2)
    with pytest.raises(ValueError, match="single variable"):
        model.constraints.box.add(y.u + y.v <= 1)
    with pytest.raises(ValueError, match="Vector shape"):
        model.constraints.box.add(var=y, upper=p)
    with pytest.raises(ValueError, match="Vector shape"):
        _ = y.vector() + p.vector()
    with pytest.raises(ValueError, match="Vector shape"):
        _ = y.vector() + model.vector([0, 0, 0])
    with pytest.raises(TypeError, match="target size"):
        model.lin([1], y.u)
    with pytest.raises(TypeError, match="Expected a scalar"):
        _ = y.u + [1, 2]
    with pytest.raises(ValueError, match="1D or 2D"):
        model.lin(model.tensor(np.ones((1, 2, 2))), y)


def test_duplicate_and_empty_symbol_names_are_rejected():
    model = NLPOptNet()
    with pytest.raises(ValueError, match="At least one name"):
        model.add_parameter([])
    with pytest.raises(ValueError, match="Names must be unique"):
        model.add_variable(["u", "u"])
    model.add_parameter("x")
    with pytest.raises(ValueError, match="Duplicate symbol name"):
        model.add_parameter("x")
    with pytest.raises(ValueError, match="Duplicate symbol name"):
        model.add_variable("x")
