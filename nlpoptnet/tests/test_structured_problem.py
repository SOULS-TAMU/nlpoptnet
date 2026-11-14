"""Numerical contracts for structured problems and their local approximations."""

import jax.numpy as jnp
import numpy as np
import pytest

from nlpoptnet.api.formulation.approximation import QuadraticObjective
from nlpoptnet.api.formulation.builder import HighLevelNLPBuilder, NLPBuilder
from nlpoptnet.api.formulation.constraints import (
    nonlinear_eq_from_parts,
    nonlinear_ineq_from_parts,
)
from nlpoptnet.api.formulation.parameters import ParameterSpec
from nlpoptnet.api.formulation.variables import VariableBuilder


def test_mixed_constraint_problem_and_sqp_data():
    x = {"x": jnp.array([2.0])}
    y = jnp.array([2.0, 3.0])
    eye = jnp.eye(2)
    Q = jnp.array([[4.0, 1.0], [1.0, 2.0]])
    builder = HighLevelNLPBuilder().add_parameter("x", 1)
    builder.add_variable("y", 2)
    builder.set_quadratic_objective(Q, jnp.array([1.0, -1.0]), constant=2)
    builder.add_affine_equality(
        var_name="y",
        A=jnp.array([[1.0, 1.0]]),
        rhs_const=1,
        param_terms=[(jnp.array([[2.0]]), "x")],
    )
    builder.add_affine_inequality(var_name="y", C=eye, rhs_const=4)
    builder.add_quadratic_equality(
        Q=eye,
        c=jnp.zeros(2),
        x_coeff=jnp.array([1.0]),
        x_name="x",
        rhs_const=1,
    )
    builder.add_quadratic_inequality(Q=eye, c=jnp.zeros(2), rhs_const=7)
    builder.set_affine_lower_bound(
        var_name="y",
        param_name="x",
        M=jnp.array([[1.0], [0.0]]),
        c=jnp.zeros(2),
    )
    builder.set_affine_upper_bound(
        var_name="y",
        param_name="x",
        M=jnp.zeros((2, 1)),
        c=jnp.full(2, 4.0),
    )
    model = builder.build(example_params=x, jit_compile=False)
    model.validate_params(x)
    np.testing.assert_allclose(model.pack_vars({"y": y}), y)
    np.testing.assert_allclose(model.unpack_vars(y)["y"], y)
    assert model.objective_value(x, y) == 24
    np.testing.assert_allclose(model.grad_y_objective(x, y), [12, 7])
    np.testing.assert_allclose(model.hess_y_objective(x, y), Q)
    np.testing.assert_allclose(model.diag_hess_y_objective(x, y), [4, 2])
    np.testing.assert_allclose(model.eq_residual(x, y), [0, 3.5])
    np.testing.assert_allclose(model.ineq_residual(x, y), [-2, -1, -0.5])
    np.testing.assert_allclose(model.jac_y_eq(x, y), [[1, 1], [2, 3]])
    np.testing.assert_allclose(model.jac_y_ineq(x, y), [[1, 0], [0, 1], [2, 3]])
    np.testing.assert_allclose(model.bound_residuals(x, y), [[0, -3], [-2, -1]])
    linear = model.linearize_constraints(x, y)
    structured = model.linearization_data(x, y)
    np.testing.assert_allclose(linear["A"] @ y - linear["b"], structured.h_val)
    np.testing.assert_allclose(linear["C"] @ y - linear["d"], structured.g_val)
    local = model.quadraticize_objective(x, y, use_diagonal_hessian=False)
    np.testing.assert_allclose(local["Q"], Q)
    np.testing.assert_allclose(local["c"], [1, -1])
    local = model.quadratic_objective_data(x, y, rho=0.5, diag_floor=3)
    np.testing.assert_allclose(local.Q_diag, [2, 1.5])
    packed = model.sqp_data(x, y)
    structured = model.sqp_subproblem_data(x, y)
    np.testing.assert_allclose(packed["Q"], structured.objective.Q)
    np.testing.assert_allclose(packed["l"], [2, 0])
    np.testing.assert_allclose(packed["u"], [4, 4])


def test_named_nonlinear_parts_and_empty_bounds():
    variables = VariableBuilder().add_scalar("a").add_vector("b", 1).build()
    x = {"x": jnp.array([1.0])}
    builder = NLPBuilder(
        var_spec=variables, parameter_spec=ParameterSpec.from_example(x)
    )
    builder.set_objective(lambda p, v: v["a"] ** 2 + jnp.sum(v["b"] ** 2))
    terms = dict(
        var_spec=variables,
        linear_y_terms=[(jnp.ones((1, 1)), "a")],
        nonlinear_y_terms=[(jnp.ones((1, 1)), jnp.square, "b")],
        rhs_const=jnp.array([2.0]),
        x_blocks=[(jnp.array([[3.0]]), "x")],
    )
    builder.add_constraints(
        [
            nonlinear_eq_from_parts(**terms, x_direct_name="x"),
            nonlinear_ineq_from_parts(**terms),
        ]
    )
    builder.add_eq_scalar(lambda p, v: v["a"] - 2)
    builder.add_ineq_scalar(lambda p, v: v["a"] - 3)
    builder.add_eq_block(lambda p, v: v["b"] - 3)
    builder.add_ineq_block(lambda p, v: v["b"] - 4)
    model = builder.build(jit_compile=False)
    y = jnp.array([2.0, 3.0])
    np.testing.assert_allclose(model.eq_residual(x, y), [5, 0, 0])
    np.testing.assert_allclose(model.ineq_residual(x, y), [6, -1, -1])
    np.testing.assert_allclose(model.jac_y_eq(x, y), [[1, 6], [1, 0], [0, 1]])
    assert model.lower_bounds(x) is None and model.upper_bounds(x) is None
    assert all(values.size == 0 for values in model.bound_residuals(x, y))
    assert model.sqp_subproblem_data(x, y).l is None


def test_structured_builder_validates_shapes():
    with pytest.raises(ValueError, match="square"):
        QuadraticObjective(jnp.ones((1, 2)), jnp.zeros(1))
    with pytest.raises(ValueError, match="must match"):
        QuadraticObjective(jnp.eye(2), jnp.zeros(1))
    builder = HighLevelNLPBuilder().add_variable("y", 1).add_parameter("x", 1)
    with pytest.raises(ValueError, match="Objective"):
        builder.build(example_params={})
    builder.set_quadratic_objective(jnp.eye(1), jnp.zeros(1))
    with pytest.raises(KeyError, match="Missing parameter"):
        builder.build(example_params={})
    with pytest.raises(ValueError, match="expected shape"):
        builder.build(example_params={"x": jnp.zeros(2)})
    builder.add_quadratic_equality(Q=jnp.eye(2), c=jnp.zeros(2))
    with pytest.raises(ValueError, match="Q"):
        builder.build(example_params={"x": jnp.zeros(1)})
