"""Regression tests for original-coordinate certificates and stopping."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from nlpoptnet.api.configuration import _ScalingState
from nlpoptnet.api.inference_diagnostics import _certificate, _run_layers


def test_scaled_certificate_accounts_for_active_bounds_and_nonlinear_residual():
    # Original problem: minimize 5*x - 3*z, subject to z**2 <= 1 and x >= 0.
    # y=(0, 1) is optimal with lower dual 5 and nonlinear inequality dual 1.5.
    model = SimpleNamespace(
        eq_residual=lambda p, y: jnp.empty(0),
        ineq_residual=lambda p, y: jnp.array([((4 * y[1]) ** 2 - 1) / 2]),
        lower_bounds=lambda p: jnp.array([0.0, -jnp.inf]),
        upper_bounds=lambda p: None,
        grad_y_objective=lambda p, y: jnp.array([10.0, -12.0]) / 8,
        jac_y_eq=lambda p, y: jnp.empty((0, 2)),
        jac_y_ineq=lambda p, y: jnp.array([[0.0, 16 * y[1]]]),
    )
    scaling = _ScalingState(D_obj=8, D_v=np.array([2, 4]), D_ineq=np.array([2]))
    residuals, duals = _certificate(
        model,
        scaling,
        jnp.empty(0),
        jnp.array([0.0, 0.25]),
        jnp.empty(0),
        jnp.array([0.375]),
        1e-8,
    )
    np.testing.assert_allclose(list(residuals.values()), 0, atol=1e-10)
    np.testing.assert_allclose(duals["lower"], [5, 0])
    np.testing.assert_allclose(duals["inequality"], [1.5])

    residuals, _ = _certificate(
        model,
        scaling,
        jnp.empty(0),
        jnp.array([-0.1, 0.5]),
        jnp.empty(0),
        jnp.array([-0.375]),
        1e-8,
    )
    np.testing.assert_allclose(residuals["bounds"], 0.2)
    np.testing.assert_allclose(residuals["primal"], 3.0)
    np.testing.assert_allclose(residuals["dual"], 1.5)
    np.testing.assert_allclose(residuals["complementarity"], 4.5)


def test_adaptive_batch_removes_feasible_samples_and_respects_cap():
    inputs = jnp.array([[1.0], [2.0], [4.0]])
    calls = []

    def step(x, y, eq, iq):
        calls.append(len(x))
        return y + 1, eq, iq, jnp.full(len(x), 7)

    def primal(x, y):
        return jnp.maximum(x[:, 0] - y[:, 0], 0)

    initial = (jnp.zeros_like(inputs), jnp.empty((3, 0)), jnp.empty((3, 0)))
    solution, used, counts = _run_layers(step, primal, inputs, initial, 3, True, 0)
    np.testing.assert_array_equal(used, [1, 2, 3])
    np.testing.assert_array_equal(solution[0][:, 0], [1, 2, 3])
    np.testing.assert_array_equal(counts, [[7, 0, 0], [7, 7, 0], [7, 7, 7]])
    assert calls == [3, 2, 1]
