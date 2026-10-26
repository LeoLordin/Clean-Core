import numpy as np
import pytest

from clean_core.config import D_LORENTZ, ETA_L, POLE
from clean_core.geometry.lorentz import minkowski_dot, minkowski_norm_sq
from clean_core.prototypes.algorithms.algorithm_03 import solve_algorithm_03


def test_algorithm_03_solves_midpoint_equations_and_endpoint_constraints():
    momentum = np.zeros(D_LORENTZ)
    momentum[1] = 0.2
    force = np.zeros(D_LORENTZ)
    force[1] = 0.03
    evaluated_midpoints = []

    def gradient(mu_mid):
        evaluated_midpoints.append(mu_mid.copy())
        return force

    dt = 0.05
    mu_next, p_next = solve_algorithm_03(POLE, momentum, gradient, dt=dt)

    assert evaluated_midpoints
    np.testing.assert_allclose(minkowski_norm_sq(mu_next), -1.0, atol=1e-9)
    np.testing.assert_allclose(minkowski_dot(mu_next, p_next), 0.0, atol=1e-9)

    mu_mid = 0.5 * (POLE + mu_next)
    p_mid = 0.5 * (momentum + p_next)
    np.testing.assert_allclose(evaluated_midpoints[-1], mu_mid, atol=1e-12)

    momentum_residual = p_next - momentum + dt * force
    momentum_normal = ETA_L * mu_mid
    lambda_mid = -np.dot(momentum_normal, momentum_residual) / (
        dt * np.dot(momentum_normal, momentum_normal)
    )
    np.testing.assert_allclose(
        momentum_residual + dt * lambda_mid * momentum_normal, 0.0, atol=1e-9
    )

    position_residual = mu_next - POLE - dt * ETA_L * p_mid
    drift_normal = ETA_L * mu_mid
    drift_multiplier = np.dot(drift_normal, position_residual) / np.dot(
        drift_normal, drift_normal
    )
    np.testing.assert_allclose(
        position_residual - drift_multiplier * drift_normal, 0.0, atol=1e-9
    )


def test_algorithm_03_zero_step_returns_copies_without_evaluating_force():
    momentum = np.zeros(D_LORENTZ)
    mu_next, p_next = solve_algorithm_03(
        POLE,
        momentum,
        lambda _: pytest.fail("gradient must not be evaluated"),
        dt=0.0,
    )

    np.testing.assert_array_equal(mu_next, POLE)
    np.testing.assert_array_equal(p_next, momentum)
    assert mu_next is not POLE
    assert p_next is not momentum


@pytest.mark.parametrize("budget", [0, -1, 2001, 1.5, True])
def test_algorithm_03_rejects_unbounded_or_invalid_solver_budgets(budget):
    with pytest.raises(ValueError, match="max_iterations"):
        solve_algorithm_03(
            POLE,
            np.zeros(D_LORENTZ),
            lambda _: np.zeros(D_LORENTZ),
            max_iterations=budget,
        )


def test_algorithm_03_propagates_nonconvergence_as_a_solver_failure():
    momentum = np.zeros(D_LORENTZ)
    momentum[1] = 0.2

    with pytest.raises(RuntimeError, match="did not converge"):
        solve_algorithm_03(
            POLE,
            momentum,
            lambda _: np.zeros(D_LORENTZ),
            dt=0.1,
            max_iterations=1,
        )
