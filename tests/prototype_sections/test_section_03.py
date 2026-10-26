import numpy as np
import pytest

from clean_core.config import D_LORENTZ, ETA_L, POLE
from clean_core.geometry.lorentz import minkowski_dot, minkowski_norm_sq
from clean_core.prototypes.sections.section_03 import implicit_midpoint_step


def test_implicit_midpoint_step_solves_midpoint_equations_and_constraints():
    momentum = np.zeros(D_LORENTZ)
    momentum[1] = 0.2
    evaluated_midpoints = []

    def gradient(mu_mid):
        evaluated_midpoints.append(mu_mid.copy())
        force = np.zeros(D_LORENTZ)
        force[1] = 0.03
        return force

    dt = 0.05
    mu_next, p_next = implicit_midpoint_step(POLE, momentum, gradient, dt=dt)

    assert evaluated_midpoints
    np.testing.assert_allclose(minkowski_norm_sq(mu_next), -1.0, atol=1e-9)
    np.testing.assert_allclose(minkowski_dot(mu_next, p_next), 0.0, atol=1e-9)
    np.testing.assert_allclose(mu_next[1], dt * 0.2, atol=5e-5)
    np.testing.assert_allclose(p_next[1], momentum[1] - dt * 0.03, atol=2e-5)

    mu_mid = 0.5 * (POLE + mu_next)
    p_mid = 0.5 * (momentum + p_next)
    np.testing.assert_allclose(evaluated_midpoints[-1], mu_mid, atol=1e-12)

    force = np.zeros(D_LORENTZ)
    force[1] = 0.03
    momentum_base_residual = p_next - momentum + dt * force
    momentum_normal = ETA_L * mu_mid
    lambda_mid = -np.dot(momentum_normal, momentum_base_residual) / (
        dt * np.dot(momentum_normal, momentum_normal)
    )
    np.testing.assert_allclose(
        momentum_base_residual + dt * lambda_mid * momentum_normal,
        0.0,
        atol=1e-9,
    )

    position_base_residual = mu_next - POLE - dt * ETA_L * p_mid
    drift_direction = ETA_L * mu_mid
    drift_multiplier = np.dot(drift_direction, position_base_residual) / np.dot(
        drift_direction, drift_direction
    )
    np.testing.assert_allclose(
        position_base_residual - drift_multiplier * drift_direction,
        0.0,
        atol=1e-9,
    )


def test_implicit_midpoint_step_zero_time_returns_validated_copies():
    momentum = np.zeros(D_LORENTZ)
    result_mu, result_p = implicit_midpoint_step(
        POLE, momentum, lambda _: pytest.fail("gradient must not be evaluated"), dt=0.0
    )

    np.testing.assert_array_equal(result_mu, POLE)
    np.testing.assert_array_equal(result_p, momentum)
    assert result_mu is not POLE
    assert result_p is not momentum


@pytest.mark.parametrize(
    ("mu", "momentum", "gradient", "dt", "message"),
    [
        (np.zeros(D_LORENTZ), np.zeros(D_LORENTZ), lambda _: np.zeros(D_LORENTZ), 0.1, "future unit"),
        (POLE, np.ones(2), lambda _: np.zeros(D_LORENTZ), 0.1, "shape"),
        (POLE, np.full(D_LORENTZ, np.nan), lambda _: np.zeros(D_LORENTZ), 0.1, "finite"),
        (POLE, np.eye(1, D_LORENTZ, 1)[0], lambda _: np.zeros(D_LORENTZ), -0.1, "dt"),
        (POLE, np.eye(1, D_LORENTZ, 0)[0], lambda _: np.zeros(2), 0.1, "tangent"),
    ],
)
def test_implicit_midpoint_step_rejects_invalid_inputs(mu, momentum, gradient, dt, message):
    with pytest.raises(ValueError, match=message):
        implicit_midpoint_step(mu, momentum, gradient, dt=dt)


def test_implicit_midpoint_step_rejects_invalid_force_shape():
    momentum = np.zeros(D_LORENTZ)

    with pytest.raises(ValueError, match="grad_F"):
        implicit_midpoint_step(POLE, momentum, lambda _: np.zeros(2))
