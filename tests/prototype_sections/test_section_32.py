import numpy as np
import pytest

from clean_core.config import ETA_L, POLE
from clean_core.geometry.lorentz import minkowski_dot
from clean_core.prototypes.sections.section_32 import (
    section_32_step,
    section_32_vector_field,
)


def _tangent_momentum(value=0.4):
    momentum = np.zeros_like(POLE)
    momentum[1] = value
    return momentum


def test_vector_field_matches_conformal_hamiltonian_equations_and_constraint():
    momentum = _tangent_momentum()
    gradient_potential = np.zeros_like(POLE)
    gradient_omega = np.zeros_like(POLE)
    gradient_potential[2] = 0.3
    gradient_omega[3] = -0.2
    omega = 1.5
    kappa_v = 0.25

    mu_dot, momentum_dot, multiplier = section_32_vector_field(
        POLE,
        momentum,
        omega,
        gradient_potential,
        gradient_omega,
        kappa_v=kappa_v,
    )

    kinetic = minkowski_dot(momentum, momentum)
    scale = 1.0 + kappa_v
    np.testing.assert_allclose(mu_dot, scale * momentum / omega**2)
    force_without_constraint = (
        -gradient_potential
        + scale * kinetic * gradient_omega / omega**3
    )
    np.testing.assert_allclose(
        momentum_dot, force_without_constraint + multiplier * ETA_L * POLE
    )
    np.testing.assert_allclose(minkowski_dot(POLE, mu_dot), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        minkowski_dot(POLE, momentum_dot) + minkowski_dot(mu_dot, momentum),
        0.0,
        atol=1e-12,
    )


def test_finite_step_preserves_hyperboloid_and_cotangent_constraints():
    momentum = _tangent_momentum()
    gradient_potential = np.zeros_like(POLE)
    gradient_omega = np.zeros_like(POLE)
    gradient_potential[2] = 0.1

    next_mu, next_momentum = section_32_step(
        POLE,
        momentum,
        0.05,
        omega=1.2,
        gradient_potential=gradient_potential,
        gradient_omega=gradient_omega,
    )

    assert next_mu[0] > 0.0
    np.testing.assert_allclose(minkowski_dot(next_mu, next_mu), -1.0, atol=1e-12)
    np.testing.assert_allclose(
        minkowski_dot(next_mu, next_momentum), 0.0, atol=1e-12
    )
    assert not np.array_equal(next_mu, POLE)
    assert not np.array_equal(next_momentum, momentum)


def test_step_evaluates_callable_hamiltonian_data_at_midpoint():
    seen_mu = []

    def omega_at(mu):
        seen_mu.append(mu.copy())
        return 1.0 + 0.1 * mu[1]

    section_32_step(
        POLE,
        _tangent_momentum(),
        0.02,
        omega=omega_at,
        gradient_potential=lambda mu: np.zeros_like(mu),
        gradient_omega=lambda mu: np.zeros_like(mu),
    )

    assert len(seen_mu) == 2
    assert not np.array_equal(seen_mu[0], seen_mu[1])


@pytest.mark.parametrize(
    ("mu", "momentum", "message"),
    [
        (np.zeros_like(POLE), np.zeros_like(POLE), "hyperboloid"),
        (POLE, np.ones_like(POLE), "cotangent"),
        (POLE, np.full_like(POLE, np.nan), "finite"),
    ],
)
def test_vector_field_rejects_invalid_constrained_state(mu, momentum, message):
    zero = np.zeros_like(POLE)
    with pytest.raises(ValueError, match=message):
        section_32_vector_field(mu, momentum, 1.0, zero, zero)


def test_vector_field_rejects_invalid_parameters_and_gradient_shapes():
    momentum = _tangent_momentum()
    zero = np.zeros_like(POLE)
    with pytest.raises(ValueError, match="omega"):
        section_32_vector_field(POLE, momentum, 0.0, zero, zero)
    with pytest.raises(ValueError, match="gradients"):
        section_32_vector_field(POLE, momentum, 1.0, zero[:-1], zero)
    with pytest.raises(ValueError, match="kappa_v"):
        section_32_vector_field(POLE, momentum, 1.0, zero, zero, kappa_v=np.nan)


def test_step_rejects_nonpositive_or_nonfinite_dt():
    momentum = _tangent_momentum()
    zero = np.zeros_like(POLE)
    for dt in (0.0, -0.1, np.inf):
        with pytest.raises(ValueError, match="dt"):
            section_32_step(POLE, momentum, dt, 1.0, zero, zero)
