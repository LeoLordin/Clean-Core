import numpy as np
import pytest

from clean_core.config import D_SPATIAL, ETA_L, POLE
from clean_core.geometry.lorentz import get_boost_frame, minkowski_norm_sq
from clean_core.prototypes.sections.section_01 import (
    stochastic_lie_euler_maruyama_step,
)


def test_section_01_deterministic_step_matches_dlr_and_ito_equations():
    mu = POLE.copy()
    error = np.zeros_like(mu)
    error[1] = 0.2
    frame = get_boost_frame(mu)
    local_error = frame.T @ (ETA_L * error)
    diagonal = np.full(D_SPATIAL, 0.8)
    basis = np.eye(D_SPATIAL, 2)
    values = np.array([0.3, 0.5])
    covariance = np.diag(diagonal) + basis @ np.diag(values) @ basis.T
    precision = np.linalg.inv(covariance)
    energy = local_error @ precision @ local_error
    temperature = 0.4
    alpha_temp = 0.7
    expected_ito = (
        -0.5
        * alpha_temp
        * temperature
        * (1.0 - np.tanh(alpha_temp * energy) ** 2)
        * (precision @ local_error)
    )
    force = np.linspace(-0.1, 0.1, D_SPATIAL)
    dt = 0.002
    epsilon_0 = 0.01
    sigma_beta = 0.6
    adaptive_precision = D_SPATIAL / energy
    expected_velocity = np.linalg.solve(
        np.eye(D_SPATIAL)
        + (dt / epsilon_0) * adaptive_precision * sigma_beta * precision,
        (dt / epsilon_0) * force + dt * expected_ito,
    )

    result = stochastic_lie_euler_maruyama_step(
        mu,
        error,
        force,
        diagonal,
        basis,
        values,
        dt=dt,
        epsilon_0=epsilon_0,
        sigma_beta=sigma_beta,
        temperature=temperature,
        alpha_temp=alpha_temp,
        temperature_max=1.0,
        annealing=0.0,
        rng=np.random.default_rng(2),
    )

    np.testing.assert_allclose(result.precision, precision, atol=1e-12)
    np.testing.assert_allclose(result.local_error, local_error, atol=1e-12)
    np.testing.assert_allclose(result.ito_drift, expected_ito, atol=1e-12)
    np.testing.assert_allclose(result.adaptive_precision, adaptive_precision)
    np.testing.assert_allclose(result.velocity, expected_velocity, atol=1e-12)
    np.testing.assert_allclose(minkowski_norm_sq(result.mu), -1.0, atol=1e-12)
    assert result.mu[0] > 0.0


def test_section_01_noise_is_reproducible_and_speed_is_clamped():
    args = (
        POLE,
        np.zeros_like(POLE),
        np.full(D_SPATIAL, 100.0),
        np.ones(D_SPATIAL),
        np.eye(D_SPATIAL, 1),
        np.ones(1),
    )
    options = dict(dt=0.01, temperature=0.5, max_speed=0.25)
    first = stochastic_lie_euler_maruyama_step(
        *args, **options, rng=np.random.default_rng(5)
    )
    second = stochastic_lie_euler_maruyama_step(
        *args, **options, rng=np.random.default_rng(5)
    )

    np.testing.assert_array_equal(first.raw_velocity, second.raw_velocity)
    np.testing.assert_array_equal(first.mu, second.mu)
    assert np.linalg.norm(first.velocity) <= 0.25 + 1e-12
    np.testing.assert_allclose(minkowski_norm_sq(first.mu), -1.0, atol=1e-12)


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        ({"mu": np.zeros(16)}, "future unit Lorentz hyperboloid"),
        ({"force": np.zeros(14)}, "force must have shape"),
        ({"d_diag": np.zeros(15)}, "d_diag entries must be positive"),
        ({"dt": 0.0}, "dt must be finite and positive"),
        ({"temperature": -1.0}, "temperature must be finite and non-negative"),
    ],
)
def test_section_01_rejects_invalid_inputs(changed, message):
    args = {
        "mu": POLE.copy(),
        "prediction_error": np.zeros_like(POLE),
        "force": np.zeros(D_SPATIAL),
        "d_diag": np.ones(D_SPATIAL),
        "low_rank_basis": np.eye(D_SPATIAL, 1),
        "low_rank_values": np.ones(1),
        "dt": 0.01,
        "temperature": 0.0,
    }
    args.update(changed)
    positional = [
        args.pop(key)
        for key in (
            "mu",
            "prediction_error",
            "force",
            "d_diag",
            "low_rank_basis",
            "low_rank_values",
        )
    ]
    with pytest.raises(ValueError, match=message):
        stochastic_lie_euler_maruyama_step(*positional, **args)
