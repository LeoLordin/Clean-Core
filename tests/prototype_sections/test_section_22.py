import numpy as np
import pytest

from clean_core.config import D_SPATIAL, ETA_L, POLE
from clean_core.geometry.lorentz import exp_map, get_boost_frame
from clean_core.prototypes.sections.section_22 import (
    thermodynamic_langevin_frame_noise,
    thermodynamic_temperature,
)


def test_temperature_matches_bounded_energy_formula():
    error = np.zeros(D_SPATIAL)
    error[0] = 2.0
    precision = np.eye(D_SPATIAL)

    assert thermodynamic_temperature(error, precision, 0.25) == pytest.approx(
        np.tanh(1.0)
    )
    assert thermodynamic_temperature(error, precision, 0.0) == 0.0
    assert thermodynamic_temperature(error, -precision, 2.0) == 0.0


def test_frame_noise_is_reproducible_finite_and_tangent():
    tangent = np.zeros(D_SPATIAL + 1)
    tangent[2] = 0.4
    mu = exp_map(POLE.copy(), tangent)
    error = np.ones(D_SPATIAL)
    precision = np.eye(D_SPATIAL)
    args = (mu, error, precision, 0.1, 0.01)

    first = thermodynamic_langevin_frame_noise(
        *args, epsilon_0=0.01, rng=np.random.default_rng(22)
    )
    second = thermodynamic_langevin_frame_noise(
        *args, epsilon_0=0.01, rng=np.random.default_rng(22)
    )

    np.testing.assert_array_equal(first, second)
    assert np.all(np.isfinite(first))
    np.testing.assert_allclose(mu @ (ETA_L * first), 0.0, atol=1e-12)


def test_local_frame_variance_matches_discrete_scale_and_annealing():
    error = np.zeros(D_SPATIAL)
    error[0] = 1.0
    precision = np.eye(D_SPATIAL)
    alpha_temp = 0.4
    temperature = thermodynamic_temperature(error, precision, alpha_temp)
    expected_variance = 2.0 * temperature * 0.05**2
    n_samples = 4096

    def samples(annealing):
        rng = np.random.default_rng(7)
        frame = get_boost_frame(POLE)
        local = np.stack(
            [
                frame.T
                @ (ETA_L * thermodynamic_langevin_frame_noise(
                    POLE,
                    error,
                    precision,
                    alpha_temp,
                    dt=0.01,
                    epsilon_0=0.01,
                    annealing=annealing,
                    rng=rng,
                ))
                for _ in range(n_samples)
            ]
        )
        return local

    base = samples(1.0)
    doubled = samples(2.0)
    np.testing.assert_allclose(np.var(base, axis=0), expected_variance, rtol=0.06)
    np.testing.assert_allclose(np.var(doubled, axis=0), 4.0 * expected_variance, rtol=0.06)


@pytest.mark.parametrize(
    ("error", "precision", "alpha_temp"),
    [
        (np.zeros(D_SPATIAL - 1), np.eye(D_SPATIAL), 1.0),
        (np.zeros(D_SPATIAL), np.eye(D_SPATIAL - 1), 1.0),
        (np.full(D_SPATIAL, np.nan), np.eye(D_SPATIAL), 1.0),
        (np.zeros(D_SPATIAL), np.eye(D_SPATIAL), -1.0),
    ],
)
def test_temperature_rejects_invalid_inputs(error, precision, alpha_temp):
    with pytest.raises(ValueError):
        thermodynamic_temperature(error, precision, alpha_temp)


def test_frame_noise_rejects_invalid_state():
    with pytest.raises(ValueError, match="hyperboloid"):
        thermodynamic_langevin_frame_noise(
            np.zeros_like(POLE),
            np.ones(D_SPATIAL),
            np.eye(D_SPATIAL),
            1.0,
            dt=0.01,
        )
