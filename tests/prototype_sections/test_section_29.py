import numpy as np
import pytest

from clean_core.config import D_SPATIAL, POLE
from clean_core.geometry.lorentz import exp_map
from clean_core.prototypes.sections.section_29 import (
    section_29_spike_encoder,
    section_29_spike_probabilities,
    section_29_ste_surrogate_gradient,
)


def _example_encoder():
    weights = np.zeros((3, D_SPATIAL))
    weights[:, 0] = [2.0, -1.0, 0.5]
    bias = np.array([0.0, 0.2, -0.3])
    return weights, bias


def test_probability_matches_rate_and_poisson_interval_formula():
    tangent = np.zeros_like(POLE)
    tangent[1] = 0.4
    mu = exp_map(POLE.copy(), tangent)
    weights, bias = _example_encoder()

    probabilities = section_29_spike_probabilities(
        mu, weights, bias, f_max=4.0, dt=0.1
    )

    tangent_features = tangent[1:]
    expected_logits = weights @ tangent_features + bias
    expected_rates = 4.0 / (1.0 + np.exp(-expected_logits))
    expected_probabilities = 1.0 - np.exp(-expected_rates * 0.1)
    np.testing.assert_allclose(probabilities, expected_probabilities)


def test_encoder_thresholds_each_probability_against_uniform_input():
    weights, bias = _example_encoder()
    probabilities = section_29_spike_probabilities(
        POLE, weights, bias, f_max=2.0, dt=0.2
    )
    uniform = np.array([0.0, 0.5 * probabilities[1], probabilities[2] + 0.01])

    spikes = section_29_spike_encoder(
        POLE, weights, bias, f_max=2.0, dt=0.2, uniform=uniform
    )

    np.testing.assert_array_equal(spikes, [1, 1, 0])
    assert spikes.dtype == np.uint8


def test_encoder_is_reproducible_with_a_seeded_generator():
    weights, bias = _example_encoder()

    first = section_29_spike_encoder(
        POLE, weights, bias, f_max=2.0, dt=0.2, rng=np.random.default_rng(17)
    )
    second = section_29_spike_encoder(
        POLE, weights, bias, f_max=2.0, dt=0.2, rng=np.random.default_rng(17)
    )

    np.testing.assert_array_equal(first, second)


def test_surrogate_gradient_is_identity_from_probability_to_upstream_gradient():
    gradient = np.array([0.25, -2.0, 3.5])

    result = section_29_ste_surrogate_gradient(gradient)

    np.testing.assert_array_equal(result, gradient)
    assert result is not gradient


@pytest.mark.parametrize(
    "mu",
    [
        np.zeros(D_SPATIAL),
        np.full(POLE.shape, np.nan),
        np.array([-1.0] + [0.0] * D_SPATIAL),
    ],
)
def test_rejects_invalid_lorentz_microstate(mu):
    weights, bias = _example_encoder()

    with pytest.raises(ValueError, match="mu"):
        section_29_spike_probabilities(mu, weights, bias, f_max=1.0, dt=0.1)


def test_rejects_encoder_shape_and_threshold_inputs():
    weights, bias = _example_encoder()

    with pytest.raises(ValueError, match="W_enc"):
        section_29_spike_probabilities(POLE, weights[:, :-1], bias, 1.0, 0.1)
    with pytest.raises(ValueError, match="uniform"):
        section_29_spike_encoder(POLE, weights, bias, 1.0, 0.1, uniform=[0.1])
    with pytest.raises(ValueError, match="both"):
        section_29_spike_encoder(
            POLE, weights, bias, 1.0, 0.1, uniform=np.zeros(3),
            rng=np.random.default_rng(1),
        )


def test_rejects_non_finite_surrogate_gradient():
    with pytest.raises(ValueError, match="upstream_gradient"):
        section_29_ste_surrogate_gradient([1.0, np.inf])
