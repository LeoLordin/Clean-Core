import numpy as np
import pytest

from clean_core.prototypes.sections.section_11 import sigmoidal_spike_activation_mask


def test_section_11_mask_matches_the_written_sigmoid_energy_equation():
    error = np.array([0.5, -0.25])
    precision = np.diag([2.0, 4.0])
    threshold = 0.2
    beta = 3.0
    energy = error @ precision @ error
    expected = 1.0 / (1.0 + np.exp(-beta * (energy - threshold)))

    actual = sigmoidal_spike_activation_mask(error, precision, threshold, beta)

    assert actual == pytest.approx(expected)


def test_section_11_mask_has_expected_threshold_and_steepness_limits():
    error = np.array([2.0])
    precision = np.eye(1)

    assert sigmoidal_spike_activation_mask(error, precision, threshold=4.0, beta=100.0) == 0.5
    assert sigmoidal_spike_activation_mask(error, precision, threshold=5.0, beta=100.0) < 1e-40
    assert sigmoidal_spike_activation_mask(error, precision, threshold=3.0, beta=100.0) > 1.0 - 1e-12


def test_section_11_mask_is_stable_for_extreme_finite_logits():
    error = np.array([1.0])
    precision = np.eye(1)

    assert sigmoidal_spike_activation_mask(
        error, precision, threshold=-1000.0, beta=10.0
    ) == 1.0
    assert sigmoidal_spike_activation_mask(
        error, precision, threshold=1000.0, beta=10.0
    ) == 0.0


@pytest.mark.parametrize(
    ("error", "precision", "threshold", "beta", "message"),
    [
        ([1.0, 2.0], np.eye(1), 0.0, 1.0, "square and match"),
        ([], np.empty((0, 0)), 0.0, 1.0, "square and match"),
        ([1.0], [[np.nan]], 0.0, 1.0, "finite"),
        ([1.0, 0.0], [[1.0, 0.5], [0.0, 1.0]], 0.0, 1.0, "symmetric"),
        ([1.0], [[-1.0]], 0.0, 1.0, "positive semidefinite"),
        ([1.0], np.eye(1), np.inf, 1.0, "threshold"),
        ([1.0], np.eye(1), [0.0], 1.0, "threshold"),
        ([1.0], np.eye(1), 0.0, 0.0, "beta"),
        ([1.0], np.eye(1), 0.0, -1.0, "beta"),
        ([1.0], np.eye(1), 0.0, np.inf, "beta"),
        ([1.0], np.eye(1), 0.0, [1.0], "beta"),
    ],
)
def test_section_11_mask_rejects_invalid_inputs(error, precision, threshold, beta, message):
    with pytest.raises(ValueError, match=message):
        sigmoidal_spike_activation_mask(error, precision, threshold, beta)
