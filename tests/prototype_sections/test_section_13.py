import numpy as np
import pytest

from clean_core.prototypes.sections.section_13 import (
    spectral_lipschitz_penalty,
    spectral_norm_estimate,
)


def test_spectral_norm_estimate_matches_largest_singular_value():
    weights = np.array([[3.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    assert spectral_norm_estimate(weights) == pytest.approx(3.0)
    assert spectral_norm_estimate(np.zeros((2, 3))) == 0.0


def test_penalty_uses_largest_singular_value_and_coefficient():
    weights = np.diag([2.0, 0.5])

    assert spectral_lipschitz_penalty([weights]) == pytest.approx(1.0)
    assert spectral_lipschitz_penalty([weights], lambda_lip=0.25) == pytest.approx(0.25)


def test_penalty_sums_all_matrices_including_rectangular_weights():
    first = np.array([[3.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    second = np.array([[0.0], [1.5]])

    assert spectral_lipschitz_penalty([first, second]) == pytest.approx(4.25)


def test_penalty_is_zero_at_or_below_the_unit_threshold():
    weights = np.diag([1.0, 0.25])

    assert spectral_lipschitz_penalty([weights]) == 0.0
    assert spectral_lipschitz_penalty([]) == 0.0
    assert spectral_lipschitz_penalty([weights], lambda_lip=0.0) == 0.0


@pytest.mark.parametrize("coefficient", [-1.0, np.inf, np.nan])
def test_rejects_invalid_coefficient(coefficient):
    with pytest.raises(ValueError, match="lambda_lip"):
        spectral_lipschitz_penalty([], lambda_lip=coefficient)


@pytest.mark.parametrize(
    "weights",
    [
        np.array([1.0, 2.0]),
        np.empty((0, 2)),
        np.empty((2, 0)),
        np.array([[np.nan]]),
        np.array([[np.inf]]),
    ],
)
def test_rejects_invalid_weight_matrices(weights):
    with pytest.raises(ValueError):
        spectral_lipschitz_penalty([weights])


@pytest.mark.parametrize(
    "weights",
    [
        np.array([1.0, 2.0]),
        np.empty((0, 2)),
        np.empty((2, 0)),
        np.array([[np.nan]]),
        np.array([[np.inf]]),
    ],
)
def test_spectral_norm_estimate_rejects_invalid_matrices(weights):
    with pytest.raises(ValueError):
        spectral_norm_estimate(weights)


def test_rejects_penalties_that_overflow_float64():
    with pytest.raises(ValueError, match="too large"):
        spectral_lipschitz_penalty([np.array([[1e200]])])
