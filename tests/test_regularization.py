import numpy as np
import pytest

from clean_core.estimation.regularization import spectral_lipschitz_penalty


def test_spectral_lipschitz_penalty_uses_largest_singular_value():
    weights = np.diag([2.0, 0.5])

    assert spectral_lipschitz_penalty([weights], limit=1.0) == pytest.approx(1.0)
    assert spectral_lipschitz_penalty([weights], limit=2.0) == pytest.approx(0.0)


def test_spectral_lipschitz_penalty_sums_independent_layers():
    first = np.diag([2.0, 1.0])
    second = np.diag([1.5, 0.5])

    assert spectral_lipschitz_penalty([first, second]) == pytest.approx(1.25)