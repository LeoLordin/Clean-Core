import numpy as np
import pytest

from clean_core.cognition.stability import lyapunov_candidate
from clean_core.config import D_SPATIAL, POLE
from clean_core.geometry.lorentz import exp_map


def test_lyapunov_candidate_is_zero_at_equilibrium():
    assert lyapunov_candidate(np.array([POLE]), POLE, np.zeros(2), np.zeros(2)) == 0.0


def test_lyapunov_candidate_sums_geodesic_and_action_errors():
    tangent = np.zeros(D_SPATIAL + 1)
    tangent[1] = 0.4
    state = exp_map(POLE.copy(), tangent)
    expected = 0.5 * 0.4**2 + 0.5 * (3.0**2 + 4.0**2)

    assert lyapunov_candidate(np.array([state]), POLE, [3.0, 4.0], [0.0, 0.0]) == pytest.approx(expected)


def test_lyapunov_candidate_rejects_mismatched_action_pairs():
    with pytest.raises(ValueError):
        lyapunov_candidate(np.array([POLE]), POLE, actions=[1.0])