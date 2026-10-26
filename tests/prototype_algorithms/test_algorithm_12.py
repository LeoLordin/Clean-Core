from math import gamma

import numpy as np
import pytest

from clean_core.config import D_LORENTZ, POLE
from clean_core.geometry.lorentz import exp_map
from clean_core.prototypes.algorithms.algorithm_12 import (
    Algorithm12HolographicMemory,
    solve_algorithm_12,
)
from clean_core.prototypes.sections.section_26 import plancherel_density


def _axis_boundary_directions():
    direction = np.zeros(D_LORENTZ - 1)
    direction[0] = 1.0
    return np.stack((direction, -direction))


def _point_at_distance(distance):
    tangent = np.zeros_like(POLE)
    tangent[1] = 1.0
    return exp_map(POLE.copy(), distance * tangent)


def _memory(**kwargs):
    return Algorithm12HolographicMemory(
        boundary_directions=_axis_boundary_directions(), **kwargs
    )


def test_algorithm_12_recall_uses_plancherel_weights_and_finite_sphere_area():
    frequencies = np.array([0.4, 0.9])
    quadrature = np.array([0.2, 0.3])
    memory = _memory(frequency_nodes=frequencies, frequency_weights=quadrature)
    memory.store(POLE, [2.0, -1.0])

    rho = (D_LORENTZ - 1.0) / 2.0
    expected_scale = (
        np.sum(quadrature * plancherel_density(frequencies))
        * 2.0
        * np.pi**rho
        / gamma(rho)
    )
    result = solve_algorithm_12(memory, POLE)

    np.testing.assert_allclose(result, np.array([2.0, -1.0]) * expected_scale)
    assert np.iscomplexobj(result)


def test_algorithm_12_retrieval_applies_radial_phase_and_hbar():
    memory = _memory(frequency_nodes=[0.8], frequency_weights=[1.0])
    memory.store(POLE, [1.0])
    query = _point_at_distance(0.6)

    hbar_one = solve_algorithm_12(memory, query, hbar_cog=1.0)
    hbar_two = solve_algorithm_12(memory, query, hbar_cog=2.0)

    assert np.isfinite(hbar_one).all()
    assert not np.allclose(hbar_one, hbar_two)


def test_algorithm_12_memory_is_bounded_and_returns_zero_when_empty():
    memory = _memory(capacity=2, frequency_nodes=[0.5])
    assert solve_algorithm_12(memory, POLE).size == 0

    memory.store(POLE, [1.0])
    memory.store(POLE, [2.0])
    first_two = solve_algorithm_12(memory, POLE)
    memory.store(POLE, [3.0])

    assert len(memory) == memory.capacity == 2
    expected = _memory(capacity=2, frequency_nodes=[0.5])
    expected.store(POLE, [2.0])
    expected.store(POLE, [3.0])
    np.testing.assert_allclose(
        solve_algorithm_12(memory, POLE), solve_algorithm_12(expected, POLE)
    )
    assert not np.allclose(first_two, solve_algorithm_12(memory, POLE))


@pytest.mark.parametrize(
    ("memory", "query", "hbar", "message"),
    [
        (object(), POLE, 1.0, "memory"),
        (_memory(frequency_nodes=[0.5]), np.zeros(2), 1.0, "query_mu"),
        (_memory(frequency_nodes=[0.5]), POLE, 0.0, "hbar_cog"),
    ],
)
def test_algorithm_12_rejects_invalid_retrieval_inputs(memory, query, hbar, message):
    with pytest.raises(ValueError, match=message):
        solve_algorithm_12(memory, query, hbar_cog=hbar)
