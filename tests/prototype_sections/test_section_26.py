from math import gamma

import numpy as np
import pytest
from scipy.special import loggamma

from clean_core.config import D_LORENTZ, POLE
from clean_core.geometry.lorentz import exp_map
from clean_core.prototypes.sections.section_26 import (
    Section26HolographicMemory,
    plancherel_density,
    poincare_plane_waves,
)


def _axis_boundary_directions():
    direction = np.zeros(D_LORENTZ - 1)
    direction[0] = 1.0
    return np.stack((direction, -direction))


def _point_at_distance(distance):
    tangent = np.zeros_like(POLE)
    tangent[1] = 1.0
    return exp_map(POLE.copy(), distance * tangent)


def test_plane_waves_follow_the_lorentzian_poincare_formula():
    distance = 0.7
    lambdas = np.array([0.4, 1.2])
    waves = poincare_plane_waves(
        _point_at_distance(distance), lambdas, _axis_boundary_directions()
    )
    rho = (D_LORENTZ - 2.0) / 2.0
    horospherical = np.array([np.exp(-distance), np.exp(distance)])

    np.testing.assert_allclose(
        waves,
        np.exp((-rho + 1j * lambdas[:, None]) * np.log(horospherical)[None, :]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        poincare_plane_waves(POLE, lambdas, _axis_boundary_directions()),
        np.ones((len(lambdas), 2), dtype=np.complex128),
    )


def test_plancherel_density_matches_gamma_ratio_and_zero_limit():
    lambdas = np.array([0.0, 0.5, 1.5])
    density = plancherel_density(lambdas, dimension=3)

    np.testing.assert_allclose(density[0], 0.0)
    np.testing.assert_allclose(density[1:], lambdas[1:] ** 2 / (2.0 * np.pi))

    dimension = D_LORENTZ
    rho = (dimension - 1.0) / 2.0
    lam = 0.8
    expected = np.exp(
        -np.log(2.0 * np.pi)
        - 2.0
        * (
            np.real(loggamma(rho))
            + np.real(loggamma(1j * lam))
            - np.real(loggamma(rho + 1j * lam))
        )
    )
    assert plancherel_density([lam])[0] == pytest.approx(expected)


def test_recall_uses_plancherel_measure_and_finite_sphere_area():
    lambdas = np.array([0.4])
    memory = Section26HolographicMemory(
        capacity=2,
        frequency_nodes=lambdas,
        frequency_weights=[0.25],
        boundary_directions=_axis_boundary_directions(),
    )
    memory.store(POLE, [2.0, -1.0])

    rho = (D_LORENTZ - 1.0) / 2.0
    sphere_area = 2.0 * np.pi**rho / gamma(rho)
    expected_scale = (
        0.25 * plancherel_density(lambdas)[0] * sphere_area
    )
    np.testing.assert_allclose(
        memory.recall(POLE), np.array([2.0, -1.0]) * expected_scale
    )


def test_recall_applies_radial_interference_phase_and_hbar():
    memory = Section26HolographicMemory(
        capacity=2,
        frequency_nodes=[0.8],
        frequency_weights=[1.0],
        boundary_directions=_axis_boundary_directions(),
    )
    memory.store(POLE, [1.0])
    query = _point_at_distance(0.6)

    recall_hbar_one = memory.recall(query, hbar_cog=1.0)
    recall_hbar_two = memory.recall(query, hbar_cog=2.0)
    assert np.isfinite(recall_hbar_one).all()
    assert np.isfinite(recall_hbar_two).all()
    assert not np.allclose(recall_hbar_one, recall_hbar_two)


def test_memory_is_bounded_fifo_and_empty_recall_is_finite():
    configuration = dict(
        capacity=2,
        frequency_nodes=[0.5],
        frequency_weights=[1.0],
        boundary_directions=_axis_boundary_directions(),
    )
    memory = Section26HolographicMemory(**configuration)
    expected = Section26HolographicMemory(**configuration)
    evicted = Section26HolographicMemory(**configuration)

    memory.store(POLE, [1.0, 2.0])
    assert len(memory) == 1
    memory.store(POLE, [2.0, 3.0])
    memory.store(POLE, [3.0, 4.0])
    expected.store(POLE, [2.0, 3.0])
    expected.store(POLE, [3.0, 4.0])
    evicted.store(POLE, [1.0, 2.0])
    evicted.store(POLE, [2.0, 3.0])

    assert len(memory) == memory.capacity
    np.testing.assert_allclose(memory.recall(POLE), expected.recall(POLE))
    assert not np.allclose(memory.recall(POLE), evicted.recall(POLE))

    empty = Section26HolographicMemory(capacity=1, frequency_nodes=[0.5])
    assert empty.recall(POLE).size == 0


def test_large_finite_angular_weights_normalize_without_overflow():
    directions = _axis_boundary_directions()
    ordinary = Section26HolographicMemory(
        capacity=1,
        frequency_nodes=[0.5],
        frequency_weights=[1.0],
        boundary_directions=directions,
        angular_weights=[1.0, 1.0],
    )
    large = Section26HolographicMemory(
        capacity=1,
        frequency_nodes=[0.5],
        frequency_weights=[1.0],
        boundary_directions=directions,
        angular_weights=[1e308, 1e308],
    )
    ordinary.store(POLE, [1.0])
    large.store(POLE, [1.0])

    np.testing.assert_allclose(large.recall(POLE), ordinary.recall(POLE))


@pytest.mark.parametrize(
    "call",
    [
        lambda: poincare_plane_waves(POLE, [], _axis_boundary_directions()),
        lambda: poincare_plane_waves(POLE, [-0.1], _axis_boundary_directions()),
        lambda: poincare_plane_waves(POLE, [1.0], np.zeros((1, D_LORENTZ - 1))),
        lambda: poincare_plane_waves(np.zeros_like(POLE), [1.0], _axis_boundary_directions()),
        lambda: plancherel_density([1.0], dimension=2),
        lambda: Section26HolographicMemory(capacity=0),
        lambda: Section26HolographicMemory(frequency_nodes=[0.0]),
        lambda: Section26HolographicMemory(frequency_nodes=[0.5, 0.4]),
    ],
)
def test_invalid_geometry_and_grid_inputs_are_rejected(call):
    with pytest.raises(ValueError):
        call()


def test_store_and_recall_validate_amplitudes_hbar_and_capacity_shape():
    memory = Section26HolographicMemory(
        capacity=1,
        frequency_nodes=[0.5],
        boundary_directions=_axis_boundary_directions(),
    )
    with pytest.raises(ValueError, match="finite vector"):
        memory.store(POLE, [np.nan])
    memory.store(POLE, [1.0])
    with pytest.raises(ValueError, match="shape"):
        memory.store(POLE, [1.0, 2.0])
    with pytest.raises(ValueError, match="hbar_cog"):
        memory.recall(POLE, hbar_cog=0.0)
