import numpy as np
import pytest

from clean_core.config import D_LORENTZ, ETA_L, POLE
from clean_core.geometry.lorentz import get_boost_frame
from clean_core.prototypes.sections.section_04 import (
    linearized_hjb_graph_step,
    optimal_geodesic_velocity,
    superpose_goal_wavefunctions,
    value_to_wavefunction,
    wavefunction_to_value,
)


def test_cole_hopf_transform_round_trip_and_superposition():
    values = np.array([0.0, 0.5, 1.0])
    wavefunction = value_to_wavefunction(values, lambda_value=2.0, sigma=0.5)

    assert np.all(np.isfinite(wavefunction))
    assert np.all(wavefunction > 0.0)
    np.testing.assert_allclose(
        wavefunction_to_value(wavefunction, lambda_value=2.0, sigma=0.5), values
    )
    np.testing.assert_allclose(
        superpose_goal_wavefunctions([wavefunction, 2.0 * wavefunction], [1.0, 0.5]),
        2.0 * wavefunction,
    )


def test_cole_hopf_and_superposition_reject_invalid_inputs():
    with pytest.raises(ValueError, match="finite"):
        value_to_wavefunction([0.0, np.nan], 1.0, 1.0)
    with pytest.raises(ValueError, match="positive"):
        wavefunction_to_value([1.0, 0.0], 1.0, 1.0)
    with pytest.raises(ValueError, match="finite positive"):
        superpose_goal_wavefunctions([[1.0, np.inf]])
    with pytest.raises(ValueError, match="weights"):
        superpose_goal_wavefunctions([[1.0, 1.0]], [-1.0])
    with pytest.raises(ValueError, match="positive"):
        value_to_wavefunction([1.0], 1.0, 0.0)


def test_graph_diffusion_step_matches_linearized_equation_and_preserves_positivity():
    result = linearized_hjb_graph_step(
        wavefunction=[1.0, 2.0],
        edge_weights=[[0.0, 1.0], [1.0, 0.0]],
        pragmatic_potential=[0.2, 0.0],
        dt=0.1,
        lambda_value=2.0,
        sigma=1.0,
    )

    np.testing.assert_allclose(result, [1.09, 1.9])
    assert np.all(np.isfinite(result))
    assert np.all(result > 0.0)


def test_graph_diffusion_rejects_nonpositive_wavefunction_and_unstable_step():
    common = {
        "edge_weights": [[0.0, 1.0], [1.0, 0.0]],
        "pragmatic_potential": [0.0, 0.0],
        "lambda_value": 2.0,
        "sigma": 1.0,
    }
    with pytest.raises(ValueError, match="strictly positive"):
        linearized_hjb_graph_step([0.0, 1.0], dt=0.1, **common)
    with pytest.raises(ValueError, match="positivity bound"):
        linearized_hjb_graph_step([1.0, 2.0], dt=1.1, **common)
    with pytest.raises(ValueError, match="finite"):
        linearized_hjb_graph_step(
            [1.0, np.nan], dt=0.1, **common
        )


def test_optimal_velocity_uses_local_lorentz_frame_projection():
    gradient = np.zeros(D_LORENTZ)
    gradient[1] = 2.0
    velocity = optimal_geodesic_velocity(
        gradient,
        get_boost_frame(POLE),
        ETA_L,
        lambda_value=2.0,
        sigma=0.5,
    )

    expected = np.zeros(D_LORENTZ - 1)
    expected[0] = 1.0
    np.testing.assert_allclose(velocity, expected)


def test_optimal_velocity_rejects_nonfinite_inputs_and_overflowing_scale():
    with pytest.raises(ValueError, match="finite"):
        optimal_geodesic_velocity(
            [np.nan], [[1.0]], [1.0], lambda_value=1.0, sigma=1.0
        )
    with pytest.raises(ValueError, match="finite and positive"):
        optimal_geodesic_velocity(
            [1.0], [[1.0]], [1.0], lambda_value=1e308, sigma=1e308
        )
