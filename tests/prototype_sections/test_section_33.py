import numpy as np
import pytest

from clean_core.prototypes.sections.section_33 import (
    optimal_geodesic_velocity,
    schrodinger_beltrami_graph_step,
    superpose_goal_wavefunctions,
    value_to_wavefunction,
    wavefunction_to_value,
)


def test_cole_hopf_transform_and_goal_wavefront_superposition():
    values = np.array([0.0, 0.5, 1.0])
    wavefunction = value_to_wavefunction(values, lambda_value=2.0, sigma=0.5)

    assert np.all(np.isfinite(wavefunction))
    assert np.all(wavefunction > 0.0)
    np.testing.assert_allclose(
        wavefunction_to_value(wavefunction, lambda_value=2.0, sigma=0.5),
        values,
    )
    np.testing.assert_allclose(
        superpose_goal_wavefunctions(
            [wavefunction, 2.0 * wavefunction],
            weights=[1.0, 0.5],
        ),
        2.0 * wavefunction,
    )


def test_weighted_graph_diffusion_matches_section_33_equation_and_stays_positive():
    result = schrodinger_beltrami_graph_step(
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


def test_weighted_graph_step_enforces_explicit_positivity_bound():
    common = {
        "edge_weights": [[0.0, 1.0], [1.0, 0.0]],
        "pragmatic_potential": [0.0, 0.0],
        "lambda_value": 2.0,
        "sigma": 1.0,
    }
    with pytest.raises(ValueError, match="positive"):
        schrodinger_beltrami_graph_step([0.0, 1.0], dt=0.1, **common)
    with pytest.raises(ValueError, match="positivity bound"):
        schrodinger_beltrami_graph_step([1.0, 2.0], dt=1.1, **common)
    with pytest.raises(ValueError, match="symmetric"):
        schrodinger_beltrami_graph_step(
            [1.0, 2.0],
            edge_weights=[[0.0, 1.0], [0.0, 0.0]],
            pragmatic_potential=[0.0, 0.0],
            dt=0.1,
            lambda_value=1.0,
            sigma=1.0,
        )


def test_section_33_transform_and_superposition_validate_inputs():
    with pytest.raises(ValueError, match="finite"):
        value_to_wavefunction([0.0, np.nan], 1.0, 1.0)
    with pytest.raises(ValueError, match="positive"):
        wavefunction_to_value([1.0, 0.0], 1.0, 1.0)
    with pytest.raises(ValueError, match="weights"):
        superpose_goal_wavefunctions([[1.0, 1.0]], [-1.0])
    with pytest.raises(ValueError, match="finite and positive"):
        value_to_wavefunction([1.0], 1e308, 1e308)


def test_optimal_velocity_uses_frame_projected_log_wavefunction_gradient():
    velocity = optimal_geodesic_velocity(
        log_wavefunction_gradient=[3.0, 2.0],
        frame=[[1.0], [4.0]],
        lorentz_metric=[-1.0, 1.0],
        lambda_value=2.0,
        sigma=0.5,
    )

    np.testing.assert_allclose(velocity, [2.5])


def test_optimal_velocity_rejects_invalid_dimensions_and_nonfinite_results():
    with pytest.raises(ValueError, match="one row"):
        optimal_geodesic_velocity([1.0, 2.0], [[1.0]], [1.0, 1.0], 1.0, 1.0)
    with pytest.raises(FloatingPointError, match="finite"):
        optimal_geodesic_velocity(
            [1.0],
            [[1e308]],
            [1.0],
            lambda_value=1e308,
            sigma=1.0,
        )
