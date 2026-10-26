import numpy as np
import pytest

from clean_core.config import D_SPATIAL, ETA_L, POLE
from clean_core.core.neuron import NeuronMicrostate
from clean_core.geometry.lorentz import exp_map, get_boost_frame
from clean_core.prototypes.sections.section_30 import (
    section_30_fast_geodesic_inference_relaxation,
)


def _neuron_at(tangent):
    neuron = NeuronMicrostate()
    neuron.mu = exp_map(POLE.copy(), tangent)
    return neuron


def _weights_and_model_state(neuron):
    return {
        key: value.copy()
        for key, value in vars(neuron).items()
        if isinstance(value, np.ndarray) and key != "mu"
    }


def test_relaxation_matches_precision_weighted_velocity_and_speed_cap():
    tangent = np.zeros_like(POLE)
    tangent[1] = 0.2
    neuron = _neuron_at(tangent)
    neuron.Pi_precision = np.diag(np.linspace(0.5, 1.5, D_SPATIAL))
    velocity = np.linspace(2.0, 5.0, D_SPATIAL)
    dt, epsilon_0, pi_value, v_max = 0.02, 0.01, 1.4, 0.3
    original_mu = neuron.mu.copy()
    weights_before = _weights_and_model_state(neuron)
    matrix = (
        np.eye(D_SPATIAL)
        + (dt / epsilon_0) * pi_value * neuron.Pi_precision
    )
    expected_velocity = np.linalg.solve(matrix, (dt / epsilon_0) * velocity)
    expected_velocity *= v_max / np.linalg.norm(expected_velocity)
    expected_mu = exp_map(
        original_mu, get_boost_frame(original_mu) @ expected_velocity
    )

    states, relaxed_velocities = section_30_fast_geodesic_inference_relaxation(
        [neuron],
        velocity[None, None, :],
        pi_value,
        dt=dt,
        epsilon_0=epsilon_0,
        v_max=v_max,
    )

    np.testing.assert_allclose(relaxed_velocities[0, 0], expected_velocity)
    np.testing.assert_allclose(states[1, 0], expected_mu)
    np.testing.assert_allclose(np.linalg.norm(relaxed_velocities[0, 0]), v_max)
    assert states.shape == (2, 1, POLE.size)
    assert all(
        np.array_equal(getattr(neuron, key), value)
        for key, value in weights_before.items()
    )
    np.testing.assert_array_equal(neuron.mu, expected_mu)


def test_iterative_relaxation_reframes_each_step_and_stays_on_lorentz_manifold():
    tangent = np.zeros_like(POLE)
    tangent[1] = 0.15
    neurons = [_neuron_at(tangent), NeuronMicrostate()]
    velocities = np.zeros((3, 2, D_SPATIAL))
    velocities[0, 0, 0] = 0.4
    velocities[1, 0, 1] = -0.3
    velocities[2, 1, 2] = 0.2
    snapshots = [_weights_and_model_state(neuron) for neuron in neurons]
    expected = np.stack([neuron.mu.copy() for neuron in neurons])
    expected_trajectory = [expected.copy()]
    scale = 0.01 / 0.001

    for step in range(velocities.shape[0]):
        updated = []
        for index, neuron in enumerate(neurons):
            local_velocity = np.linalg.solve(
                np.eye(D_SPATIAL)
                + scale * neuron.Pi_precision,
                scale * velocities[step, index],
            )
            local_velocity *= min(1.0, 0.25 / np.linalg.norm(local_velocity)) if np.linalg.norm(local_velocity) else 1.0
            updated.append(
                exp_map(expected[index], get_boost_frame(expected[index]) @ local_velocity)
            )
        expected = np.stack(updated)
        expected_trajectory.append(expected.copy())

    states, velocities_out = section_30_fast_geodesic_inference_relaxation(
        neurons, velocities, 1.0, v_max=0.25
    )

    np.testing.assert_allclose(states, np.stack(expected_trajectory), atol=1e-12)
    assert velocities_out.shape == velocities.shape
    assert np.all(np.linalg.norm(velocities_out, axis=-1) <= 0.25 + 1e-12)
    for points in states:
        lorentz_norms = np.sum(points * ETA_L * points, axis=1)
        np.testing.assert_allclose(lorentz_norms, -1.0, atol=1e-12)
        assert np.all(points[:, 0] > 0.0)
    for neuron, snapshot in zip(neurons, snapshots):
        for key, value in snapshot.items():
            np.testing.assert_array_equal(getattr(neuron, key), value)


@pytest.mark.parametrize(
    "velocities, pi_values, kwargs",
    [
        (np.zeros((1, 1, D_SPATIAL)), 0.0, {}),
        (np.zeros((1, 1, D_SPATIAL)), 1.0, {"dt": 0.0}),
        (np.zeros((1, 1, D_SPATIAL)), 1.0, {"epsilon_0": -1.0}),
        (np.zeros((1, 1, D_SPATIAL)), 1.0, {"v_max": np.inf}),
        (np.zeros((1, 1, D_SPATIAL - 1)), 1.0, {}),
        (np.full((1, 1, D_SPATIAL), np.nan), 1.0, {}),
    ],
)
def test_rejects_invalid_relaxation_inputs(velocities, pi_values, kwargs):
    with pytest.raises(ValueError):
        section_30_fast_geodesic_inference_relaxation(
            [NeuronMicrostate()], velocities, pi_values, **kwargs
        )


def test_rejects_invalid_neuron_precision_and_lorentz_state():
    neuron = NeuronMicrostate()
    neuron.Pi_precision[0, 1] = 1.0
    with pytest.raises(ValueError, match="symmetric"):
        section_30_fast_geodesic_inference_relaxation(
            [neuron], np.zeros((1, 1, D_SPATIAL)), 1.0
        )

    neuron = NeuronMicrostate()
    neuron.mu[0] = -1.0
    with pytest.raises(ValueError, match="hyperboloid"):
        section_30_fast_geodesic_inference_relaxation(
            [neuron], np.zeros((1, 1, D_SPATIAL)), 1.0
        )
