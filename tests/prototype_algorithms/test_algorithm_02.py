import copy

import numpy as np
import pytest

from clean_core.config import D_SPATIAL, ETA_L, POLE
from clean_core.core.neuron import NeuronMicrostate
from clean_core.geometry.lorentz import exp_map, log_map
from clean_core.prototypes.algorithms.algorithm_02 import run_algorithm_02


def _point(offset=0.0):
    tangent = np.zeros_like(POLE)
    tangent[1:] = np.linspace(-0.02, 0.02, D_SPATIAL) + offset
    return exp_map(POLE.copy(), tangent)


def _neuron():
    neuron = NeuronMicrostate()
    neuron.W1.fill(0.01)
    neuron.W2.fill(0.01)
    neuron.w.fill(0.1)
    return neuron


def _assert_future_unit_hyperboloid(point):
    assert point.shape == POLE.shape
    assert np.all(np.isfinite(point))
    assert point[0] > 0.0
    assert np.sum(point * ETA_L * point) == pytest.approx(-1.0, abs=1e-9)


def test_algorithm_02_is_deterministic_bounded_and_preserves_manifold():
    first_neuron = _neuron()
    second_neuron = copy.deepcopy(first_neuron)
    weights_before = (first_neuron.W1.copy(), first_neuron.W2.copy())
    initial_mu = first_neuron.mu.copy()
    context = np.linspace(-0.5, 0.5, D_SPATIAL)
    barycenter = _point(0.01)

    first = run_algorithm_02(first_neuron, context, barycenter)
    second = run_algorithm_02(second_neuron, context, barycenter)

    np.testing.assert_array_equal(first.mu, second.mu)
    np.testing.assert_array_equal(first.velocity, second.velocity)
    np.testing.assert_array_equal(first.precision, second.precision)
    assert np.linalg.norm(first.velocity) <= 5.0
    assert np.linalg.norm(log_map(initial_mu, first.mu)) <= 0.01 * 5.0 + 1e-10
    assert np.linalg.norm(log_map(barycenter, first.prediction)) <= 3.0 + 1e-10
    assert first.prediction.shape == POLE.shape
    assert first.local_error.shape == (D_SPATIAL,)
    assert first.precision.shape == (D_SPATIAL, D_SPATIAL)
    assert np.all(np.isfinite(first.local_error))
    assert np.all(np.isfinite(first.precision))
    assert np.allclose(first.precision, first.precision.T, atol=1e-10)
    assert 1e-3 <= first.adaptive_precision <= 1e3
    _assert_future_unit_hyperboloid(first.mu)
    _assert_future_unit_hyperboloid(first.prediction)
    np.testing.assert_array_equal(first_neuron.W1, weights_before[0])
    np.testing.assert_array_equal(first_neuron.W2, weights_before[1])
    np.testing.assert_allclose(
        first_neuron.V_basis.T @ first_neuron.V_basis,
        np.eye(first_neuron.V_basis.shape[1]),
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("context", "barycenter", "message"),
    [
        (np.zeros(D_SPATIAL - 1), POLE, "context must have shape"),
        (np.full(D_SPATIAL, np.nan), POLE, "context must contain only finite"),
        (np.zeros(D_SPATIAL), np.zeros_like(POLE), "barycenter_mu"),
    ],
)
def test_algorithm_02_rejects_invalid_inputs_without_mutating_state(
    context, barycenter, message
):
    neuron = _neuron()
    original_mu = neuron.mu.copy()
    original_velocity = neuron.w.copy()
    original_basis = neuron.V_basis.copy()

    with pytest.raises(ValueError, match=message):
        run_algorithm_02(neuron, context, barycenter)

    np.testing.assert_array_equal(neuron.mu, original_mu)
    np.testing.assert_array_equal(neuron.w, original_velocity)
    np.testing.assert_array_equal(neuron.V_basis, original_basis)


@pytest.mark.parametrize("dt", [0.0, -0.1, 0.11, float("inf")])
def test_algorithm_02_rejects_nonpositive_or_unbounded_step_size(dt):
    with pytest.raises(ValueError, match="dt"):
        run_algorithm_02(_neuron(), np.zeros(D_SPATIAL), POLE, dt=dt)


def test_algorithm_02_rejects_malformed_model_weights():
    neuron = _neuron()
    neuron.W2 = np.zeros((D_SPATIAL, D_SPATIAL))

    with pytest.raises(ValueError, match="neuron.W2"):
        run_algorithm_02(neuron, np.zeros(D_SPATIAL), POLE)
