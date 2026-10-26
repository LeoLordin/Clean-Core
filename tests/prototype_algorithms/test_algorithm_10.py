import numpy as np
import pytest

from clean_core.config import D_SPATIAL, POLE
from clean_core.core.neuron import NeuronMicrostate
from clean_core.geometry.lorentz import exp_map, log_map
from clean_core.prototypes.algorithms.algorithm_10 import (
    RELAXATION_STEPS,
    run_algorithm_10,
)
from clean_core.prototypes.sections.section_26 import Section26HolographicMemory


def _query_point():
    tangent = np.zeros_like(POLE)
    tangent[1:] = np.linspace(-0.02, 0.02, D_SPATIAL)
    return exp_map(POLE.copy(), tangent)


def test_algorithm_10_relaxes_without_training_and_returns_finite_bounded_recall():
    memory = Section26HolographicMemory(capacity=2)
    query = _query_point()
    memory.store(query, np.linspace(0.1, 1.5, D_SPATIAL))
    memory.store(POLE, np.ones(D_SPATIAL))
    memory.store(query, np.full(D_SPATIAL, 2.0))
    assert len(memory) == memory.capacity

    neuron = NeuronMicrostate()
    weights_before = (neuron.W1.copy(), neuron.W2.copy())
    initial_distance = np.linalg.norm(log_map(neuron.mu, query))

    result = run_algorithm_10([neuron], query, POLE, memory)

    final_distance = np.linalg.norm(log_map(result.stationary_state, query))
    assert RELAXATION_STEPS == 4
    assert result.relaxed_states.shape == (1, len(POLE))
    assert len(result.errors) == 1
    assert final_distance < initial_distance
    assert np.all(np.isfinite(result.associative_recall))
    assert np.all(np.isfinite(result.target_action))
    assert np.iscomplexobj(result.associative_recall)
    np.testing.assert_array_equal(neuron.W1, weights_before[0])
    np.testing.assert_array_equal(neuron.W2, weights_before[1])


def test_algorithm_10_handles_empty_memory_with_finite_feature_proxy():
    result = run_algorithm_10(
        [NeuronMicrostate()],
        _query_point(),
        POLE,
        Section26HolographicMemory(capacity=1),
    )

    assert result.associative_recall.shape == (0,)
    assert result.target_action.shape == (D_SPATIAL,)
    assert np.all(np.isfinite(result.target_action))


def test_algorithm_10_accepts_a_fixed_inference_decoder():
    memory = Section26HolographicMemory(capacity=1)
    memory.store(POLE, np.ones(D_SPATIAL))
    decoder_inputs = []

    def decoder(features):
        decoder_inputs.append(features.copy())
        return features[:3]

    result = run_algorithm_10(
        [NeuronMicrostate()],
        _query_point(),
        POLE,
        memory,
        motor_decoder=decoder,
    )

    assert result.target_action.shape == (3,)
    np.testing.assert_array_equal(result.target_action, decoder_inputs[0][:3])


@pytest.mark.parametrize(
    ("query", "barycenter", "memory", "message"),
    [
        (np.zeros_like(POLE), POLE, Section26HolographicMemory(), "query_mu"),
        (POLE, np.zeros_like(POLE), Section26HolographicMemory(), "barycenter_mu"),
        (POLE, POLE, object(), "Section26HolographicMemory"),
    ],
)
def test_algorithm_10_rejects_invalid_inputs(query, barycenter, memory, message):
    with pytest.raises(ValueError, match=message):
        run_algorithm_10([NeuronMicrostate()], query, barycenter, memory)
