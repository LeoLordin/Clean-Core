import numpy as np
import pytest

from clean_core.config import D_SPATIAL, ETA_L
from clean_core.core.neuron import NeuronMicrostate
from clean_core.prototypes.catalog import (
    load_algorithm,
    load_all_prototypes,
    load_section,
)
from clean_core.prototypes.runtime import sensory_to_inference_step


def test_catalog_imports_all_sections_and_algorithms_in_number_order():
    sections, algorithms = load_all_prototypes()

    assert len(sections) == 38
    assert len(algorithms) == 13
    assert sections[0].__name__.endswith("section_01")
    assert sections[-1].__name__.endswith("section_38")
    assert algorithms[0].__name__.endswith("algorithm_01")
    assert algorithms[-1].__name__.endswith("algorithm_13")


@pytest.mark.parametrize(
    ("loader", "number"),
    [(load_section, 0), (load_section, 39), (load_algorithm, 0), (load_algorithm, 14)],
)
def test_catalog_rejects_out_of_range_prototype_numbers(loader, number):
    with pytest.raises(ValueError):
        loader(number)


@pytest.mark.parametrize(
    ("loader", "number"),
    [(load_section, True), (load_algorithm, 1.5)],
)
def test_catalog_rejects_non_integer_prototype_numbers(loader, number):
    with pytest.raises(ValueError):
        loader(number)


def test_sensory_to_inference_cycle_connects_grounding_and_manifold_update():
    neuron = NeuronMicrostate()
    neuron.W1.fill(0.01)
    neuron.W2.fill(0.01)
    image = np.linspace(0.0, 1.0, 24).reshape(2, 3, 4)
    audio = np.linspace(-1.0, 1.0, 16)
    text = np.array([1, 7, 18, 24], dtype=np.int64)

    result = sensory_to_inference_step(neuron, image, audio, text)

    assert result.grounding.posterior_mean.shape == (D_SPATIAL,)
    np.testing.assert_allclose(np.sum(result.grounding.posterior_mean), 1.0)
    np.testing.assert_allclose(
        np.sum(result.inference.mu * ETA_L * result.inference.mu), -1.0, atol=1e-9
    )
    np.testing.assert_allclose(neuron.mu, result.inference.mu)
    assert np.all(np.isfinite(result.inference.local_error))


def test_failed_grounding_does_not_mutate_neuron():
    neuron = NeuronMicrostate()
    original_mu = neuron.mu.copy()
    original_weights = (neuron.W1.copy(), neuron.W2.copy())

    with pytest.raises(ValueError, match="text"):
        sensory_to_inference_step(
            neuron,
            np.zeros((1, 2, 2)),
            np.zeros(3),
            np.array([1.5]),
        )

    np.testing.assert_array_equal(neuron.mu, original_mu)
    np.testing.assert_array_equal(neuron.W1, original_weights[0])
    np.testing.assert_array_equal(neuron.W2, original_weights[1])
