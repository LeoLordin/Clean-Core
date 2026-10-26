import numpy as np
import pytest

from clean_core.agent import CleanCoreProductionAgent
from clean_core.config import D_SPATIAL
from clean_core.geometry.lorentz import minkowski_norm_sq
from clean_core.interfaces.dirichlet_sensory import DirichletSensoryLayer


def test_agent_train_step_keeps_network_states_on_hyperboloid():
    agent = CleanCoreProductionAgent(num_neurons=3)

    result = agent.train_step({"position": 0.25, "velocity": -0.1})

    assert set(result) == {"selected_action", "score_margin", "motor_action", "state_error"}
    for neuron in agent.neurons:
        assert abs(minkowski_norm_sq(neuron.mu) + 1.0) < 1e-12
    assert abs(minkowski_norm_sq(agent.barycenter_L1) + 1.0) < 1e-12
    assert np.all(np.isfinite(result["motor_action"]))


def test_agent_rejects_empty_or_non_finite_sensory_input():
    agent = CleanCoreProductionAgent(num_neurons=1)

    with pytest.raises(ValueError):
        agent.train_step({})
    with pytest.raises(ValueError):
        agent.train_step({"invalid": np.nan})


@pytest.mark.parametrize("input_size", [1, D_SPATIAL, D_SPATIAL + 5])
def test_sensory_layer_always_returns_valid_hyperboloid_point(input_size):
    layer = DirichletSensoryLayer()
    point = layer.ground_sensory_vector(np.arange(input_size, dtype=np.float64))

    assert point.shape == (D_SPATIAL + 1,)
    assert abs(minkowski_norm_sq(point) + 1.0) < 1e-12


def test_sensory_dirichlet_prior_affects_grounded_point():
    observation = np.linspace(-1.0, 1.0, D_SPATIAL)
    uniform = DirichletSensoryLayer(np.ones(D_SPATIAL))
    biased_prior = DirichletSensoryLayer(np.arange(1, D_SPATIAL + 1))

    uniform_point = uniform.ground_sensory_vector(observation)
    biased_point = biased_prior.ground_sensory_vector(observation)

    assert not np.allclose(uniform_point, biased_point)