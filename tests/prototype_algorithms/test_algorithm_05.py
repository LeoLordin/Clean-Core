import numpy as np
import pytest

from clean_core.prototypes.algorithms.algorithm_05 import synthesize_motor_action
from clean_core.prototypes.sections.section_06 import Section06PolyhedralDecider
from clean_core.prototypes.sections.section_27 import ConformalTheoryOfMindPlanner


def _components():
    decider = Section06PolyhedralDecider(2, 1, r_max=1.0, discount=0.0)
    decider.add_candidate("0", [[1.0], [0.0]])
    decider.add_candidate("1", [[1.0], [0.0]])
    decider.advance(2)
    planner = ConformalTheoryOfMindPlanner()

    return decider, planner


def _inputs():
    decider, planner = _components()
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    edge_weights = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    )
    return {
        "goal_wavefunctions": np.array(
            [[1.0, 2.0, 4.0], [2.0, 1.0, 1.0]]
        ),
        "edge_weights": edge_weights,
        "pragmatic_potential": np.zeros(3),
        "node_coordinates": coordinates,
        "top_layer_coordinate": coordinates[1],
        "frame": np.eye(2),
        "lorentz_metric": np.ones(2),
        "lambda_value": 1.0,
        "sigma": 1.0,
        "graph_dt": 0.1,
        "polyhedral_decider": decider,
        "optimal_action": 0,
        "social_planner": planner,
        "social_curiosity": [0.0, 0.0],
        "other_action_beliefs": [[1.0, 0.0]],
        "social_utility_tensor": np.array([[[0.0, 0.0]], [[1.0, 0.0]]]),
        "motor_weights": np.eye(2),
        "polyhedral_weights": np.ones((2, 2)),
        "social_weights": np.array([[0.5, -0.5], [-0.5, 0.5]]),
        "motor_bias": np.zeros(2),
        "current_action": np.array([-1.0, 1.0]),
    }


def test_algorithm_05_composes_finite_graph_polyhedral_and_social_signals():
    result = synthesize_motor_action(**_inputs())

    np.testing.assert_allclose(result.goal_wavefunction, [3.0, 3.0, 5.0])
    assert np.all(np.isfinite(result.propagated_wavefunction))
    assert np.all(result.propagated_wavefunction > 0.0)
    assert result.graph_log_gradient.shape == (2,)
    assert result.geodesic_signal.shape == (2,)
    assert result.discrete_action == 0
    assert result.polyhedral_safe_radius > 0.0
    assert result.polyhedral_pressure.shape == (2,)
    np.testing.assert_allclose(result.social_action, [0.0, 1.0])
    assert result.target_action.shape == (2,)
    assert np.all((-1.0 <= result.target_action) & (result.target_action <= 1.0))
    assert np.all((-1.0 <= result.updated_action) & (result.updated_action <= 1.0))
    assert np.any(result.updated_action != [-1.0, 1.0])


def test_algorithm_05_can_weight_goal_superposition_and_set_motor_step():
    inputs = _inputs()
    inputs["goal_weights"] = [1.0, 0.0]
    inputs["motor_dt"] = 0.5
    result = synthesize_motor_action(**inputs)

    np.testing.assert_allclose(result.goal_wavefunction, [1.0, 2.0, 4.0])
    np.testing.assert_allclose(
        result.updated_action,
        np.clip(
            np.array([-1.0, 1.0]) + 0.5 * result.motor_velocity, -1.0, 1.0
        ),
    )


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("goal_wavefunctions", [[1.0, 2.0]], "shape"),
        ("top_layer_coordinate", [0.0], "coordinate dimension"),
        ("edge_weights", [[0.0, 1.0], [1.0, 0.0]], "square"),
        ("current_action", [0.0], "current_action"),
        ("motor_dt", 0.0, "motor_dt"),
        ("graph_dt", np.nan, "graph_dt"),
        ("motor_learning_rate", np.inf, "motor_learning_rate"),
    ],
)
def test_algorithm_05_rejects_invalid_shapes_and_parameters(key, value, message):
    inputs = _inputs()
    inputs[key] = value

    with pytest.raises(ValueError, match=message):
        synthesize_motor_action(**inputs)


def test_algorithm_05_requires_the_requested_action_to_match_finite_decider():
    inputs = _inputs()
    inputs["optimal_action"] = 1

    with pytest.raises(ValueError, match="match the Section 6 decider"):
        synthesize_motor_action(**inputs)


def test_algorithm_05_rejects_unbounded_current_motor_action():
    inputs = _inputs()
    inputs["current_action"] = [1.1, 0.0]

    with pytest.raises(ValueError, match="current_action"):
        synthesize_motor_action(**inputs)
