import numpy as np
import pytest

from clean_core.prototypes.sections.section_06 import (
    Section06PolyhedralDecider,
    decision_hyperplane,
)


def test_pairwise_decision_hyperplane_matches_simplex_boundary():
    q_values = np.array([[2.0, 0.0], [0.0, 2.0]])

    coefficients = decision_hyperplane(q_values, 0, 1)

    np.testing.assert_allclose(coefficients, [2.0, -2.0])
    assert np.dot([0.5, 0.5], coefficients) == 0.0
    assert np.dot([0.75, 0.25], coefficients) > 0.0
    np.testing.assert_allclose(
        decision_hyperplane(q_values, 1, 0), -coefficients
    )
    np.testing.assert_allclose(
        Section06PolyhedralDecider.decision_hyperplane(q_values, 0, 1),
        coefficients,
    )


def test_decision_regions_follow_caller_supplied_observation_distribution():
    decider = Section06PolyhedralDecider(2, 2, r_max=2.0, discount=0.5)
    q_values = [[2.0, 0.0], [0.0, 2.0]]
    decider.add_candidate("0", q_values)
    decider.add_candidate("1", q_values)

    decider.set_observation_probabilities([0.75, 0.25])
    first_region = decider.advance(2)
    decider.set_observation_probabilities([0.25, 0.75])
    second_region = decider.decision()

    assert first_region["action"] == 0
    assert first_region["bounds_separate"]
    assert second_region["action"] == 1
    assert second_region["bounds_separate"]


def test_finite_prefix_models_produce_bounds_and_safe_pressure():
    decider = Section06PolyhedralDecider(2, 1, r_max=1.0, discount=0.5)
    decider.add_candidate("0", [[1.0], [0.0]])
    decider.add_candidate("10", [[0.0], [2.0]])
    decider.add_candidate("11", [[0.0], [2.0]])

    partial = decider.advance()
    assert partial["unresolved_mass"] == 0.5
    assert not partial["bounds_separate"]
    assert partial["safe_radius"] == 0.0

    result = decider.advance(2)
    assert result["unresolved_mass"] == 0.0
    assert result["bounds_separate"]
    assert result["action"] == 1
    assert result["q_min"][1] > result["q_max"][0]
    assert result["safe_radius"] > 0.0
    assert np.linalg.norm(result["motor_pressure"]) > 0.0


@pytest.mark.parametrize(
    ("q_values", "action_a", "action_b", "message"),
    [
        ([[1.0]], 0, 1, "at least 2 actions"),
        ([[1.0], [np.inf]], 0, 1, "finite"),
        ([[1.0], [0.0]], -1, 1, "out of range"),
        ([[1.0], [0.0]], 0, 0, "distinct"),
        ([[1.0], [0.0]], 0.5, 1, "integers"),
    ],
)
def test_decision_hyperplane_rejects_invalid_inputs(
    q_values, action_a, action_b, message
):
    with pytest.raises(ValueError, match=message):
        decision_hyperplane(q_values, action_a, action_b)


def test_finite_decider_reuses_prefix_and_q_validation():
    decider = Section06PolyhedralDecider(2, 1)

    with pytest.raises(ValueError, match="shape"):
        decider.add_candidate("0", [[1.0, 2.0], [0.0, 1.0]])
    decider.add_candidate("0", [[1.0], [0.0]])
    with pytest.raises(ValueError, match="prefix-free"):
        decider.add_candidate("01", [[1.0], [0.0]])
