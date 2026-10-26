import numpy as np

from clean_core.prototypes.algorithms.algorithm_11 import Algorithm11AnytimeDecider


def test_algorithm_11_budget_iterations_transition_from_undecided_to_separated():
    decider = Algorithm11AnytimeDecider(
        2, 1, r_max=1.0, discount=0.0, m_k=1.0
    )
    q_values = [[0.8], [0.2]]
    decider.add_candidate("0", q_values)
    decider.add_candidate("1", q_values)

    partial = decider.iterate(budget=1)

    assert partial["evaluated_programs"] == 1
    assert partial["pending_programs"] == 1
    assert partial["unresolved_mass"] == 0.5
    assert not partial["bounds_separate"]
    assert partial["fixed_action"] is None
    assert partial["epsilon_safe"] == 0.0

    complete = decider.iterate(budget=1)

    assert complete["evaluated_programs"] == 2
    assert complete["pending_programs"] == 0
    assert complete["unresolved_mass"] == 0.0
    assert complete["bounds_separate"]
    assert complete["action"] == complete["fixed_action"] == 0
    np.testing.assert_allclose(complete["q_min"], [0.8, 0.2])
    np.testing.assert_allclose(complete["q_max"], [0.8, 0.2])
    np.testing.assert_allclose(complete["epsilon_safe"], 0.6)
    np.testing.assert_allclose(np.linalg.norm(complete["motor_pressure"]), 1.0)
