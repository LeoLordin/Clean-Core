import numpy as np
import pytest

from clean_core.prototypes.sections.section_34 import Section34ExplicitProjection


def test_finite_prefix_bounds_use_kraft_weights_normalizer_and_unresolved_mass():
    projection = Section34ExplicitProjection(
        2, 1, r_max=1.0, discount=0.5, m_k=2.0, epsilon_safe=0.4
    )
    projection.add_candidate("0", [[2.0], [0.0]])

    partial = projection.advance()
    np.testing.assert_allclose(partial["q_min"], [0.5, 0.0])
    np.testing.assert_allclose(partial["q_max"], [1.5, 1.0])
    assert partial["unresolved_mass"] == 0.5
    assert not partial["bounds_separate"]
    np.testing.assert_array_equal(partial["motor_pressure"], [0.0, 0.0])

    projection.add_candidate("1", [[2.0], [0.0]])
    complete = projection.advance()
    np.testing.assert_allclose(complete["q_min"], [1.0, 0.0])
    np.testing.assert_allclose(complete["q_max"], [1.0, 0.0])
    assert complete["unresolved_mass"] == 0.0
    assert complete["bounds_separate"]
    assert complete["action"] == 0
    np.testing.assert_allclose(
        complete["motor_pressure"],
        [0.4 / np.sqrt(2.0), -0.4 / np.sqrt(2.0)],
    )
    np.testing.assert_allclose(np.linalg.norm(complete["motor_pressure"]), 0.4)


def test_exact_action_boundary_is_not_strict_separation_and_has_no_pressure():
    projection = Section34ExplicitProjection(2, 1, r_max=1.0, discount=0.5)
    projection.add_candidate("0", [[1.0], [1.0]])
    projection.add_candidate("1", [[1.0], [1.0]])

    result = projection.advance(2)

    np.testing.assert_array_equal(result["q_min"], [1.0, 1.0])
    np.testing.assert_array_equal(result["q_max"], [1.0, 1.0])
    assert result["separation"] == 0.0
    assert not result["bounds_separate"]
    np.testing.assert_array_equal(result["motor_pressure"], [0.0, 0.0])


def test_action_pressure_uses_each_competitors_exact_margin():
    projection = Section34ExplicitProjection(
        3, 1, r_max=1.0, discount=0.5, epsilon_safe=2.0, m_k=1.0
    )
    projection.add_candidate("0", [[2.0], [0.0], [1.0]])
    projection.add_candidate("1", [[2.0], [0.0], [1.0]])

    result = projection.advance(2, delta_eps=0.5)

    weights = np.array([1.0 / 2.5, 1.0 / 1.5])
    direction = np.array([weights.sum(), -weights[0], -weights[1]])
    expected = 2.0 * direction / np.linalg.norm(direction)
    np.testing.assert_allclose(result["motor_pressure"], expected)
    np.testing.assert_allclose(np.linalg.norm(result["motor_pressure"]), 2.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"m_k": 0.0}, "m_k"),
        ({"m_k": np.inf}, "m_k"),
        ({"epsilon_safe": -1.0}, "epsilon_safe"),
        ({"epsilon_safe": np.nan}, "epsilon_safe"),
    ],
)
def test_configuration_requires_positive_finite_normalization_and_pressure(
    kwargs, message
):
    with pytest.raises(ValueError, match=message):
        Section34ExplicitProjection(2, 1, **kwargs)


def test_delta_epsilon_must_be_positive_and_finite():
    projection = Section34ExplicitProjection(2, 1)
    projection.add_candidate("0", [[1.0], [0.0]])

    for delta_eps in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError, match="delta_eps"):
            projection.decision(delta_eps=delta_eps)


def test_candidate_prefix_and_return_bound_validation_are_inherited():
    projection = Section34ExplicitProjection(2, 1, r_max=1.0, discount=0.5)
    with pytest.raises(ValueError, match="discounted-return bounds"):
        projection.add_candidate("0", [[2.1], [0.0]])

    projection.add_candidate("0", [[2.0], [0.0]])
    with pytest.raises(ValueError, match="prefix-free"):
        projection.add_candidate("01", [[1.0], [0.0]])
