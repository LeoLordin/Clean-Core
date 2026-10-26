import numpy as np
import pytest

from clean_core.config import D_SPATIAL, POLE
from clean_core.geometry.lorentz import exp_map
from clean_core.prototypes.sections.section_23 import (
    section_23_associative_attention,
    section_23_crossbar_current,
    section_23_project_memory,
    section_23_update,
)


def _point_at_distance(distance):
    tangent = np.zeros_like(POLE)
    tangent[1] = distance
    return exp_map(POLE.copy(), tangent)


def test_memory_projection_matches_origin_exponential_map():
    memory = np.array([[0.25], [-0.5]])
    key_weights = np.zeros((D_SPATIAL, 1))
    value_weights = np.zeros((D_SPATIAL, 1))
    key_weights[0, 0] = 2.0
    value_weights[1, 0] = -1.0

    result = section_23_project_memory(memory, key_weights, value_weights)

    expected_key_tangent = np.zeros_like(POLE)
    expected_key_tangent[1] = 0.5
    expected_value_tangent = np.zeros_like(POLE)
    expected_value_tangent[2] = -0.25
    np.testing.assert_allclose(
        result["keys"][0], exp_map(POLE.copy(), expected_key_tangent)
    )
    np.testing.assert_allclose(
        result["values"][0], exp_map(POLE.copy(), expected_value_tangent)
    )
    assert np.all(np.isfinite(result["keys"]))
    assert np.all(np.isfinite(result["values"]))


def test_associative_attention_matches_scores_weights_and_local_value_update():
    keys = np.stack([POLE, _point_at_distance(0.4)])
    values = np.stack([_point_at_distance(0.2), _point_at_distance(-0.3)])
    temperature = 0.5
    dopamine = 1.5
    scores = np.array(
        [0.0, -(0.4**2) / (temperature * np.sqrt(D_SPATIAL))]
    )
    weights = np.exp(scores - np.max(scores))
    weights /= np.sum(weights)
    expected_update = dopamine * (weights[0] * 0.2 - weights[1] * 0.3)

    result = section_23_associative_attention(
        POLE, keys, values, temperature, dopamine=dopamine
    )

    np.testing.assert_allclose(result["scores"], scores, atol=1e-12)
    np.testing.assert_allclose(result["weights"], weights, atol=1e-12)
    assert np.sum(result["weights"]) == pytest.approx(1.0)
    assert result["update"][0] == pytest.approx(expected_update, abs=1e-12)
    np.testing.assert_allclose(result["update"][1:], 0.0, atol=1e-12)


def test_crossbar_current_matches_linear_current_at_origin():
    analog_weights = np.eye(D_SPATIAL) * 2.0
    analog_weights[0, 1] = -0.5
    state = np.arange(1.0, D_SPATIAL + 1.0)

    current = section_23_crossbar_current(POLE, analog_weights, state)

    np.testing.assert_allclose(current, analog_weights @ state, atol=1e-12)


def test_update_combines_attention_with_crossbar_current():
    keys = np.stack([POLE, POLE])
    values = np.stack([_point_at_distance(0.4), _point_at_distance(-0.2)])
    analog_weights = np.eye(D_SPATIAL)
    state = np.zeros(D_SPATIAL)
    state[2] = 0.75

    result = section_23_update(
        POLE, keys, values, 1.0, analog_weights, state, dopamine=2.0
    )

    np.testing.assert_allclose(result["weights"], [0.5, 0.5])
    assert result["update"][0] == pytest.approx(0.2)
    np.testing.assert_allclose(result["crossbar_current"], state)
    np.testing.assert_allclose(
        result["total_update"], result["update"] + result["crossbar_current"]
    )


@pytest.mark.parametrize(
    ("temperature", "dopamine"),
    [(0.0, 1.0), (-1.0, 1.0), (np.nan, 1.0), (1.0, -0.1), (1.0, np.inf)],
)
def test_attention_rejects_invalid_temperature_and_dopamine(temperature, dopamine):
    with pytest.raises(ValueError):
        section_23_associative_attention(
            POLE, [POLE], [POLE], temperature, dopamine=dopamine
        )


@pytest.mark.parametrize(
    "call",
    [
        lambda: section_23_associative_attention(
            POLE[:-1], [POLE], [POLE], 1.0
        ),
        lambda: section_23_associative_attention(
            POLE, np.zeros((1, D_SPATIAL)), [POLE], 1.0
        ),
        lambda: section_23_associative_attention(
            POLE, [POLE], [], 1.0
        ),
        lambda: section_23_associative_attention(
            POLE, [POLE], [np.full_like(POLE, np.nan)], 1.0
        ),
        lambda: section_23_crossbar_current(
            POLE, np.eye(D_SPATIAL - 1), np.ones(D_SPATIAL)
        ),
        lambda: section_23_crossbar_current(
            POLE, np.eye(D_SPATIAL), np.full(D_SPATIAL, np.inf)
        ),
    ],
)
def test_section_23_rejects_invalid_shapes_and_non_finite_values(call):
    with pytest.raises(ValueError):
        call()
