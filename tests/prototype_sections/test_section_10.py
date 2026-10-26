import numpy as np
import pytest

from clean_core.config import D_SPATIAL, ETA_L, POLE
from clean_core.geometry.lorentz import (
    chordal_distance,
    exp_map,
    get_boost_frame,
    log_map,
    minkowski_dot,
    minkowski_norm_sq,
)
from clean_core.prototypes.sections.section_10 import (
    aggregate_parent_context,
    introspection_tangent_context,
    retraction_perceptron,
    softvq_tangent,
    softvq_tangent_weights,
)


def test_parent_context_aggregation_and_introspection_projection():
    codes = np.zeros((2, D_SPATIAL))
    codes[0, 1] = 0.5
    codes[1, 1] = 1.5
    np.testing.assert_allclose(aggregate_parent_context(codes, [2.0, -1.0]), [0.0, -0.5, *([0.0] * (D_SPATIAL - 2))])

    reference_tangent = np.zeros(POLE.size)
    reference_tangent[2] = 0.4
    introspection = exp_map(POLE.copy(), reference_tangent)
    context = introspection_tangent_context(POLE, introspection)
    np.testing.assert_allclose(
        context,
        get_boost_frame(POLE).T @ (ETA_L * log_map(POLE, introspection)),
        atol=1e-12,
    )
    np.testing.assert_allclose(context[1], 0.4, atol=1e-12)


def test_softvq_uses_active_codes_and_preserves_null_token():
    centroids = np.zeros((4, D_SPATIAL))
    centroids[1, 0] = 0.25
    centroids[2, 0] = 0.8
    centroids[3, 0] = 1.2
    active = np.array([True, True, True, False])
    query = exp_map(POLE.copy(), get_boost_frame(POLE) @ centroids[1])

    weights = softvq_tangent_weights(
        query, centroids, active_mask=active, temperature=0.05
    )
    tangent_code = softvq_tangent(
        query, centroids, active_mask=active, temperature=0.05
    )

    assert weights[0] > 0.0
    assert weights[1] > weights[2]
    assert weights[3] == 0.0
    assert 0.0 < np.sum(weights) <= 1.0
    np.testing.assert_allclose(tangent_code, weights @ centroids)
    assert tangent_code.shape == (D_SPATIAL,)


def test_softvq_weights_match_active_distance_formula_and_ignore_pruned_codes():
    centroids = np.zeros((3, D_SPATIAL))
    centroids[1, 0] = 0.4
    centroids[2, 0] = 1e200
    active = np.array([True, True, False])
    query_tangent = np.zeros(POLE.size)
    query_tangent[1] = 1.0
    query = exp_map(POLE.copy(), query_tangent)
    temperature = 0.3
    epsilon = 1e-9

    code_points = [
        exp_map(POLE.copy(), get_boost_frame(POLE) @ centroids[index])
        for index in (0, 1)
    ]
    distances_sq = np.array(
        [chordal_distance(query, point) ** 2 for point in code_points]
    )
    unnormalized = np.exp(
        -(distances_sq - np.min(distances_sq)) / temperature
    )
    expected = np.array(
        [unnormalized[0], unnormalized[1], 0.0]
    ) / (np.sum(unnormalized) + epsilon)

    weights = softvq_tangent_weights(
        query,
        centroids,
        active_mask=active,
        temperature=temperature,
        epsilon=epsilon,
    )

    np.testing.assert_allclose(weights, expected, atol=1e-14)
    assert np.all(np.isfinite(weights))
    assert weights[2] == 0.0


def test_retraction_perceptron_clips_and_returns_tangent_prediction_error():
    w1 = np.zeros((2, D_SPATIAL))
    b1 = np.zeros(2)
    w2 = np.zeros((D_SPATIAL, 2))
    w2[0, 0] = 8.0
    w2[1, 1] = 4.0
    reference_tangent = np.zeros(POLE.size)
    reference_tangent[1] = 0.3
    reference = exp_map(POLE.copy(), reference_tangent)
    actual_tangent = np.zeros(POLE.size)
    actual_tangent[2] = 0.2
    actual = exp_map(reference.copy(), actual_tangent)

    result = retraction_perceptron(
        np.zeros(D_SPATIAL),
        reference,
        actual,
        w1,
        b1,
        w2,
        np.zeros(D_SPATIAL),
    )

    assert np.linalg.norm(result.raw_tangent_coordinates) > 3.0
    assert np.linalg.norm(result.tangent_coordinates) == pytest.approx(3.0)
    np.testing.assert_allclose(
        result.ambient_tangent,
        get_boost_frame(reference) @ result.tangent_coordinates,
        atol=1e-12,
    )
    assert minkowski_norm_sq(result.predicted_point) == pytest.approx(-1.0)
    assert result.predicted_point[0] > 0.0
    np.testing.assert_allclose(
        minkowski_dot(actual, result.error), 0.0, atol=1e-10
    )
    np.testing.assert_allclose(
        result.error, log_map(actual, result.predicted_point), atol=1e-12
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda: aggregate_parent_context(np.zeros((0, D_SPATIAL)), []),
        lambda: aggregate_parent_context(np.zeros((1, D_SPATIAL)), [np.nan]),
        lambda: softvq_tangent(
            POLE, np.zeros((2, D_SPATIAL)), active_mask=[False, True]
        ),
        lambda: softvq_tangent(
            POLE, np.zeros((2, D_SPATIAL)), active_mask=[True, False], temperature=0.0
        ),
        lambda: softvq_tangent(
            np.zeros_like(POLE), np.zeros((1, D_SPATIAL))
        ),
        lambda: retraction_perceptron(
            np.zeros(D_SPATIAL),
            POLE,
            POLE,
            np.zeros((1, D_SPATIAL)),
            np.zeros(2),
            np.zeros((D_SPATIAL, 1)),
            np.zeros(D_SPATIAL),
        ),
        lambda: retraction_perceptron(
            np.zeros(D_SPATIAL),
            POLE,
            POLE,
            np.zeros((1, D_SPATIAL)),
            np.zeros(1),
            np.zeros((D_SPATIAL, 1)),
            np.zeros(D_SPATIAL),
            max_radius=0.0,
        ),
    ],
)
def test_section_10_rejects_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()
