import numpy as np
import pytest

from clean_core.prototypes.sections.section_07 import (
    make_full_microstate_supervector,
    motor_target_action,
    section_07_motor_loop_step,
)


def test_supervector_snapshots_formula_fields_and_validates_dense_parameters():
    snapshot = make_full_microstate_supervector(
        [1.0, 0.0],
        [0.2],
        {"W1": [[1.0, 2.0]], "W2": [[3.0]], "b1": [0.5], "b2": [0.1]},
        [0.8],
        [[0.3]],
        [0.0, 1.0],
    )

    np.testing.assert_allclose(snapshot.mu, [1.0, 0.0])
    np.testing.assert_allclose(snapshot.theta["W2"], [[3.0]])
    with pytest.raises(ValueError, match="W2 input dimension"):
        make_full_microstate_supervector(
            [1.0], [0.0], {"W1": [[1.0, 2.0]], "W2": [[1.0, 2.0]], "b1": [0.0], "b2": [0.0]},
            [1.0], [0.0], [0.0],
        )
    with pytest.raises(ValueError, match="finite"):
        make_full_microstate_supervector(
            [np.nan], [0.0], {"W1": [[1.0]], "W2": [[1.0]], "b1": [0.0], "b2": [0.0]},
            [1.0], [0.0], [0.0],
        )


def test_motor_target_combines_geodesic_polyhedral_and_social_sources():
    target = motor_target_action(
        [0.5], [0.25], [-0.5], [[2.0]], [[1.0]], [[-1.0]], [0.1]
    )
    np.testing.assert_allclose(target, [np.tanh(1.85)])
    assert np.all(np.abs(target) <= 1.0)


def test_motor_loop_finite_difference_descent_and_action_bounds():
    result = section_07_motor_loop_step(
        current_action=[0.8],
        target_distribution=[0.5, 0.5],
        predict_state=lambda action: [
            1.0 / (1.0 + np.exp(-action[0])),
            1.0 - 1.0 / (1.0 + np.exp(-action[0])),
        ],
        causal_information=lambda action: float(action[0] ** 2),
        geodesic_velocity=[0.0],
        polyhedral_pressure=[0.0],
        social_action=[0.0],
        geodesic_weights=[[0.0]],
        polyhedral_weights=[[0.0]],
        social_weights=[[0.0]],
        bias=[0.0],
        dt=0.1,
        learning_rate=0.2,
        lambda_pearl=0.1,
        lambda_policy=0.5,
    )

    assert result.gradient[0] > 0.0
    assert result.action[0] < 0.8
    assert -1.0 <= result.action[0] <= 1.0
    assert result.target_kl > 0.0
    assert result.causal_information == pytest.approx(0.64)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"current_action": [1.1]},
        {"dt": 0.0},
        {"learning_rate": -1.0},
        {"lambda_pearl": -1.0},
        {"lambda_policy": -1.0},
        {"finite_difference_step": 0.0},
    ],
)
def test_motor_loop_rejects_invalid_inputs(kwargs):
    arguments = dict(
        current_action=[0.0],
        target_distribution=[1.0],
        predict_state=lambda _: [1.0],
        causal_information=lambda _: 0.0,
        geodesic_velocity=[0.0],
        polyhedral_pressure=[0.0],
        social_action=[0.0],
        geodesic_weights=[[0.0]],
        polyhedral_weights=[[0.0]],
        social_weights=[[0.0]],
        bias=[0.0],
    )
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        section_07_motor_loop_step(**arguments)


def test_motor_loop_rejects_incompatible_shapes_and_nonfinite_callback_values():
    with pytest.raises(ValueError, match="map its source"):
        motor_target_action([0.0], [0.0], [0.0], [[1.0, 2.0]], [[0.0]], [[0.0]], [0.0])
    with pytest.raises(ValueError, match="finite"):
        section_07_motor_loop_step(
            current_action=[0.0],
            target_distribution=[1.0],
            predict_state=lambda _: [np.nan],
            causal_information=lambda _: 0.0,
            geodesic_velocity=[0.0],
            polyhedral_pressure=[0.0],
            social_action=[0.0],
            geodesic_weights=[[0.0]],
            polyhedral_weights=[[0.0]],
            social_weights=[[0.0]],
            bias=[0.0],
        )
