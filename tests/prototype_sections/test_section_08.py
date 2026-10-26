import numpy as np
import pytest

from clean_core.prototypes.sections.section_08 import projected_phr_step


def test_projected_phr_step_updates_dual_objective_gradient_and_state():
    result = projected_phr_step(
        state=[2.0, -2.0],
        objective_value=3.0,
        objective_gradient=[3.0, -4.0],
        constraint_value=1.5,
        limit=1.0,
        constraint_gradient=[1.0, 2.0],
        multiplier=0.5,
        penalty=2.0,
        step_size=0.5,
        lower_bound=[-1.0, -1.0],
        upper_bound=[1.0, 1.0],
    )

    assert result["violation"] == pytest.approx(0.5)
    assert result["multiplier"] == pytest.approx(1.5)
    assert result["objective"] == pytest.approx(3.5)
    np.testing.assert_allclose(result["gradient"], [4.5, -1.0])
    np.testing.assert_allclose(result["state"], [-0.25, -1.0])


def test_feasible_inequality_projects_inactive_multiplier_and_state():
    result = projected_phr_step(
        state=[2.0],
        objective_value=1.0,
        objective_gradient=[-2.0],
        constraint_value=0.0,
        limit=1.0,
        constraint_gradient=[4.0],
        multiplier=0.2,
        penalty=1.0,
        step_size=1.0,
        lower_bound=[0.0],
        upper_bound=[1.0],
    )

    assert result["multiplier"] == 0.0
    assert result["objective"] == pytest.approx(0.98)
    np.testing.assert_allclose(result["gradient"], [-2.0])
    np.testing.assert_allclose(result["state"], [1.0])


@pytest.mark.parametrize(
    "overrides",
    [
        {"multiplier": -0.1},
        {"penalty": 0.0},
        {"step_size": -0.1},
        {"constraint_value": float("nan")},
        {"lower_bound": [2.0], "upper_bound": [1.0]},
        {"objective_gradient": [1.0, 2.0]},
    ],
)
def test_projected_phr_step_rejects_invalid_inputs(overrides):
    arguments = {
        "state": [0.0],
        "objective_value": 1.0,
        "objective_gradient": [1.0],
        "constraint_value": 0.5,
        "limit": 1.0,
        "constraint_gradient": [1.0],
        "multiplier": 0.0,
        "penalty": 1.0,
        "step_size": 0.1,
        "lower_bound": [-1.0],
        "upper_bound": [1.0],
    }
    arguments.update(overrides)

    with pytest.raises(ValueError):
        projected_phr_step(**arguments)
