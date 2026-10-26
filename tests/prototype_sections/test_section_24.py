import numpy as np
import pytest

from clean_core.config import D_SPATIAL, POLE
from clean_core.geometry.lorentz import exp_map
from clean_core.prototypes.sections.section_24 import (
    finite_step_decay_diagnostic,
    lyapunov_candidate,
)


def test_lyapunov_candidate_is_positive_away_from_equilibrium():
    tangent = np.zeros(D_SPATIAL + 1)
    tangent[1] = 0.4
    state = exp_map(POLE.copy(), tangent)

    assert lyapunov_candidate(np.array([state]), POLE) > 0.0


def test_lyapunov_candidate_is_positive_for_action_error():
    assert lyapunov_candidate(
        np.array([POLE]), POLE, actions=[1.0, -2.0], target_actions=[0.0, 0.0]
    ) == pytest.approx(2.5)


def test_lyapunov_candidate_vanishes_at_reference():
    assert lyapunov_candidate(
        np.array([POLE]), POLE, actions=[1.0], target_actions=[1.0]
    ) == 0.0


def test_finite_step_decay_diagnostic_compares_measured_and_allowed_rates():
    result = finite_step_decay_diagnostic(
        value_before=2.0,
        value_after=1.7,
        dt=0.1,
        alpha_damp=1.0,
        sigma_noise=0.5,
    )

    assert result["observed_rate"] == pytest.approx(-3.0)
    assert result["allowed_rate"] == pytest.approx(-1.75)
    assert result["rate_residual"] == pytest.approx(-1.25)
    assert result["within_bound"] is True


def test_finite_step_decay_diagnostic_reports_sampled_bound_violation():
    result = finite_step_decay_diagnostic(
        value_before=2.0,
        value_after=2.1,
        dt=0.1,
        alpha_damp=1.0,
    )

    assert result["within_bound"] is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dt": 0.0}, "dt must be positive"),
        ({"value_before": -1.0}, "non-negative"),
        ({"alpha_damp": -1.0}, "alpha_damp"),
    ],
)
def test_finite_step_decay_diagnostic_validates_inputs(kwargs, message):
    arguments = {
        "value_before": 1.0,
        "value_after": 0.9,
        "dt": 0.1,
        "alpha_damp": 1.0,
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        finite_step_decay_diagnostic(**arguments)
