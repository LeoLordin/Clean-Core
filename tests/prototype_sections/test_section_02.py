import numpy as np
import pytest

from clean_core.config import D_LORENTZ, ETA_L, POLE
from clean_core.geometry.lorentz import get_boost_frame
from clean_core.prototypes.sections.section_02 import (
    conformal_connection,
    conformal_drift,
    conformal_factor,
    conformal_metric,
)


def test_conformal_factor_and_metric_follow_the_section_equations():
    omega = conformal_factor(2.0, 5.0, delta_eps=0.1)
    assert omega == 3.0

    metric, inverse_metric = conformal_metric(omega)
    np.testing.assert_allclose(metric, omega**2 * np.diag(ETA_L))
    np.testing.assert_allclose(inverse_metric, omega**-2 * np.diag(ETA_L))
    np.testing.assert_allclose(metric @ inverse_metric, np.eye(D_LORENTZ))

    assert conformal_factor(5.0, 4.0, delta_eps=0.1) == 0.1


def test_conformal_connection_matches_component_formula():
    gradient = np.zeros(D_LORENTZ)
    gradient[1] = 0.25
    transformed = conformal_connection(gradient)

    assert transformed[1, 1, 1] == pytest.approx(0.25)
    assert transformed[1, 0, 0] == pytest.approx(0.25)
    assert transformed[0, 0, 1] == pytest.approx(0.25)

    background = np.ones((D_LORENTZ, D_LORENTZ, D_LORENTZ))
    np.testing.assert_allclose(conformal_connection(gradient, background), transformed + background)


def test_conformal_drift_uses_gradient_of_log_omega_in_local_frame():
    gradient = np.zeros(D_LORENTZ)
    gradient[1] = 0.4
    velocity = np.linspace(-0.2, 0.2, D_LORENTZ - 1)
    omega = 2.0
    gradient_ln_omega = -gradient / omega
    frame = get_boost_frame(POLE)
    internal_gradient = frame.T @ (ETA_L * gradient_ln_omega)
    expected = (
        -np.dot(velocity, velocity) * internal_gradient
        + 2.0 * np.dot(velocity, internal_gradient) * velocity
    )

    np.testing.assert_allclose(
        conformal_drift(POLE, velocity, 3.0, gradient, 5.0),
        expected,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        conformal_drift(POLE, velocity, 5.0, gradient, 4.0),
        np.zeros_like(velocity),
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda: conformal_factor(float("nan"), 1.0),
        lambda: conformal_factor(1.0, float("inf")),
        lambda: conformal_factor(1.0, 2.0, delta_eps=float("inf")),
        lambda: conformal_metric(0.0),
        lambda: conformal_metric(float("inf")),
        lambda: conformal_connection(np.zeros(2)),
        lambda: conformal_connection(np.zeros(D_LORENTZ), np.zeros((2, 2, 2))),
        lambda: conformal_connection(
            np.zeros(D_LORENTZ), np.full((D_LORENTZ,) * 3, np.nan)
        ),
        lambda: conformal_drift(np.zeros(D_LORENTZ), np.zeros(D_LORENTZ - 1), 1.0,
                                np.zeros(D_LORENTZ), 2.0),
        lambda: conformal_drift(POLE, np.full(D_LORENTZ - 1, np.inf), 1.0,
                                np.zeros(D_LORENTZ), 2.0),
        lambda: conformal_drift(POLE, np.zeros(D_LORENTZ - 1), float("nan"),
                                np.zeros(D_LORENTZ), 2.0),
    ],
)
def test_invalid_conformal_inputs_are_rejected(call):
    with pytest.raises(ValueError):
        call()


def test_conformal_drift_requires_tangent_gradient_and_finite_state():
    velocity = np.ones(D_LORENTZ - 1)
    with pytest.raises(ValueError, match="tangent"):
        conformal_drift(
            POLE,
            velocity,
            0.0,
            np.ones(D_LORENTZ),
            1.0,
        )
    with pytest.raises(ValueError, match="finite"):
        conformal_drift(
            POLE,
            velocity,
            0.0,
            np.full(D_LORENTZ, np.nan),
            1.0,
        )
