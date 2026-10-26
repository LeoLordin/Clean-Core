import numpy as np
import pytest

from clean_core.config import D_LORENTZ, D_SPATIAL, POLE
from clean_core.geometry.lorentz import get_boost_frame
from clean_core.prototypes.algorithms.algorithm_06 import algorithm_06_step
from clean_core.prototypes.sections.section_19 import regularized_resolvent
from clean_core.prototypes.sections.section_14 import geodesic_exp, geodesic_log


def _inputs():
    W_self_v = np.eye(D_SPATIAL) * 0.2
    W_self_hist = np.zeros((D_SPATIAL, 2))
    W_self_hist[0, :] = [0.1, -0.05]
    feat_hist = np.array([0.4, -0.2])
    b_self = np.zeros(D_SPATIAL)
    b_self[1] = 0.03
    return W_self_v, W_self_hist, feat_hist, b_self, POLE.copy(), POLE.copy()


def test_algorithm_06_computes_finite_reflection_and_bounded_update():
    W_self_v, W_self_hist, feat_hist, b_self, barycenter, introspection = _inputs()

    result = algorithm_06_step(
        W_self_v,
        W_self_hist,
        feat_hist,
        b_self,
        barycenter,
        introspection,
        dt=0.2,
        eta_self=0.1,
        gamma_ema=0.25,
    )

    v_self_star = regularized_resolvent(W_self_v) @ (
        W_self_hist @ feat_hist + b_self
    )
    expected_self = geodesic_exp(
        barycenter, get_boost_frame(barycenter) @ v_self_star
    )
    expected_introspection = geodesic_exp(
        introspection, 0.25 * geodesic_log(introspection, expected_self)
    )
    assert result.mu_self_star.shape == (D_LORENTZ,)
    assert result.introspection.shape == (D_LORENTZ,)
    np.testing.assert_allclose(result.mu_self_star, expected_self)
    np.testing.assert_allclose(result.introspection, expected_introspection)
    assert np.all(np.isfinite(result.gradient_tilde))
    assert np.linalg.norm(result.W_self_v, ord=2) <= 1.0 - 1e-3 + 1e-10
    for point in (result.mu_self_star, result.introspection):
        assert point[0] > 0.0
        assert np.isclose(point[0] ** 2 - np.dot(point[1:], point[1:]), 1.0)


def test_algorithm_06_regularized_adjoint_resolvent_is_bounded():
    matrix = np.eye(D_SPATIAL) * 20.0
    delta_reg = 0.2
    epsilon_gap = 0.05

    resolvent = regularized_resolvent(
        matrix, delta_reg=delta_reg, epsilon_gap=epsilon_gap
    )

    assert np.all(np.isfinite(resolvent))
    assert np.linalg.norm(resolvent, ord=2) <= 1.0 / (
        delta_reg + epsilon_gap
    ) + 1e-10


@pytest.mark.parametrize(
    ("input_index", "replacement", "message"),
    [
        (0, np.zeros((D_SPATIAL, D_SPATIAL - 1)), "W_self_v"),
        (1, np.zeros((D_SPATIAL - 1, 2)), "W_self_hist"),
        (2, [1.0], "feat_hist"),
        (3, np.full(D_SPATIAL, np.nan), "b_self"),
        (4, np.zeros(D_LORENTZ), "barycenter"),
        (5, np.zeros(D_LORENTZ), "introspection"),
    ],
)
def test_algorithm_06_rejects_invalid_array_inputs(input_index, replacement, message):
    inputs = list(_inputs())
    inputs[input_index] = replacement

    with pytest.raises(ValueError, match=message):
        algorithm_06_step(*inputs)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"delta_reg": 0.0}, "delta_reg"),
        ({"dt": -1.0}, "dt"),
        ({"eta_self": -1.0}, "eta_self"),
        ({"gamma_ema": 1.1}, "gamma_ema"),
        ({"epsilon_gap": 1.0}, "epsilon_gap"),
        ({"gamma_ema": np.nan}, "gamma_ema"),
    ],
)
def test_algorithm_06_rejects_invalid_options(options, message):
    with pytest.raises(ValueError, match=message):
        algorithm_06_step(*_inputs(), **options)
