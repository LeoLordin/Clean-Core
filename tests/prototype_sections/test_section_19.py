import numpy as np
import pytest

from clean_core.prototypes.sections.section_19 import (
    ego_reflection_step,
    regularized_resolvent,
)


def test_regularized_resolvent_is_finite_and_has_the_predicted_norm_bound():
    delta_reg = 0.2
    epsilon_gap = 0.1
    resolvent = regularized_resolvent(
        np.array([[4.0, -2.0], [1.0, 3.0]]),
        delta_reg=delta_reg,
        epsilon_gap=epsilon_gap,
    )

    assert np.all(np.isfinite(resolvent))
    assert np.linalg.norm(resolvent, ord=2) <= 1.0 / (
        delta_reg + epsilon_gap
    ) + 1e-12


def test_regularized_resolvent_handles_zero_rank_deficient_and_pole_like_matrices():
    delta_reg = 0.2
    epsilon_gap = 0.1
    zero_resolvent = regularized_resolvent(
        np.zeros((2, 2)), delta_reg=delta_reg, epsilon_gap=epsilon_gap
    )
    np.testing.assert_allclose(zero_resolvent, np.eye(2) / (1.0 + delta_reg))

    rank_deficient_resolvent = regularized_resolvent(
        np.array([[2.0, 0.0], [0.0, 0.0]]),
        delta_reg=delta_reg,
        epsilon_gap=epsilon_gap,
    )
    np.testing.assert_allclose(
        rank_deficient_resolvent,
        np.diag([1.0 / (delta_reg + epsilon_gap), 1.0 / (1.0 + delta_reg)]),
    )

    # The unregularized system is singular, but spectral clipping keeps this solve finite.
    pole_like_resolvent = regularized_resolvent(
        (1.0 + delta_reg) * np.eye(2),
        delta_reg=delta_reg,
        epsilon_gap=epsilon_gap,
    )
    assert np.all(np.isfinite(pole_like_resolvent))
    assert np.linalg.norm(pole_like_resolvent, ord=2) <= 1.0 / (
        delta_reg + epsilon_gap
    ) + 1e-12
    huge_resolvent = regularized_resolvent(
        np.eye(2) * 1e308, delta_reg=delta_reg, epsilon_gap=epsilon_gap
    )
    assert np.all(np.isfinite(huge_resolvent))
    assert np.linalg.norm(huge_resolvent, ord=2) <= 1.0 / (
        delta_reg + epsilon_gap
    ) + 1e-12


def test_ego_reflection_step_matches_forward_and_adjoint_resolvent_equations():
    W_self_v = np.array([[0.1, 0.2], [-0.1, 0.05]])
    W_self_hist = np.array([[0.4, -0.2, 0.1], [0.3, 0.2, -0.5]])
    feat_hist = np.array([0.5, -0.3, 0.8])
    b_self = np.array([0.1, -0.2])
    g_tilde = np.array([-0.4, 0.7])
    delta_reg = 0.2
    dt = 0.05
    eta_self = 0.3
    system = (1.0 + delta_reg) * np.eye(2) - W_self_v
    expected_v = np.linalg.solve(system, W_self_hist @ feat_hist + b_self)
    adjoint = np.linalg.solve(system.T, g_tilde)
    step_size = dt * eta_self

    result = ego_reflection_step(
        W_self_v,
        W_self_hist,
        feat_hist,
        b_self,
        g_tilde,
        delta_reg=delta_reg,
        dt=dt,
        eta_self=eta_self,
    )

    np.testing.assert_allclose(result.v_self_star, expected_v)
    np.testing.assert_allclose(
        result.W_self_v,
        W_self_v - step_size * np.outer(adjoint, expected_v),
    )
    np.testing.assert_allclose(
        result.W_self_hist,
        W_self_hist - step_size * np.outer(adjoint, feat_hist),
    )
    np.testing.assert_allclose(result.b_self, b_self - step_size * adjoint)


def test_ego_reflection_update_enforces_spectral_norm_cap_and_is_finite():
    result = ego_reflection_step(
        np.eye(2) * 10.0,
        np.ones((2, 2)),
        np.array([1.0, -1.0]),
        np.zeros(2),
        np.ones(2),
        epsilon_gap=0.05,
    )

    assert np.linalg.norm(result.W_self_v, ord=2) <= 0.95 + 1e-12
    for array in (
        result.v_self_star,
        result.W_self_v,
        result.W_self_hist,
        result.b_self,
    ):
        assert np.all(np.isfinite(array))


def test_ego_reflection_step_handles_degenerate_zero_inputs():
    result = ego_reflection_step(
        np.zeros((1, 1)),
        np.zeros((1, 2)),
        np.zeros(2),
        np.zeros(1),
        np.zeros(1),
    )

    np.testing.assert_array_equal(result.v_self_star, np.zeros(1))
    np.testing.assert_array_equal(result.W_self_v, np.zeros((1, 1)))
    np.testing.assert_array_equal(result.W_self_hist, np.zeros((1, 2)))
    np.testing.assert_array_equal(result.b_self, np.zeros(1))


@pytest.mark.parametrize(
    "call",
    [
        lambda: regularized_resolvent(np.ones((2, 3))),
        lambda: regularized_resolvent(np.empty((0, 0))),
        lambda: regularized_resolvent(np.array([[np.nan]])),
        lambda: regularized_resolvent(np.array([[1.0j]])),
        lambda: regularized_resolvent(np.eye(2), delta_reg=0.0),
        lambda: regularized_resolvent(np.eye(2), epsilon_gap=1.0),
        lambda: ego_reflection_step(
            np.eye(2), np.ones((2, 3)), np.ones(2), np.zeros(2), np.ones(2)
        ),
        lambda: ego_reflection_step(
            np.eye(2), np.ones((2, 2)), np.ones(3), np.zeros(2), np.ones(2)
        ),
        lambda: ego_reflection_step(
            np.eye(2),
            np.ones((2, 2)),
            np.ones(2),
            np.zeros(2),
            np.ones(2),
            dt=-1.0,
        ),
        lambda: ego_reflection_step(
            np.eye(2),
            np.ones((2, 2)),
            np.ones(2),
            np.zeros(2),
            np.array([np.inf, 1.0]),
        ),
    ],
)
def test_section_19_rejects_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()
