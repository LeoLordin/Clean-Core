import numpy as np
import pytest

from clean_core.cognition.aixi_bridge import BoundedActionHeuristic, SolomonoffAIXIBridge
from clean_core.cognition.causal_scm import NOTEARSCausalSCM
from clean_core.cognition.goedel_reflection import GoedelEgoReflectionEngine
from clean_core.config import POLE
from clean_core.geometry.lorentz import (
    exp_map,
    get_boost_frame,
    log_map,
    minkowski_dot,
    minkowski_norm_sq,
    parallel_transport,
)


def test_bounded_action_heuristic_returns_score_gap_and_normalized_pressure():
    heuristic = BoundedActionHeuristic(num_actions=3)

    action, margin, pressure = heuristic.select_action(np.array([-0.5, 0.1, 0.8]))

    assert action == 2
    assert margin > 0.0
    np.testing.assert_allclose(np.linalg.norm(pressure), margin, atol=1e-12)
    assert np.argmax(pressure) == action


def test_ties_do_not_claim_a_positive_margin_or_apply_pressure():
    heuristic = BoundedActionHeuristic(num_actions=3)

    action, margin, pressure = heuristic.select_action(np.zeros(1))

    assert action == 0
    assert margin == 0.0
    np.testing.assert_array_equal(pressure, np.zeros(3))


def test_legacy_aixi_name_is_an_explicit_heuristic_alias():
    assert SolomonoffAIXIBridge is BoundedActionHeuristic


def test_notears_log_acyclicity_is_zero_for_a_dag_and_positive_for_a_cycle():
    scm = NOTEARSCausalSCM(d_scm=3)
    scm.A_scm[0, 1] = 0.4
    scm.A_scm[1, 2] = -0.2
    assert abs(scm.compute_acyclicity_h()) < 1e-12

    scm.A_scm[2, 0] = 0.3
    assert scm.compute_acyclicity_h() > 0.0


def test_notears_update_enforces_zero_diagonal_and_spectral_radius_cap():
    scm = NOTEARSCausalSCM(d_scm=4)
    scm.A_scm[:] = 2.0
    np.fill_diagonal(scm.A_scm, 0.0)
    scm.update_scm_and_clip(np.random.default_rng(4).normal(size=(32, 4)))

    np.testing.assert_array_equal(np.diag(scm.A_scm), np.zeros(4))
    assert max(abs(np.linalg.eigvals(scm.A_scm))) <= 0.95 + 1e-12


def test_ego_reflection_lifts_tangent_velocity_at_non_pole_barycenter():
    engine = GoedelEgoReflectionEngine()
    barycenter = exp_map(POLE.copy(), np.array([0.0, 0.2, -0.1] + [0.0] * 13))
    errors = [np.linspace(-0.1, 0.1, 15)]

    mu_self, introspection = engine.step_ego_reflection(errors, barycenter)
    ego_tangent = log_map(barycenter, mu_self)

    np.testing.assert_allclose(minkowski_norm_sq(mu_self), -1.0, atol=1e-12)
    np.testing.assert_allclose(minkowski_norm_sq(introspection), -1.0, atol=1e-12)
    assert abs(minkowski_dot(barycenter, ego_tangent)) < 1e-12


def test_ego_reflection_parallel_transports_errors_to_barycenter_frame():
    engine = GoedelEgoReflectionEngine()
    engine.W_self_v.fill(0.0)
    barycenter = exp_map(POLE.copy(), np.array([0.0, 0.2, -0.1] + [0.0] * 13))
    neuron_mu = exp_map(POLE.copy(), np.array([0.0, -0.15, 0.05] + [0.0] * 13))
    error_local = np.linspace(-0.1, 0.1, 15)
    error_ambient = get_boost_frame(neuron_mu) @ error_local
    error_at_barycenter = parallel_transport(neuron_mu, barycenter, error_ambient)
    error_expected = get_boost_frame(barycenter).T @ (np.diag([-1.0] + [1.0] * 15) @ error_at_barycenter)
    expected_self = exp_map(
        barycenter,
        get_boost_frame(barycenter) @ (error_expected / 1.1),
    )

    actual_self, _ = engine.step_ego_reflection([(neuron_mu, error_local)], barycenter)

    np.testing.assert_allclose(actual_self, expected_self, atol=1e-12)


@pytest.mark.parametrize("observations", [[], [[1.0]], [np.nan]])
def test_heuristic_rejects_invalid_observations(observations):
    with pytest.raises(ValueError):
        BoundedActionHeuristic().select_action(observations)