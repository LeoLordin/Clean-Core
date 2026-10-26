import numpy as np

from clean_core.config import D_SPATIAL, POLE
from clean_core.geometry.lorentz import (
    exp_map,
    get_boost_frame,
    log_map,
    chordal_distance,
    minkowski_dot,
    minkowski_norm_sq,
    parallel_transport,
    streaming_frechet_barycenter,
)
from clean_core.geometry.symplectic import projected_kick_drift_step
from clean_core.estimation.fisher_rao import FisherRaoDLREstimator
from clean_core.geometry.eikonal import godunov_fast_sweeping_solver
from clean_core.core.inference import (
    compute_maupertuis_conformal_drift,
    system1_fast_inference_step,
    system1_query_relaxation,
    thermodynamic_langevin_noise,
)
from clean_core.core.neuron import NeuronMicrostate
from clean_core.core.quantizer import SoftVQQuantizer
from clean_core.config import ETA_L


def test_boost_frame_is_orthonormal_and_tangent():
    tangent = np.linspace(-0.2, 0.2, D_SPATIAL)
    mu = exp_map(POLE.copy(), np.concatenate(([0.0], tangent)))
    frame = get_boost_frame(mu)
    metric = np.diag([-1.0] + [1.0] * D_SPATIAL)

    np.testing.assert_allclose(frame.T @ metric @ frame, np.eye(D_SPATIAL), atol=1e-12)
    np.testing.assert_allclose(mu @ metric @ frame, np.zeros(D_SPATIAL), atol=1e-12)


def test_exp_and_log_maps_are_inverse_for_tangent_vectors():
    tangent = np.linspace(-0.1, 0.1, D_SPATIAL)
    tangent = np.concatenate(([0.0], tangent))
    mu = exp_map(POLE.copy(), tangent)

    np.testing.assert_allclose(log_map(POLE, mu), tangent, atol=1e-12)
    np.testing.assert_allclose(minkowski_norm_sq(mu), -1.0, atol=1e-12)


def test_chordal_distance_returns_hyperbolic_geodesic_distance():
    distance = 0.8
    tangent = np.zeros(D_SPATIAL + 1)
    tangent[1] = distance
    point = exp_map(POLE.copy(), tangent)

    np.testing.assert_allclose(chordal_distance(POLE, point), distance, atol=1e-12)


def test_parallel_transport_preserves_tangency_and_metric_norm():
    tangent = np.zeros(D_SPATIAL + 1)
    tangent[1] = 1.0
    target = exp_map(POLE.copy(), np.eye(D_SPATIAL + 1)[1] * 0.5)
    transported = parallel_transport(POLE, target, tangent)

    assert abs(minkowski_dot(target, transported)) < 1e-12
    assert abs(minkowski_norm_sq(transported) - minkowski_norm_sq(tangent)) < 1e-12


def test_streaming_frechet_barycenter_stays_on_hyperboloid():
    points = np.stack(
        [
            exp_map(POLE.copy(), np.concatenate(([0.0], np.eye(D_SPATIAL)[axis] * 0.2)))
            for axis in (0, 1)
        ]
    )

    barycenter = streaming_frechet_barycenter(POLE, points)

    np.testing.assert_allclose(minkowski_norm_sq(barycenter), -1.0, atol=1e-12)


def test_symplectic_step_preserves_phase_point_constraints():
    mu = exp_map(POLE.copy(), np.concatenate(([0.0], np.linspace(-0.05, 0.05, D_SPATIAL))))
    p = np.linspace(-0.1, 0.1, D_SPATIAL + 1)
    p += minkowski_dot(mu, p) * mu
    gradient = np.linspace(0.02, 0.08, D_SPATIAL + 1)

    mu_next, p_next = projected_kick_drift_step(mu, p, gradient, dt=0.005)

    assert abs(minkowski_norm_sq(mu_next) + 1.0) < 1e-12
    assert abs(minkowski_dot(mu_next, p_next)) < 1e-12


def test_dlr_precision_matches_direct_covariance_inverse():
    estimator = FisherRaoDLREstimator(rank=2, dim=4)
    estimator.d_diag = np.array([0.5, 0.8, 1.2, 1.5], dtype=np.float32)
    estimator.V_basis = np.eye(4, 2, dtype=np.float32)
    estimator.s_values = np.array([0.3, 0.6], dtype=np.float32)

    covariance = np.diag(estimator.d_diag) + estimator.V_basis @ np.diag(
        estimator.s_values
    ) @ estimator.V_basis.T

    np.testing.assert_allclose(
        estimator.get_precision_matrix(), np.linalg.inv(covariance), rtol=1e-6, atol=1e-6
    )


def test_dlr_update_keeps_covariance_factors_finite_and_orthonormal():
    estimator = FisherRaoDLREstimator(rank=3, dim=6)
    estimator.update_error(np.arange(6, dtype=np.float32) / 5.0)

    assert np.all(np.isfinite(estimator.d_diag))
    assert np.all(np.isfinite(estimator.s_values))
    assert np.all(estimator.d_diag > 0.0)
    assert np.all(estimator.s_values > 0.0)
    np.testing.assert_allclose(estimator.V_basis.T @ estimator.V_basis, np.eye(3), atol=1e-6)


def test_softvq_uses_hyperbolic_distances_and_normalized_soft_weights():
    quantizer = SoftVQQuantizer(k_max=4, dim=D_SPATIAL)
    quantizer.centroids = np.zeros((2, D_SPATIAL))
    quantizer.centroids[1, 0] = 0.2
    tangent = np.zeros(D_SPATIAL + 1)
    tangent[1] = 0.05
    point = exp_map(POLE.copy(), tangent)

    selected, centroid, weights = quantizer.quantize_tangent(point)

    assert selected == 0
    np.testing.assert_allclose(centroid, POLE, atol=1e-12)
    np.testing.assert_allclose(np.sum(weights), 1.0, atol=1e-12)
    assert weights[0] > weights[1]


def test_eikonal_fast_sweeping_uses_geodesic_edge_lengths():
    tangent = np.zeros(D_SPATIAL + 1)
    tangent[1] = 1.0
    nodes = np.stack([exp_map(POLE.copy(), distance * tangent) for distance in (0.0, 1.0, 2.0)])
    adjacency = np.array(
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
        dtype=np.int8,
    )

    action = godunov_fast_sweeping_solver(
        adjacency,
        nodes,
        np.ones(3),
        eps_visc=0.0,
    )

    np.testing.assert_allclose(action, [0.0, 1.0, 2.0], atol=1e-10)


def test_conformal_drift_matches_local_frame_equation():
    tangent = np.zeros(D_SPATIAL + 1)
    tangent[1] = 0.3
    mu = exp_map(POLE.copy(), tangent)
    velocity = np.linspace(-0.1, 0.1, D_SPATIAL)
    gradient = np.linspace(0.05, 0.15, D_SPATIAL)
    frame = get_boost_frame(mu)
    gradient_ambient = frame @ gradient
    gradient_local = frame.T @ (ETA_L * gradient_ambient)
    expected = -np.dot(velocity, velocity) * gradient_local + 2 * np.dot(velocity, gradient_local) * velocity

    np.testing.assert_allclose(
        compute_maupertuis_conformal_drift(mu, velocity, gradient_ambient), expected, atol=1e-12
    )


def test_langevin_noise_uses_temperature_scaling_and_reproducible_rng():
    first = thermodynamic_langevin_noise(8, 0.5, 0.01, rng=np.random.default_rng(3))
    second = thermodynamic_langevin_noise(8, 0.5, 0.01, rng=np.random.default_rng(3))

    np.testing.assert_array_equal(first, second)
    assert np.all(np.isfinite(first))
    assert np.any(first != 0.0)
    np.testing.assert_array_equal(thermodynamic_langevin_noise(8, 0.0, 0.01), np.zeros(8))


def test_inference_error_is_projected_into_updated_neuron_frame():
    neuron = NeuronMicrostate()
    parent = exp_map(POLE.copy(), np.concatenate(([0.0], np.linspace(-0.02, 0.02, D_SPATIAL))))

    mu_next, error_local = system1_fast_inference_step(neuron, parent, POLE)
    expected_error = get_boost_frame(mu_next).T @ (ETA_L * log_map(mu_next, parent))

    np.testing.assert_allclose(error_local, expected_error, atol=1e-12)
    np.testing.assert_allclose(neuron.eps_tilde, expected_error.astype(np.float32), atol=1e-7)


def test_system1_query_relaxation_preserves_weights_and_manifold_state():
    neuron = NeuronMicrostate()
    weights_before = (neuron.W1.copy(), neuron.W2.copy())
    query = exp_map(POLE.copy(), np.concatenate(([0.0], np.linspace(-0.01, 0.01, D_SPATIAL))))

    states, errors = system1_query_relaxation([neuron], query, POLE, steps=4)

    assert states.shape == (1, D_SPATIAL + 1)
    assert len(errors) == 1
    assert abs(minkowski_norm_sq(states[0]) + 1.0) < 1e-12
    np.testing.assert_array_equal(neuron.W1, weights_before[0])
    np.testing.assert_array_equal(neuron.W2, weights_before[1])
