import numpy as np
import pytest

from clean_core.prototypes.sections.section_16 import (
    log_acyclicity,
    log_acyclicity_gradient,
    schur_capped_intervention,
)


def test_log_acyclicity_is_zero_for_dag_and_positive_for_cycle():
    dag = np.array([[0.0, 0.7, 0.0], [0.0, 0.0, 0.4], [0.0, 0.0, 0.0]])
    cycle = dag.copy()
    cycle[2, 0] = 0.5

    assert log_acyclicity(dag) == pytest.approx(0.0, abs=1e-14)
    assert log_acyclicity(cycle) > 0.0
    np.testing.assert_allclose(log_acyclicity_gradient(dag), np.zeros_like(dag))
    assert np.all(np.isfinite(log_acyclicity_gradient(cycle)))


def test_log_acyclicity_gradient_matches_finite_differences():
    adjacency = np.array([[0.1, 0.4], [-0.2, 0.3]])
    gradient = log_acyclicity_gradient(adjacency)
    step = 1e-6

    for row in range(adjacency.shape[0]):
        for column in range(adjacency.shape[1]):
            perturbation = np.zeros_like(adjacency)
            perturbation[row, column] = step
            finite_difference = (
                log_acyclicity(adjacency + perturbation)
                - log_acyclicity(adjacency - perturbation)
            ) / (2.0 * step)
            assert gradient[row, column] == pytest.approx(
                finite_difference, rel=1e-6, abs=1e-8
            )


def test_intervention_preserves_dag_when_penalty_gradient_is_zero():
    dag = np.array([[0.0, 0.2, 0.0], [0.0, 0.0, 0.3], [0.0, 0.0, 0.0]])

    result = schur_capped_intervention(
        dag, np.zeros_like(dag), dt=0.1, epsilon1=1.0, mu_dag=2.0
    )

    np.testing.assert_allclose(result, dag, atol=1e-14)
    assert log_acyclicity(result) == pytest.approx(0.0, abs=1e-14)


def test_intervention_applies_formula_gradient_step():
    adjacency = np.zeros((2, 2))
    causal_gradient = np.array([[0.0, 0.8], [-0.4, 0.0]])

    result = schur_capped_intervention(
        adjacency,
        causal_gradient,
        dt=0.2,
        epsilon1=2.0,
        mu_dag=3.0,
    )

    np.testing.assert_allclose(result, 0.1 * causal_gradient, atol=1e-14)


def test_intervention_schur_caps_spectral_radius():
    adjacency = np.array([[0.0, 2.0], [2.0, 0.0]])

    result = schur_capped_intervention(
        adjacency,
        np.zeros_like(adjacency),
        dt=0.1,
        epsilon1=1.0,
        mu_dag=0.0,
        epsilon_gap=0.1,
    )

    assert max(abs(np.linalg.eigvals(result))) < 0.9


@pytest.mark.parametrize(
    ("adjacency", "gradient", "parameters"),
    [
        (np.ones((2, 3)), np.ones((2, 3)), {}),
        (np.eye(2), np.ones((3, 3)), {}),
        ([[0.0, np.nan], [1.0, 0.0]], np.zeros((2, 2)), {}),
        (np.eye(2), np.zeros((2, 2)), {"dt": 0.0}),
        (np.eye(2), np.zeros((2, 2)), {"epsilon1": 0.0}),
        (np.eye(2), np.zeros((2, 2)), {"mu_dag": -1.0}),
        (np.eye(2), np.zeros((2, 2)), {"epsilon_gap": 1.0}),
    ],
)
def test_intervention_rejects_invalid_shapes_values_and_parameters(
    adjacency, gradient, parameters
):
    options = {"dt": 0.1, "epsilon1": 1.0, **parameters}
    with pytest.raises(ValueError):
        schur_capped_intervention(adjacency, gradient, **options)


@pytest.mark.parametrize(
    "matrix",
    [
        np.empty((0, 0)),
        np.ones((2, 3)),
        np.array([[0.0, np.inf], [0.0, 0.0]]),
        np.array([[0.0, 1j], [0.0, 0.0]]),
    ],
)
def test_acyclicity_rejects_invalid_matrices(matrix):
    with pytest.raises(ValueError):
        log_acyclicity(matrix)
