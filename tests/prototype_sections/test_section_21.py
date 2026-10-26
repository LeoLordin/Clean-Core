import numpy as np
import pytest

from clean_core.config import D_SPATIAL, POLE
from clean_core.geometry.lorentz import exp_map
from clean_core.prototypes.sections.section_21 import SoftVQCodebookEvolution


def test_usage_decay_counts_assignments_and_prunes_deterministically():
    centroids = np.zeros((3, D_SPATIAL))
    centroids[2, 0] = 0.5
    codebook = SoftVQCodebookEvolution(
        centroids,
        usage=np.array([0.1, 0.001, 0.5]),
        max_size=4,
    )

    np.testing.assert_allclose(codebook.update_usage([2], gamma_usage=0.5), [0.05, 0.0005, 1.25])
    assert codebook.prune(theta_prune=0.05) == (1,)
    np.testing.assert_array_equal(codebook.active, [True, False, True])
    assert codebook.usage[0] == pytest.approx(0.05)


def test_pruning_keeps_protected_and_threshold_usage_codes_active():
    codebook = SoftVQCodebookEvolution(
        np.zeros((3, D_SPATIAL)),
        usage=[0.0, 0.049, 0.05],
        protected_indices=(0, 2),
    )

    assert codebook.prune(theta_prune=0.05) == (1,)
    np.testing.assert_array_equal(codebook.active, [True, False, True])


def test_spawn_is_seeded_and_reuses_a_pruned_slot():
    centroids = np.zeros((2, D_SPATIAL))
    centroids[1, 0] = 0.2
    first = SoftVQCodebookEvolution(
        centroids, usage=[1.0, 0.0], max_size=2, active=[True, False], seed=7
    )
    second = SoftVQCodebookEvolution(
        centroids, usage=[1.0, 0.0], max_size=2, active=[True, False], seed=7
    )
    tangent = np.zeros(D_SPATIAL + 1)
    tangent[1] = 0.8
    point = exp_map(POLE.copy(), tangent)
    error = np.full(D_SPATIAL, 2.0)
    precision = np.eye(D_SPATIAL)

    assert first.maybe_spawn(
        point, error, precision, theta_spawn=1.0, d_min=0.1, noise_scale=0.01
    )
    assert second.maybe_spawn(
        point, error, precision, theta_spawn=1.0, d_min=0.1, noise_scale=0.01
    )
    np.testing.assert_array_equal(first.centroids, second.centroids)
    np.testing.assert_array_equal(first.active, [True, True])
    assert first.usage[1] == 1.0
    assert len(first.centroids) == 2


def test_spawn_after_pruning_reuses_slot_without_exceeding_capacity():
    centroids = np.zeros((2, D_SPATIAL))
    centroids[1, 0] = 0.5
    codebook = SoftVQCodebookEvolution(
        centroids, usage=[1.0, 0.0], max_size=2
    )
    assert codebook.prune(theta_prune=0.05) == (1,)
    tangent = np.zeros(D_SPATIAL + 1)
    tangent[1] = 0.8

    assert codebook.maybe_spawn(
        exp_map(POLE.copy(), tangent),
        np.ones(D_SPATIAL),
        np.eye(D_SPATIAL),
        theta_spawn=1.0,
        d_min=0.1,
    )
    assert len(codebook.centroids) == codebook.max_size == 2
    np.testing.assert_array_equal(codebook.active, [True, True])
    assert codebook.usage[1] == 1.0


def test_spawn_respects_divergence_novelty_and_capacity_gates():
    codebook = SoftVQCodebookEvolution(np.zeros((1, D_SPATIAL)), max_size=1)
    tangent = np.zeros(D_SPATIAL + 1)
    tangent[1] = 1.0
    distant_point = exp_map(POLE.copy(), tangent)

    assert not codebook.maybe_spawn(
        distant_point, np.zeros(D_SPATIAL), np.eye(D_SPATIAL), theta_spawn=1.0
    )
    assert not codebook.maybe_spawn(
        POLE.copy(), np.ones(D_SPATIAL), np.eye(D_SPATIAL), theta_spawn=1.0
    )
    assert not codebook.maybe_spawn(
        distant_point, np.ones(D_SPATIAL), np.eye(D_SPATIAL), theta_spawn=1.0
    )
    assert len(codebook.centroids) == 1


@pytest.mark.parametrize(
    ("point", "error", "precision", "message"),
    [
        (np.full(D_SPATIAL + 1, np.nan), np.ones(D_SPATIAL), np.eye(D_SPATIAL), "mu_point"),
        (POLE, np.full(D_SPATIAL, np.inf), np.eye(D_SPATIAL), "prediction_error"),
        (POLE, np.ones(D_SPATIAL), -np.eye(D_SPATIAL), "precision"),
    ],
)
def test_spawn_rejects_invalid_geometry_or_state(point, error, precision, message):
    codebook = SoftVQCodebookEvolution(np.zeros((1, D_SPATIAL)), max_size=2)

    with pytest.raises(ValueError, match=message):
        codebook.maybe_spawn(point, error, precision)


def test_usage_update_rejects_non_finite_state():
    codebook = SoftVQCodebookEvolution(np.zeros((1, D_SPATIAL)))
    codebook.usage[0] = np.nan

    with pytest.raises(ValueError, match="usage"):
        codebook.update_usage([], gamma_usage=0.9)


def test_usage_update_preserves_large_finite_usage():
    codebook = SoftVQCodebookEvolution(np.zeros((1, D_SPATIAL)))
    codebook.usage[0] = np.finfo(float).max

    result = codebook.update_usage([0])
    assert np.all(np.isfinite(result))
    assert codebook.usage[0] == np.finfo(float).max


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"active": [True, 2]}, "active"),
        ({"protected_indices": (0, 1.5)}, "protected_indices"),
    ],
)
def test_constructor_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SoftVQCodebookEvolution(np.zeros((2, D_SPATIAL)), **kwargs)


def test_constructor_rejects_boolean_max_size():
    with pytest.raises(ValueError, match="max_size"):
        SoftVQCodebookEvolution(np.zeros((1, D_SPATIAL)), max_size=True)


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (lambda book: book.update_usage([], gamma_usage=np.nan), "gamma_usage"),
        (lambda book: book.update_usage([1.5]), "assignments"),
        (lambda book: book.prune(theta_prune=-0.1), "theta_prune"),
    ],
)
def test_updates_reject_invalid_values(operation, message):
    codebook = SoftVQCodebookEvolution(
        np.zeros((2, D_SPATIAL)), active=[True, False]
    )
    with pytest.raises(ValueError, match=message):
        operation(codebook)
