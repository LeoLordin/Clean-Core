import numpy as np
import pytest

from clean_core.config import D_SPATIAL, POLE
from clean_core.geometry.lorentz import exp_map
from clean_core.prototypes.algorithms.algorithm_08 import Algorithm08Dynamics


def _make_dynamics():
    centroids = np.zeros((3, D_SPATIAL))
    centroids[1, 0] = 0.5
    centroids[2, 1] = 0.75
    return Algorithm08Dynamics(
        centroids,
        usage=[1.0, 0.001, 0.2],
        max_size=3,
        gamma_soc=1.0,
        n_min=2,
        eta_alpha=1.0,
        alpha_damp=0.06,
        alpha_min=0.05,
        delta_alpha_max=0.002,
    )


def test_step_deterministically_updates_criticality_and_prunes_usage():
    first = _make_dynamics()
    second = _make_dynamics()
    errors = [np.array([1.0]), np.array([2.0])]
    precisions = [np.eye(1), np.eye(1)]

    result = first.step(
        errors,
        precisions,
        dt=0.1,
        assignments=[2],
        gamma_usage=0.5,
        theta_prune=0.05,
    )
    repeated = second.step(
        errors,
        precisions,
        dt=0.1,
        assignments=[2],
        gamma_usage=0.5,
        theta_prune=0.05,
    )

    assert result["alpha_measured"] is not None
    assert result["alpha_damp"] == pytest.approx(0.062)
    assert abs(result["alpha_damp"] - 0.06) <= first.delta_alpha_max + 1e-12
    np.testing.assert_allclose(result["usage"], [0.5, 0.0005, 1.1])
    np.testing.assert_array_equal(result["active"], [True, False, True])
    assert result["pruned_indices"] == (1,)
    assert result["spawned"] is False
    assert repeated["alpha_damp"] == result["alpha_damp"]
    np.testing.assert_array_equal(repeated["usage"], result["usage"])
    np.testing.assert_array_equal(repeated["active"], result["active"])


def test_step_keeps_damping_at_its_lower_bound_and_ignores_small_tail():
    dynamics = Algorithm08Dynamics(
        np.zeros((1, D_SPATIAL)),
        n_min=2,
        alpha_damp=0.05,
        alpha_min=0.05,
        delta_alpha_max=0.001,
    )

    result = dynamics.step(
        [np.array([1.0])],
        [np.eye(1)],
        dt=1.0,
    )

    assert result["alpha_damp"] == 0.05
    assert result["alpha_measured"] is None
    assert result["log_tail_mean"] == 0.0


def test_step_spawns_only_after_pruning_within_capacity():
    dynamics = Algorithm08Dynamics(
        np.zeros((2, D_SPATIAL)),
        usage=[1.0, 0.0],
        max_size=2,
        n_min=1,
        alpha_damp=0.05,
    )
    tangent = np.zeros(D_SPATIAL + 1)
    tangent[1] = 1.0
    candidate = (
        exp_map(POLE.copy(), tangent),
        np.ones(D_SPATIAL),
        np.eye(D_SPATIAL),
    )

    result = dynamics.step(
        [],
        [],
        dt=0.1,
        theta_prune=0.05,
        spawn_candidate=candidate,
        theta_spawn=1.0,
        d_min=0.1,
    )

    assert result["pruned_indices"] == (1,)
    assert result["spawned"] is True
    assert dynamics.codebook.active.tolist() == [True, True]
    assert len(dynamics.codebook.centroids) == dynamics.codebook.max_size
    assert dynamics.codebook.usage[1] == 1.0


@pytest.mark.parametrize(
    ("errors", "precisions", "dt", "assignments", "message"),
    [
        ([np.array([1.0])], [], 0.1, [], "equal lengths"),
        ([np.array([np.nan])], [np.eye(1)], 0.1, [], "finite"),
        ([], [], 0.0, [], "dt"),
        ([], [], 0.1, [99], "existing codes"),
    ],
)
def test_step_rejects_invalid_inputs(errors, precisions, dt, assignments, message):
    dynamics = _make_dynamics()

    with pytest.raises(ValueError, match=message):
        dynamics.step(errors, precisions, dt, assignments)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"epsilon_min": 0.0}, "epsilon_min"),
        ({"gamma_soc": 1.1}, "gamma_soc"),
        ({"n_min": 0}, "n_min"),
        ({"alpha_damp": 0.01, "alpha_min": 0.05}, "alpha_damp"),
        ({"delta_alpha_max": -0.1}, "delta_alpha_max"),
    ],
)
def test_constructor_rejects_invalid_criticality_bounds(options, message):
    with pytest.raises(ValueError, match=message):
        Algorithm08Dynamics(np.zeros((1, D_SPATIAL)), **options)


def test_step_rejects_malformed_spawn_candidate():
    dynamics = _make_dynamics()

    with pytest.raises(ValueError, match="spawn_candidate"):
        dynamics.step([], [], 0.1, spawn_candidate=(POLE, np.ones(D_SPATIAL)))
