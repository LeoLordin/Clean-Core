import pytest

from clean_core.prototypes.algorithms.algorithm_09 import Algorithm09


def _algorithm(seed=7, **options):
    return Algorithm09(
        capacity=8,
        clusters=("a", "b"),
        seed=seed,
        **options,
    )


def test_algorithm_09_sleep_replay_is_seed_deterministic_and_applies_negative_rate():
    first = _algorithm(eta_replay=0.25)
    second = _algorithm(eta_replay=0.25)
    for algorithm in (first, second):
        for cluster in ("a", "b"):
            for episode in range(6):
                algorithm.remember(cluster, (cluster, episode))

    def consolidate(algorithm):
        seen = {}

        def project(episode):
            return f"projected:{episode[0]}:{episode[1]}"

        def natural_gradient(reconstructions, episodes, step_size):
            seen.update(
                reconstructions=reconstructions,
                episodes=episodes,
                step_size=step_size,
            )
            return "updated"

        assert algorithm.sleep(5, project, natural_gradient) == "updated"
        return seen

    first_batch = consolidate(first)
    second_batch = consolidate(second)

    assert first_batch == second_batch
    assert len(first_batch["episodes"]) == 5
    assert {episode[0] for episode in first_batch["episodes"]} == {"a", "b"}
    assert first_batch["reconstructions"] == tuple(
        f"projected:{cluster}:{episode}"
        for cluster, episode in first_batch["episodes"]
    )
    assert first_batch["step_size"] == -0.25
    assert first.replay_buffer.size <= first.replay_buffer.capacity


def test_algorithm_09_rollback_restores_snapshot_and_regrounds_sensory_layer():
    algorithm = _algorithm()
    checkpoint = {
        "neurons": {"state": [1, 2]},
        "kfac": {"factor": [3, 4]},
        "episodic_memory": ["saved"],
        "sensory_layer": "old stream",
    }
    algorithm.save_checkpoint(checkpoint)
    checkpoint["neurons"]["state"][0] = 99
    current = {"neurons": {"state": [9]}, "transient": True}

    result = algorithm.wake(5.1, 10.0, current, {"input": "now"})

    assert result.rolled_back
    assert current == {
        "neurons": {"state": [1, 2]},
        "kfac": {"factor": [3, 4]},
        "episodic_memory": ["saved"],
        "sensory_layer": {"input": "now"},
        "temperatures": 1.0,
    }
    assert result.consecutive_resets == 1
    assert result.cooldown == 2.0
    assert result.temperature == 1.0
    current["neurons"]["state"][0] = 42
    assert algorithm.checkpoints.latest()["neurons"]["state"] == [1, 2]


def test_algorithm_09_impasse_threshold_cooldown_and_backoff():
    algorithm = _algorithm(cooldown_initial=2.0, cooldown_max=5.0)
    algorithm.save_checkpoint({"state": "checkpoint"})
    current = {"state": "initial"}

    assert not algorithm.wake(5.0, 0.0, current, "stream").rolled_back
    first = algorithm.wake(5.01, 0.0, current, "stream")
    assert first.rolled_back
    assert first.cooldown == 4.0

    current["state"] = "changed"
    cooling = algorithm.wake(9.0, 3.99, current, "stream")
    assert not cooling.rolled_back
    assert current["state"] == "changed"
    second = algorithm.wake(9.0, 4.0, current, "stream")
    assert second.rolled_back
    assert second.cooldown == 5.0
    assert second.consecutive_resets == 2
    assert second.temperature == pytest.approx(0.8)


def test_algorithm_09_nonpositive_derivative_clears_failures_and_cooldown():
    algorithm = _algorithm(cooldown_initial=1.0, cooldown_max=10.0)
    algorithm.save_checkpoint({"state": "checkpoint"})
    current = {}
    assert algorithm.wake(6.0, 0.0, current, "stream").rolled_back
    assert algorithm.watchdog.cooldown == 2.0
    second = algorithm.wake(6.0, 2.0, current, "stream")
    assert second.rolled_back
    assert second.temperature == pytest.approx(0.8)

    result = algorithm.wake(0.0, 2.5, current, "stream")

    assert not result.rolled_back
    assert result.consecutive_resets == 0
    assert result.cooldown == 1.0
    assert result.temperature == pytest.approx(0.8)


def test_algorithm_09_bounded_checkpoint_ring_keeps_latest_entries():
    algorithm = _algorithm(checkpoint_capacity=2)
    for value in range(3):
        algorithm.save_checkpoint({"value": value})

    assert len(algorithm.checkpoints) == 2
    current = {}
    result = algorithm.wake(6.0, 0.0, current, "grounded")
    assert result.rolled_back
    assert current["value"] == 2


@pytest.mark.parametrize(
    ("derivative", "timestamp"),
    [
        (float("nan"), 1.0),
        (1.0, float("inf")),
        (1.0, -1.0),
    ],
)
def test_algorithm_09_rejects_invalid_watchdog_observations(derivative, timestamp):
    algorithm = _algorithm()
    with pytest.raises(ValueError):
        algorithm.wake(derivative, timestamp, {}, "stream")
