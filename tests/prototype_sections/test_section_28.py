import pytest

from clean_core.prototypes.sections.section_28 import Section28RollbackWatchdog


def test_checkpoint_ring_is_bounded_and_stores_independent_copies():
    watchdog = Section28RollbackWatchdog(capacity=2)
    original = {"step": 0, "nested": [1]}
    watchdog.push_checkpoint(original)
    original["nested"][0] = 99
    watchdog.push_checkpoint({"step": 1})
    watchdog.push_checkpoint({"step": 2})

    assert list(watchdog.ring_buffer) == [{"step": 1}, {"step": 2}]


def test_impasse_rolls_back_and_anneals_with_bounded_cooldown():
    watchdog = Section28RollbackWatchdog(
        capacity=2,
        window_size=3,
        impasse_threshold=1.0,
        cooldown_initial=2.0,
        cooldown_max=5.0,
        temperature_max=10.0,
    )
    checkpoint = {"step": 4, "state": [3]}
    watchdog.push_checkpoint(checkpoint)
    checkpoint["state"][0] = 99

    first = watchdog.observe(0.0, 0.0, {"input": "grounded"})
    assert not first.rolled_back
    assert first.slope is None
    assert not watchdog.observe(3.0, 0.0, {}).rolled_back
    event = watchdog.observe(6.0, 0.0, {"input": "grounded"})

    assert event.rolled_back
    assert event.checkpoint == {"step": 4, "state": [3]}
    assert event.grounded_state == {"input": "grounded"}
    assert event.slope == pytest.approx(3.0)
    assert event.consecutive_resets == 1
    assert event.temperature == pytest.approx(8.0)
    assert event.cooldown == pytest.approx(4.0)

    event.checkpoint["state"][0] = 0
    assert watchdog.ring_buffer[-1] == {"step": 4, "state": [3]}

    assert not watchdog.observe(9.0, 1.0, {}).rolled_back
    assert not watchdog.observe(12.0, 2.0, {}).rolled_back
    blocked_by_cooldown = watchdog.observe(15.0, 3.0, {})
    assert not blocked_by_cooldown.rolled_back
    assert blocked_by_cooldown.consecutive_resets == 1

    second = watchdog.observe(18.0, 4.0, {})
    assert second.rolled_back
    assert second.consecutive_resets == 2
    assert second.temperature == pytest.approx(6.4)
    assert second.cooldown == pytest.approx(5.0)


def test_non_increasing_loss_trend_resets_annealing_state():
    watchdog = Section28RollbackWatchdog(
        window_size=3,
        impasse_threshold=0.5,
        cooldown_initial=2.0,
        cooldown_max=8.0,
    )
    watchdog.push_checkpoint({"step": 1})

    assert not watchdog.observe(0.0, 0.0, {}).rolled_back
    assert not watchdog.observe(2.0, 0.0, {}).rolled_back
    assert watchdog.observe(4.0, 0.0, {}).rolled_back
    assert watchdog.consecutive_resets == 1

    assert not watchdog.observe(3.0, 1.0, {}).rolled_back
    recovered = watchdog.observe(2.0, 2.0, {})
    assert not recovered.rolled_back
    assert recovered.slope == pytest.approx(-1.0)
    assert recovered.consecutive_resets == 0
    assert recovered.cooldown == pytest.approx(2.0)
    assert recovered.temperature == pytest.approx(1.0)


def test_impasse_without_a_checkpoint_does_not_claim_rollback():
    watchdog = Section28RollbackWatchdog(window_size=2, impasse_threshold=0.0)

    assert not watchdog.observe(1.0, 0.0, {}).rolled_back
    result = watchdog.observe(2.0, 1.0, {})

    assert not result.rolled_back
    assert result.checkpoint is None
    assert result.consecutive_resets == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"capacity": 0}, "capacity"),
        ({"window_size": 1}, "window_size"),
        ({"impasse_threshold": -1.0}, "impasse_threshold"),
        ({"cooldown_initial": -1.0}, "cooldown_initial"),
        ({"cooldown_initial": 3.0, "cooldown_max": 2.0}, "cooldown_max"),
        ({"temperature_max": 0.0}, "temperature_max"),
    ],
)
def test_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Section28RollbackWatchdog(**kwargs)


def test_rejects_invalid_observations_and_regressing_time():
    watchdog = Section28RollbackWatchdog(window_size=2)
    with pytest.raises(ValueError, match="loss"):
        watchdog.observe(float("nan"), 0.0, {})
    with pytest.raises(ValueError, match="timestamp"):
        watchdog.observe(1.0, float("inf"), {})

    watchdog.observe(1.0, 2.0, {})
    with pytest.raises(ValueError, match="non-decreasing"):
        watchdog.observe(2.0, 1.0, {})
