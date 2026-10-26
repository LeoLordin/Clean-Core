from clean_core.memory.sleep_replay import SleepReplayAntiLivelock


def test_rollback_returns_latest_immutable_checkpoint_copy():
    replay = SleepReplayAntiLivelock(capacity=3)
    first = {"step": 1, "weights": [1.0]}
    replay.push_checkpoint(first)
    first["weights"][0] = 99.0
    replay.push_checkpoint({"step": 2, "weights": [2.0]})

    should_rollback, checkpoint = replay.evaluate_impasse_and_rollback([0, 3, 6, 9, 12])

    assert should_rollback
    assert checkpoint == {"step": 2, "weights": [2.0]}


def test_ring_buffer_discards_oldest_checkpoint_at_capacity():
    replay = SleepReplayAntiLivelock(capacity=2)
    for step in range(3):
        replay.push_checkpoint({"step": step})

    assert list(replay.ring_buffer) == [{"step": 1}, {"step": 2}]