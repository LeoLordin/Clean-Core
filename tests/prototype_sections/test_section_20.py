import random

import pytest

from clean_core.prototypes.sections.section_20 import StratifiedReplayReservoir


def test_reservoir_is_bounded_per_stratum_and_tracks_seen_items():
    reservoir = StratifiedReplayReservoir(
        capacity=7,
        strata=("recent", "older", "rare"),
        rng=random.Random(17),
    )

    for index in range(100):
        reservoir.add(("recent", "older", "rare")[index % 3], index)

    assert reservoir.stratum_capacity == 2
    assert reservoir.size == 6
    assert [reservoir.stratum_size(label) for label in reservoir.strata] == [2, 2, 2]
    assert [reservoir.seen_count(label) for label in reservoir.strata] == [34, 33, 33]


def test_reservoir_sampling_is_reproducible_with_seeded_rng():
    def retained_items():
        reservoir = StratifiedReplayReservoir(
            capacity=6,
            strata=("a", "b"),
            rng=random.Random(1234),
        )
        for value in range(40):
            reservoir.add("a" if value % 2 == 0 else "b", value)
        return reservoir.sample_by_stratum(count=5)

    assert retained_items() == retained_items()


def test_reservoir_replay_selection_is_uniform_over_stream_items():
    trials = 3000
    stream_size = 10
    retained_count = [0] * stream_size

    for seed in range(trials):
        reservoir = StratifiedReplayReservoir(
            capacity=2,
            strata=("all",),
            rng=random.Random(seed),
        )
        for item in range(stream_size):
            reservoir.add("all", item)
        for item in reservoir.sample_by_stratum(count=1)["all"]:
            retained_count[item] += 1

    expected = trials / stream_size
    assert all(expected * 0.85 < count < expected * 1.15 for count in retained_count)


def test_replay_batch_balances_non_empty_strata():
    reservoir = StratifiedReplayReservoir(
        capacity=6,
        strata=("a", "b", "not-yet-seen"),
        rng=random.Random(9),
    )
    reservoir.add("a", ("a", 1))
    reservoir.add("b", ("b", 1))

    batch = reservoir.sample(batch_size=7)
    counts = {
        stratum: sum(item[0] == stratum for item in batch)
        for stratum in ("a", "b")
    }

    assert len(batch) == 7
    assert sorted(counts.values()) == [3, 4]


def test_single_item_replay_selects_each_active_stratum_uniformly():
    reservoir = StratifiedReplayReservoir(
        capacity=3,
        strata=("a", "b", "c"),
        rng=random.Random(29),
    )
    for stratum in reservoir.strata:
        reservoir.add(stratum, stratum)

    trials = 6000
    selected = {stratum: 0 for stratum in reservoir.strata}
    for _ in range(trials):
        selected[reservoir.sample(batch_size=1)[0]] += 1

    expected = trials / len(reservoir.strata)
    assert all(expected * 0.9 < count < expected * 1.1 for count in selected.values())


def test_sample_by_stratum_returns_each_available_stratum():
    reservoir = StratifiedReplayReservoir(
        capacity=4,
        strata=("a", "b"),
        rng=random.Random(3),
    )
    reservoir.add("b", "remembered")

    assert reservoir.sample_by_stratum(count=3) == {"b": ["remembered"] * 3}


@pytest.mark.parametrize(
    ("capacity", "strata", "message"),
    [
        (0, ("a",), "capacity"),
        (True, ("a",), "capacity"),
        (1, (), "stratum"),
        (1, ("a", "b"), "capacity"),
        (2, ("a", "a"), "unique"),
    ],
)
def test_reservoir_validates_configuration(capacity, strata, message):
    with pytest.raises(ValueError, match=message):
        StratifiedReplayReservoir(capacity=capacity, strata=strata)


def test_reservoir_rejects_unknown_strata_and_empty_sampling():
    reservoir = StratifiedReplayReservoir(capacity=2, strata=("a", "b"))

    with pytest.raises(ValueError, match="unknown stratum"):
        reservoir.add("missing", "item")
    with pytest.raises(ValueError, match="empty reservoir"):
        reservoir.sample(batch_size=1)


@pytest.mark.parametrize("batch_size", [0, -1, True])
def test_sample_requires_positive_integer_batch_size(batch_size):
    reservoir = StratifiedReplayReservoir(capacity=1, strata=("a",))
    with pytest.raises(ValueError, match="batch_size"):
        reservoir.sample(batch_size)
