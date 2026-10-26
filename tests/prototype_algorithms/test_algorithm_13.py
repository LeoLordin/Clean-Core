from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from time import sleep

import pytest

from clean_core.prototypes.algorithms.algorithm_13 import Algorithm13EngineDispatcher


def test_algorithm_13_versioned_swap_and_timescale_dispatch():
    dispatcher = Algorithm13EngineDispatcher(
        {"fast": lambda value: value + 1, "slow": lambda value: value - 1}
    )

    assert dispatcher.dispatch("fast", 4) == 5
    assert dispatcher.dispatch("slow", 4) == 3
    changed, version = dispatcher.swap_if_better(
        "fast", lambda value: value * 3, lambda engine: engine(2)
    )

    assert (changed, version) == (True, 1)
    assert dispatcher.dispatch("fast", 4) == 12
    assert dispatcher.dispatch("slow", 4) == 3
    engines, snapshot_version = dispatcher.snapshot()
    assert snapshot_version == 1
    assert engines["fast"](4) == 12


def test_algorithm_13_does_not_publish_non_improving_candidate():
    incumbent = lambda: 2
    dispatcher = Algorithm13EngineDispatcher({"fast": incumbent})

    changed, version = dispatcher.swap_if_better("fast", lambda: 1, lambda engine: engine())

    assert (changed, version) == (False, 0)
    engines, snapshot_version = dispatcher.snapshot()
    assert snapshot_version == 0
    assert engines["fast"] is incumbent


def test_algorithm_13_discards_swap_when_publication_changes_during_evaluation():
    dispatcher = Algorithm13EngineDispatcher({"fast": lambda: 1})
    candidate = lambda: 3
    candidate_evaluated = Event()
    continue_evaluation = Event()

    def utility(engine):
        score = engine()
        if engine is candidate:
            candidate_evaluated.set()
            assert continue_evaluation.wait(timeout=5)
        return score

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending_swap = pool.submit(dispatcher.swap_if_better, "fast", candidate, utility)
        assert candidate_evaluated.wait(timeout=5)
        published_version = dispatcher.publish("fast", lambda: 2)
        continue_evaluation.set()
        changed, version = pending_swap.result(timeout=5)

    assert published_version == 1
    assert (changed, version) == (False, 1)
    assert dispatcher.dispatch("fast") == 2


def test_algorithm_13_publish_many_exposes_coherent_snapshots_concurrently():
    generation = lambda value: lambda: value
    dispatcher = Algorithm13EngineDispatcher({"fast": generation(0), "slow": generation(0)})
    start = Barrier(5)

    def publish_generations():
        start.wait(timeout=5)
        for value in range(1, 301):
            dispatcher.publish_many(
                {"fast": generation(value), "slow": generation(value)}
            )
            sleep(0)

    def read_snapshots():
        start.wait(timeout=5)
        previous_version = -1
        for _ in range(500):
            engines, version = dispatcher.snapshot()
            assert engines["fast"]() == engines["slow"]()
            assert version >= previous_version
            previous_version = version
            sleep(0)
        return previous_version

    with ThreadPoolExecutor(max_workers=5) as pool:
        publisher = pool.submit(publish_generations)
        readers = [pool.submit(read_snapshots) for _ in range(4)]
        publisher.result(timeout=10)
        for reader in readers:
            assert reader.result(timeout=10) >= 0

    engines, version = dispatcher.snapshot()
    assert version == 300
    assert engines["fast"]() == engines["slow"]() == 300


@pytest.mark.parametrize(
    ("engines", "error", "message"),
    [
        ({}, ValueError, "non-empty"),
        ({"": lambda: None}, ValueError, "timescale"),
        ({"fast": None}, TypeError, "callable"),
    ],
)
def test_algorithm_13_validates_initial_engines(engines, error, message):
    with pytest.raises(error, match=message):
        Algorithm13EngineDispatcher(engines)


def test_algorithm_13_rejects_unknown_timescale_and_invalid_publication():
    dispatcher = Algorithm13EngineDispatcher({"fast": lambda: 1})

    with pytest.raises(KeyError, match="unknown timescale"):
        dispatcher.dispatch("slow")
    with pytest.raises(ValueError, match="non-empty"):
        dispatcher.publish_many({})
    with pytest.raises(TypeError, match="callable"):
        dispatcher.publish("fast", None)
