from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

import pytest

from clean_core.prototypes.sections.section_37 import (
    TIMESCALE_SECONDS,
    Section37MultiTimescaleDispatcher,
)


def _engines(value):
    return {timescale: (lambda *args, **kwargs: value) for timescale in TIMESCALE_SECONDS}


def test_dispatcher_defines_the_three_documented_timescales_and_cow_versions():
    dispatcher = Section37MultiTimescaleDispatcher(_engines("initial"))

    assert TIMESCALE_SECONDS == {
        "epsilon_0": 1e-3,
        "epsilon_1": 1.0,
        "epsilon_2": 1e5,
    }
    engines, version = dispatcher.snapshot()
    assert set(engines) == set(TIMESCALE_SECONDS)
    assert version == 0

    engines["epsilon_0"] = lambda: "modified snapshot"
    assert dispatcher.dispatch("epsilon_0") == "initial"

    assert dispatcher.publish("epsilon_0", lambda: "updated") == 1
    assert dispatcher.dispatch("epsilon_0") == "updated"
    assert dispatcher.dispatch("epsilon_1") == "initial"
    assert dispatcher.snapshot()[1] == 1


def test_dispatch_runs_captured_engine_outside_publication_lock():
    entered_engine = Event()
    resume_engine = Event()

    def blocked_engine():
        entered_engine.set()
        if not resume_engine.wait(timeout=3):
            raise TimeoutError("test did not release the in-flight engine")
        return "captured"

    dispatcher = Section37MultiTimescaleDispatcher(_engines("initial"))
    dispatcher.publish("epsilon_0", blocked_engine)

    with ThreadPoolExecutor(max_workers=1) as executor:
        in_flight = executor.submit(dispatcher.dispatch, "epsilon_0")
        try:
            assert entered_engine.wait(timeout=3)
            assert dispatcher.publish("epsilon_0", lambda: "new") == 2
            assert dispatcher.dispatch("epsilon_0") == "new"
        finally:
            resume_engine.set()
        assert in_flight.result(timeout=3) == "captured"


def test_concurrent_dispatch_and_publication_preserve_valid_engines_and_versions():
    dispatchers = 4
    iterations = 200
    start = Barrier(dispatchers + 1)
    dispatcher = Section37MultiTimescaleDispatcher(
        {
            timescale: (lambda scale=timescale: (scale, 0))
            for timescale in TIMESCALE_SECONDS
        }
    )

    def read_engines():
        results = []
        start.wait(timeout=3)
        for index in range(iterations):
            timescale = tuple(TIMESCALE_SECONDS)[index % len(TIMESCALE_SECONDS)]
            results.append(dispatcher.dispatch(timescale))
        return results

    def publish_engines():
        start.wait(timeout=3)
        for generation in range(1, iterations + 1):
            for timescale in TIMESCALE_SECONDS:
                dispatcher.publish(
                    timescale,
                    lambda scale=timescale, value=generation: (scale, value),
                )

    with ThreadPoolExecutor(max_workers=dispatchers + 1) as executor:
        readers = [executor.submit(read_engines) for _ in range(dispatchers)]
        writer = executor.submit(publish_engines)
        writer.result(timeout=5)
        results = [result for reader in readers for result in reader.result(timeout=5)]

    assert len(results) == dispatchers * iterations
    for timescale, generation in results:
        assert timescale in TIMESCALE_SECONDS
        assert 0 <= generation <= iterations
    assert dispatcher.snapshot()[1] == iterations * len(TIMESCALE_SECONDS)


@pytest.mark.parametrize(
    "engines",
    [
        {},
        {"epsilon_0": lambda: 0},
        {
            "epsilon_0": lambda: 0,
            "epsilon_1": lambda: 0,
            "epsilon_2": lambda: 0,
            "other": lambda: 0,
        },
    ],
)
def test_dispatcher_requires_exactly_the_three_named_timescales(engines):
    with pytest.raises(ValueError, match="exactly epsilon_0"):
        Section37MultiTimescaleDispatcher(engines)


def test_publish_rejects_unrecognized_timescales():
    dispatcher = Section37MultiTimescaleDispatcher(_engines("initial"))

    with pytest.raises(ValueError, match="unknown timescale"):
        dispatcher.publish("epsilon_3", lambda: "unexpected")


def test_dispatch_rejects_unrecognized_timescales():
    dispatcher = Section37MultiTimescaleDispatcher(_engines("initial"))

    with pytest.raises(KeyError, match="unknown timescale"):
        dispatcher.dispatch("epsilon_3")
