from threading import Barrier, Event, Thread

from clean_core.prototypes.sections.section_35 import Section35EngineSlot


def test_concurrent_utility_winners_cannot_both_publish_from_same_version():
    initial = lambda value: value
    candidate_a = lambda value: value + 1
    candidate_b = lambda value: value + 2
    slot = Section35EngineSlot(initial)
    candidate_barrier = Barrier(2)
    utilities = {initial: 0.0, candidate_a: 1.0, candidate_b: 2.0}
    results = []

    def evaluate(engine):
        if engine in (candidate_a, candidate_b):
            candidate_barrier.wait(timeout=2)
        return utilities[engine]

    def attempt(candidate):
        results.append(slot.swap_if_better(candidate, evaluate))

    threads = [
        Thread(target=attempt, args=(candidate,))
        for candidate in (candidate_a, candidate_b)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert all(not thread.is_alive() for thread in threads)
    assert sorted(changed for changed, _ in results) == [False, True]
    assert [version for _, version in results] == [1, 1]
    published, version = slot.snapshot()
    assert published in (candidate_a, candidate_b)
    assert version == 1


def test_dispatch_keeps_its_snapshot_while_new_calls_use_published_callable():
    entered_old_callable = Event()
    release_old_callable = Event()

    def old_engine(value):
        entered_old_callable.set()
        assert release_old_callable.wait(timeout=2)
        return ("old", value)

    new_engine = lambda value: ("new", value)
    slot = Section35EngineSlot(old_engine)
    in_flight_results = []
    dispatch_thread = Thread(
        target=lambda: in_flight_results.append(slot.dispatch("request"))
    )

    dispatch_thread.start()
    assert entered_old_callable.wait(timeout=2)
    assert slot.swap_if_better(new_engine, lambda engine: engine is new_engine) == (
        True,
        1,
    )
    assert slot.dispatch("next") == ("new", "next")

    release_old_callable.set()
    dispatch_thread.join(timeout=3)

    assert not dispatch_thread.is_alive()
    assert in_flight_results == [("old", "request")]
    assert slot.snapshot() == (new_engine, 1)


def test_swap_requires_strictly_higher_utility():
    initial = lambda: "initial"
    candidate = lambda: "candidate"
    slot = Section35EngineSlot(initial)

    assert slot.swap_if_better(candidate, lambda _engine: 1.0) == (False, 0)
    assert slot.snapshot() == (initial, 0)
