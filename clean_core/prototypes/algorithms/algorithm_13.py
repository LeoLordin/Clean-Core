"""Thread-safe callable publication and multi-timescale dispatch for Algorithm 13.

This is a Python runtime prototype: utility comparisons are not proof-checked,
the callables are not JIT-compiled, and publication is lock-based rather than
lock-free RCU. No zero-stall or zero-jitter guarantee is provided.
"""

from collections.abc import Callable, Mapping
from threading import Lock
from typing import Any


Engine = Callable[..., Any]


class Algorithm13EngineDispatcher:
    """Versioned, copy-on-write engine map with optimistic utility-based swaps.

    Dispatch takes a synchronized snapshot of one callable and invokes it
    outside the lock. ``publish`` and ``swap_if_better`` atomically publish
    registry versions; ``publish_many`` updates several timescales in one
    publication. A conditional swap is rejected if any publication occurred
    while its utility evaluator was running.
    """

    def __init__(self, engines: Mapping[str, Engine]) -> None:
        if not isinstance(engines, Mapping) or not engines:
            raise ValueError("engines must be a non-empty mapping")
        initial_engines = dict(engines)
        self._validate_engines(initial_engines)
        self._engines = initial_engines
        self._version = 0
        self._lock = Lock()

    @staticmethod
    def _validate_engines(engines: Mapping[str, Engine]) -> None:
        if any(not isinstance(name, str) or not name for name in engines):
            raise ValueError("timescale names must be non-empty strings")
        if any(not callable(engine) for engine in engines.values()):
            raise TypeError("every engine must be callable")

    def snapshot(self) -> tuple[dict[str, Engine], int]:
        """Return a detached engine-map copy and its publication version."""
        with self._lock:
            return self._engines.copy(), self._version

    def dispatch(self, timescale: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke the engine selected for ``timescale`` from a synchronized read."""
        with self._lock:
            try:
                engine = self._engines[timescale]
            except KeyError:
                raise KeyError(f"unknown timescale: {timescale}") from None
        return engine(*args, **kwargs)

    def publish(self, timescale: str, engine: Engine) -> int:
        """Publish one callable at a named timescale and return its new version."""
        return self.publish_many({timescale: engine})

    def publish_many(self, engines: Mapping[str, Engine]) -> int:
        """Atomically publish one or more callable updates as a single version."""
        if not isinstance(engines, Mapping) or not engines:
            raise ValueError("engines must be a non-empty mapping")
        updates = dict(engines)
        self._validate_engines(updates)
        with self._lock:
            updated = self._engines.copy()
            updated.update(updates)
            self._engines = updated
            self._version += 1
            return self._version

    def swap_if_better(
        self, timescale: str, candidate: Engine, utility_evaluator: Callable[[Engine], float]
    ) -> tuple[bool, int]:
        """Publish ``candidate`` only if it scores strictly better than its incumbent.

        Utility evaluation happens outside the lock. If another publication
        changes the registry version before the candidate is committed, this
        optimistic swap is discarded rather than evaluated against stale state.
        """
        if not isinstance(timescale, str) or not timescale:
            raise ValueError("timescale must be a non-empty string")
        if not callable(candidate) or not callable(utility_evaluator):
            raise TypeError("candidate and utility_evaluator must be callable")

        engines, observed_version = self.snapshot()
        try:
            incumbent = engines[timescale]
        except KeyError:
            raise KeyError(f"unknown timescale: {timescale}") from None

        incumbent_utility = float(utility_evaluator(incumbent))
        candidate_utility = float(utility_evaluator(candidate))
        if not candidate_utility > incumbent_utility:
            return False, self.snapshot()[1]

        with self._lock:
            if self._version != observed_version:
                return False, self._version
            updated = self._engines.copy()
            updated[timescale] = candidate
            self._engines = updated
            self._version += 1
            return True, self._version


__all__ = ["Algorithm13EngineDispatcher"]
