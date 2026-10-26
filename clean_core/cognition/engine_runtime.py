from threading import Lock


class AtomicEngineSlot:
    """Versioned thread-safe engine publication with evaluation outside the lock."""

    def __init__(self, engine):
        if not callable(engine):
            raise TypeError("engine must be callable")
        self._engine = engine
        self._version = 0
        self._lock = Lock()

    def snapshot(self):
        with self._lock:
            return self._engine, self._version

    def dispatch(self, *args, **kwargs):
        engine, _ = self.snapshot()
        return engine(*args, **kwargs)

    def swap_if_better(self, candidate, utility_evaluator):
        if not callable(candidate) or not callable(utility_evaluator):
            raise TypeError("candidate and utility_evaluator must be callable")
        current, observed_version = self.snapshot()
        current_utility = float(utility_evaluator(current))
        candidate_utility = float(utility_evaluator(candidate))
        if not (candidate_utility > current_utility):
            return False, self.snapshot()[1]
        with self._lock:
            if self._version != observed_version:
                return False, self._version
            self._engine = candidate
            self._version += 1
            return True, self._version


class MultiTimescaleDispatcher:
    """Copy-on-write engine map with synchronized snapshot reads."""

    def __init__(self, engines):
        if not engines or any(not isinstance(name, str) or not name for name in engines):
            raise ValueError("engines must be a non-empty mapping with named entries")
        if any(not callable(engine) for engine in engines.values()):
            raise TypeError("every engine must be callable")
        self._engines = dict(engines)
        self._version = 0
        self._lock = Lock()

    def dispatch(self, timescale, *args, **kwargs):
        with self._lock:
            engine = self._engines.get(timescale)
        if engine is None:
            raise KeyError(f"unknown timescale: {timescale}")
        return engine(*args, **kwargs)

    def publish(self, timescale, engine):
        if not isinstance(timescale, str) or not timescale or not callable(engine):
            raise ValueError("timescale must be named and engine callable")
        with self._lock:
            updated = self._engines.copy()
            updated[timescale] = engine
            self._engines = updated
            self._version += 1
            return self._version

    def snapshot(self):
        with self._lock:
            return self._engines.copy(), self._version