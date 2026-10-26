"""Lock-based copy-on-write dispatcher prototype for Section 37.

The runtime synchronizes engine-map snapshots and publication with a lock.
Dispatch evaluates the captured engine after releasing that lock, so this
prototype is thread-safe but is not lock-free RCU.
"""

from types import MappingProxyType

from clean_core.cognition.engine_runtime import MultiTimescaleDispatcher


TIMESCALE_SECONDS = MappingProxyType(
    {
        "epsilon_0": 1e-3,
        "epsilon_1": 1.0,
        "epsilon_2": 1e5,
    }
)


class Section37MultiTimescaleDispatcher(MultiTimescaleDispatcher):
    """Copy-on-write engine dispatcher for Section 37's three time scales.

    Each dispatch reads one engine under the runtime's lock and invokes it
    outside the lock. Publishing creates and swaps a copied engine map under
    that lock. Individual publications are atomic; publishing multiple scales
    requires one publication per scale and is not a transaction.
    """

    def __init__(self, engines):
        if set(engines) != set(TIMESCALE_SECONDS):
            raise ValueError(
                "engines must define exactly epsilon_0, epsilon_1, and epsilon_2"
            )
        super().__init__(engines)

    def publish(self, timescale, engine):
        if timescale not in TIMESCALE_SECONDS:
            raise ValueError(f"unknown timescale: {timescale}")
        return super().publish(timescale, engine)


__all__ = ["Section37MultiTimescaleDispatcher", "TIMESCALE_SECONDS"]
