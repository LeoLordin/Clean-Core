"""Bounded stratified replay prototype for Section 20.

Each stratum owns an independent uniform reservoir maintained with Algorithm
R. This is not an implementation of Vitter's skip-based sampling algorithms.
Replay batches balance draws across strata that currently contain samples.
"""

import random
from collections.abc import Hashable, Iterable
from typing import TypeVar

T = TypeVar("T")
S = TypeVar("S", bound=Hashable)


class StratifiedReplayReservoir:
    """Keep a bounded uniform reservoir for each of a fixed set of strata.

    A caller-supplied :class:`random.Random` controls reservoir replacement and
    replay sampling, enabling deterministic runs when initialized with a seed.
    """

    def __init__(
        self,
        capacity: int,
        strata: Iterable[S],
        rng: random.Random | None = None,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")

        strata = tuple(strata)
        if not strata:
            raise ValueError("at least one stratum is required")
        if len(set(strata)) != len(strata):
            raise ValueError("strata must be unique")
        if capacity < len(strata):
            raise ValueError("capacity must provide at least one slot per stratum")
        if rng is not None and not isinstance(rng, random.Random):
            raise TypeError("rng must be an instance of random.Random")

        self.capacity = capacity
        self.strata = strata
        self.stratum_capacity = capacity // len(strata)
        self._rng = rng if rng is not None else random.Random()
        self._reservoirs: dict[S, list[T]] = {stratum: [] for stratum in strata}
        self._seen: dict[S, int] = {stratum: 0 for stratum in strata}

    @property
    def size(self) -> int:
        """Return the number of currently retained items."""
        return sum(map(len, self._reservoirs.values()))

    def stratum_size(self, stratum: S) -> int:
        """Return the number of retained items for ``stratum``."""
        return len(self._reservoirs[self._validate_stratum(stratum)])

    def seen_count(self, stratum: S) -> int:
        """Return the number of items ever added to ``stratum``."""
        return self._seen[self._validate_stratum(stratum)]

    def add(self, stratum: S, item: T) -> None:
        """Add an item using Algorithm R within its stratum.

        Once full, each item seen so far has the same probability of being
        retained; every later item replaces a random slot with probability
        ``stratum_capacity / seen_count``.
        """
        stratum = self._validate_stratum(stratum)
        reservoir = self._reservoirs[stratum]
        seen = self._seen[stratum] + 1
        self._seen[stratum] = seen

        if len(reservoir) < self.stratum_capacity:
            reservoir.append(item)
            return

        replacement_index = self._rng.randrange(seen)
        if replacement_index < self.stratum_capacity:
            reservoir[replacement_index] = item

    def sample_by_stratum(self, count: int = 1) -> dict[S, list[T]]:
        """Draw ``count`` replay items from each currently non-empty stratum.

        Sampling is with replacement so batches can exceed reservoir sizes.
        Empty strata are omitted until they receive an item.
        """
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("count must be a positive integer")

        return {
            stratum: [
                reservoir[self._rng.randrange(len(reservoir))]
                for _ in range(count)
            ]
            for stratum, reservoir in self._reservoirs.items()
            if reservoir
        }

    def sample(self, batch_size: int) -> list[T]:
        """Draw a batch with per-stratum sample counts differing by at most one.

        The batch is balanced across strata that currently contain items.
        Sampling is with replacement, allowing ``batch_size`` to exceed the
        total number of retained items.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")

        available_strata = [
            stratum for stratum, reservoir in self._reservoirs.items() if reservoir
        ]
        if not available_strata:
            raise ValueError("cannot sample from an empty reservoir")

        base_count, remainder = divmod(batch_size, len(available_strata))
        extra_strata = set(self._rng.sample(available_strata, remainder))
        batch: list[T] = []
        for stratum in available_strata:
            count = base_count + (stratum in extra_strata)
            reservoir = self._reservoirs[stratum]
            batch.extend(
                reservoir[self._rng.randrange(len(reservoir))]
                for _ in range(count)
            )
        self._rng.shuffle(batch)
        return batch

    def _validate_stratum(self, stratum: S) -> S:
        if stratum not in self._reservoirs:
            raise ValueError(f"unknown stratum: {stratum!r}")
        return stratum


__all__ = ["StratifiedReplayReservoir"]
