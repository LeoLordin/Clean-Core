"""Testable callable-publication prototype for Section 35.

This reuses the runtime's version-checked utility comparison and snapshot
dispatch. It is not a proof checker or JIT, and does not guarantee zero
downtime or bounded execution jitter.
"""

from clean_core.cognition.engine_runtime import AtomicEngineSlot


class Section35EngineSlot:
    """Versioned concurrent callable slot with utility-gated publication."""

    def __init__(self, engine):
        self._slot = AtomicEngineSlot(engine)

    def snapshot(self):
        """Return the currently published callable and its version."""
        return self._slot.snapshot()

    def dispatch(self, *args, **kwargs):
        """Run the callable captured by one synchronized snapshot."""
        return self._slot.dispatch(*args, **kwargs)

    def swap_if_better(self, candidate, utility_evaluator):
        """Publish a candidate only if its measured utility is higher.

        Evaluation occurs outside the publication lock. If another swap
        publishes after this attempt's snapshot, this attempt is rejected.
        """
        return self._slot.swap_if_better(candidate, utility_evaluator)


__all__ = ["Section35EngineSlot"]
