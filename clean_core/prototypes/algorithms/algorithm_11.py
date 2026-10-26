"""Finite supplied-model implementation of Algorithm 11.

This evaluates caller-supplied prefix-code Q tables in bounded batches. It
does not run universal programs, perform Levin search, or certify AIXI.
"""

from clean_core.prototypes.sections.section_34 import Section34ExplicitProjection


class Algorithm11AnytimeDecider(Section34ExplicitProjection):
    """Anytime Q-bound decider over a finite caller-supplied model set.

    Each call to :meth:`iterate` evaluates at most ``budget`` pending models.
    A returned ``fixed_action`` is set only when its finite-prefix lower bound
    strictly exceeds every competing upper bound; this is not an AIXI
    certificate.
    """

    def iterate(self, budget=1, delta_eps=1e-12):
        """Advance the finite evaluation by ``budget`` models and report bounds."""
        result = self.advance(budget=budget, delta_eps=delta_eps)
        separated = result["bounds_separate"]
        result["fixed_action"] = result["action"] if separated else None
        result["epsilon_safe"] = result["separation"] if separated else 0.0
        return result


__all__ = ["Algorithm11AnytimeDecider"]
