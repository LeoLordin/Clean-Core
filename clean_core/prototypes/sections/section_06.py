"""Finite prototype for the polyhedral decider and safe-horizon formulas.

The implementation operates on caller-supplied finite prefix-code models. It
does not implement universal search, a Solomonoff mixture, or classical AIXI.
"""

import numpy as np

from clean_core.cognition.polyhedral_aixi import PolyhedralAIXIPrototype
from clean_core.config import FP64


def decision_hyperplane(q_values, action_a, action_b):
    """Return coefficients ``c`` for the boundary ``p @ c == 0``.

    ``q_values`` contains one value per action and observation. The boundary
    is meaningful on the observation-probability simplex.
    """
    values = np.asarray(q_values, dtype=FP64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("q_values must have shape (at least 2 actions, at least 1 observation)")
    if not np.all(np.isfinite(values)):
        raise ValueError("q_values must contain only finite values")

    for action in (action_a, action_b):
        if not isinstance(action, (int, np.integer)) or isinstance(action, (bool, np.bool_)):
            raise ValueError("action indices must be integers")
        if action < 0 or action >= values.shape[0]:
            raise ValueError("action index is out of range")
    if action_a == action_b:
        raise ValueError("action indices must be distinct")
    return values[action_a] - values[action_b]


class Section06PolyhedralDecider(PolyhedralAIXIPrototype):
    """Bounded finite-model approximation of Section 6's decider.

    Inherited results are conditional on the supplied prefix codes, Q tables,
    and return bound; they are not universal-search or AIXI certificates.
    """

    decision_hyperplane = staticmethod(decision_hyperplane)


__all__ = ["Section06PolyhedralDecider", "decision_hyperplane"]
