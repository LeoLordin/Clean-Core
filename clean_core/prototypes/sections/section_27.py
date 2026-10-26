"""Bounded, explicit-distribution prototype for Section 27.

This scores actions from supplied beliefs; it does not infer or learn another
agent's mental state. Its prototype transform shrinks each belief toward a
uniform distribution by a configurable amount; it makes no calibration or
coverage guarantee.
"""

import numpy as np


def conformalize_belief(action_distribution, uniform_shrinkage=0.0):
    """Normalize a non-negative belief and optionally shrink it toward uniform.

    ``uniform_shrinkage`` is in [0, 1]: zero preserves the normalized input,
    and one returns a uniform distribution.
    """
    belief = np.asarray(action_distribution, dtype=np.float64)
    shrinkage = float(uniform_shrinkage)
    if (
        belief.ndim != 1
        or belief.size == 0
        or not np.all(np.isfinite(belief))
        or np.any(belief < 0.0)
        or not np.any(belief > 0.0)
    ):
        raise ValueError("action_distribution must be a non-empty finite non-negative vector with positive mass")
    if not np.isfinite(shrinkage) or not 0.0 <= shrinkage <= 1.0:
        raise ValueError("uniform_shrinkage must be finite and between 0 and 1")

    scaled = belief / np.max(belief)
    normalized = scaled / np.sum(scaled)
    uniform = np.full(belief.size, 1.0 / belief.size)
    result = (1.0 - shrinkage) * normalized + shrinkage * uniform
    return result


class ConformalTheoryOfMindPlanner:
    """Finite-action multi-agent planner driven by explicit action beliefs."""

    def __init__(self, uniform_shrinkage=0.0, score_bound=1e6):
        self.uniform_shrinkage = float(uniform_shrinkage)
        self.score_bound = float(score_bound)
        if not np.isfinite(self.uniform_shrinkage) or not 0.0 <= self.uniform_shrinkage <= 1.0:
            raise ValueError("uniform_shrinkage must be finite and between 0 and 1")
        if not np.isfinite(self.score_bound) or self.score_bound <= 0.0:
            raise ValueError("score_bound must be finite and positive")

    def score_actions(self, curiosity_information, other_action_beliefs, utility_tensor):
        """Return curiosity plus summed expected utility for each own action.

        ``utility_tensor`` has shape (own actions, other agents, other actions).
        Utilities and curiosity are clipped to ``[-score_bound, score_bound]``.
        """
        curiosity = np.asarray(curiosity_information, dtype=np.float64)
        utility = np.asarray(utility_tensor, dtype=np.float64)
        beliefs = [np.asarray(belief, dtype=np.float64) for belief in other_action_beliefs]
        if curiosity.ndim != 1 or curiosity.size == 0 or not np.all(np.isfinite(curiosity)):
            raise ValueError("curiosity_information must be a non-empty finite vector")
        if not beliefs:
            raise ValueError("at least one other-agent belief is required")
        if utility.ndim != 3 or utility.shape[:2] != (curiosity.size, len(beliefs)):
            raise ValueError("utility_tensor must have shape (own_actions, agents, other_actions)")
        if utility.shape[2] == 0 or not np.all(np.isfinite(utility)):
            raise ValueError("utility_tensor must have finite values and at least one other action")

        transformed_beliefs = []
        for belief in beliefs:
            if belief.ndim != 1 or belief.size != utility.shape[2]:
                raise ValueError("each belief must have one entry per other action")
            transformed_beliefs.append(
                conformalize_belief(belief, self.uniform_shrinkage)
            )

        bounded_curiosity = np.clip(curiosity, -self.score_bound, self.score_bound)
        bounded_utility = np.clip(utility, -self.score_bound, self.score_bound)
        scores = bounded_curiosity.copy()
        for agent_index, belief in enumerate(transformed_beliefs):
            scores += bounded_utility[:, agent_index, :] @ belief
        if not np.all(np.isfinite(scores)):
            raise ValueError("action scores are too large to represent")
        return scores

    def choose_action(self, curiosity_information, other_action_beliefs, utility_tensor):
        """Return the index of the highest-scoring own action and all scores."""
        scores = self.score_actions(curiosity_information, other_action_beliefs, utility_tensor)
        return int(np.argmax(scores)), scores


__all__ = ["ConformalTheoryOfMindPlanner", "conformalize_belief"]
