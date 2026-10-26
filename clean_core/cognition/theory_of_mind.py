import numpy as np

from clean_core.config import FP64


class TheoryOfMindPlanner:
    """Finite discrete planner using explicit beliefs over other-agent actions."""

    def score_actions(self, curiosity_information, other_action_beliefs, utility_tensor):
        curiosity = np.asarray(curiosity_information, dtype=FP64)
        utility = np.asarray(utility_tensor, dtype=FP64)
        beliefs = [np.asarray(belief, dtype=FP64) for belief in other_action_beliefs]
        if curiosity.ndim != 1 or curiosity.size == 0 or not np.all(np.isfinite(curiosity)):
            raise ValueError("curiosity_information must be a non-empty finite vector")
        if not beliefs:
            raise ValueError("at least one other-agent belief is required")
        if utility.ndim != 3 or utility.shape[0] != curiosity.size or utility.shape[1] != len(beliefs):
            raise ValueError("utility_tensor must have shape (own_actions, agents, other_actions)")
        if not np.all(np.isfinite(utility)):
            raise ValueError("utility_tensor must contain only finite values")
        if any(
            belief.ndim != 1
            or belief.size != utility.shape[2]
            or not np.all(np.isfinite(belief))
            or np.any(belief < 0.0)
            or np.sum(belief) <= 0.0
            for belief in beliefs
        ):
            raise ValueError("each belief must be a finite non-negative action distribution")

        normalized_beliefs = [belief / np.sum(belief) for belief in beliefs]
        expected_social_utility = np.zeros(curiosity.size, dtype=FP64)
        for agent_index, belief in enumerate(normalized_beliefs):
            expected_social_utility += utility[:, agent_index, :] @ belief
        return curiosity + expected_social_utility

    def choose_action(self, curiosity_information, other_action_beliefs, utility_tensor):
        scores = self.score_actions(curiosity_information, other_action_beliefs, utility_tensor)
        return int(np.argmax(scores)), scores