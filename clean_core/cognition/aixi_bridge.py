import numpy as np

from clean_core.config import D_ACT, FP64


class BoundedActionHeuristic:
    """Selects an action from bounded observation scores; it is not an AIXI solver."""

    def __init__(self, num_actions=D_ACT):
        if num_actions < 2:
            raise ValueError("num_actions must be at least 2")
        self.num_actions = num_actions

    def select_action(self, obs_vector):
        observations = np.asarray(obs_vector, dtype=FP64)
        if observations.ndim != 1 or observations.size == 0:
            raise ValueError("obs_vector must be a non-empty one-dimensional vector")
        if not np.all(np.isfinite(observations)):
            raise ValueError("obs_vector must contain only finite values")

        scores = 0.5 + 0.4 * np.tanh(observations[np.arange(self.num_actions) % observations.size])
        best_action = int(np.argmax(scores))
        competitor_scores = np.delete(scores, best_action)
        score_margin = float(scores[best_action] - np.max(competitor_scores))

        motor_pressure = np.zeros(self.num_actions, dtype=FP64)
        if score_margin > 0.0:
            for action in range(self.num_actions):
                if action != best_action:
                    weight = 1.0 / (scores[best_action] - scores[action])
                    motor_pressure[best_action] += weight
                    motor_pressure[action] -= weight
            norm = np.linalg.norm(motor_pressure)
            if norm > 0.0:
                motor_pressure *= score_margin / norm

        return best_action, score_margin, motor_pressure

    def solve_polyhedral_aixi(self, obs_vector, r_max=None, gamma=None):
        """Compatibility wrapper; the returned margin is heuristic, not certified."""
        return self.select_action(obs_vector)


# Keep existing imports working while making the implementation's limits explicit.
SolomonoffAIXIBridge = BoundedActionHeuristic