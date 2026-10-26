from dataclasses import dataclass

import numpy as np

from clean_core.config import FP64


@dataclass(frozen=True)
class ProgramHypothesis:
    code: str
    q_values: np.ndarray


class PolyhedralAIXIPrototype:
    """Anytime bounds over a finite, caller-supplied prefix-code model stream.

    This is not universal program execution and cannot certify equivalence to
    classical AIXI. Each Q table contains discounted returns per action/outcome.
    """

    def __init__(self, num_actions, num_observations, r_max=1.0, discount=0.99):
        if num_actions < 2 or num_observations < 1:
            raise ValueError("need at least two actions and one observation")
        if not np.isfinite(r_max) or r_max <= 0.0:
            raise ValueError("r_max must be finite and positive")
        if not np.isfinite(discount) or not 0.0 <= discount < 1.0:
            raise ValueError("discount must be in [0, 1)")
        self.num_actions = int(num_actions)
        self.num_observations = int(num_observations)
        self.max_return = float(r_max / (1.0 - discount))
        self.observation_probabilities = np.full(
            self.num_observations, 1.0 / self.num_observations, dtype=FP64
        )
        self._pending = []
        self._evaluated = []
        self._kraft_mass = 0.0

    def set_observation_probabilities(self, probabilities):
        probabilities = np.asarray(probabilities, dtype=FP64)
        if probabilities.shape != (self.num_observations,) or not np.all(np.isfinite(probabilities)):
            raise ValueError("probabilities must be a finite vector matching observations")
        if np.any(probabilities < 0.0) or np.sum(probabilities) <= 0.0:
            raise ValueError("probabilities must be non-negative with positive mass")
        self.observation_probabilities = probabilities / np.sum(probabilities)

    def add_candidate(self, code, q_values):
        if not isinstance(code, str) or not code or any(bit not in "01" for bit in code):
            raise ValueError("code must be a non-empty binary prefix code")
        if any(code.startswith(item.code) or item.code.startswith(code) for item in self._pending + self._evaluated):
            raise ValueError("candidate codes must be prefix-free")
        q_values = np.asarray(q_values, dtype=FP64)
        expected_shape = (self.num_actions, self.num_observations)
        if q_values.shape != expected_shape or not np.all(np.isfinite(q_values)):
            raise ValueError(f"q_values must be a finite matrix with shape {expected_shape}")
        if np.any(q_values < 0.0) or np.any(q_values > self.max_return):
            raise ValueError("q_values must lie within the configured discounted-return bounds")
        weight = 2.0 ** (-len(code))
        if self._kraft_mass + weight > 1.0 + 1e-12:
            raise ValueError("candidate codes violate the Kraft mass bound")
        hypothesis = ProgramHypothesis(code, q_values.copy())
        self._pending.append(hypothesis)
        self._pending.sort(key=lambda item: (len(item.code), item.code))
        self._kraft_mass += weight

    def advance(self, budget=1, delta_eps=1e-12):
        if not isinstance(budget, int) or budget < 1:
            raise ValueError("budget must be a positive integer")
        if not np.isfinite(delta_eps) or delta_eps <= 0.0:
            raise ValueError("delta_eps must be finite and positive")
        count = min(budget, len(self._pending))
        self._evaluated.extend(self._pending[:count])
        del self._pending[:count]
        return self.decision(delta_eps=delta_eps)

    def decision(self, delta_eps=1e-12):
        if not self._evaluated:
            raise RuntimeError("no candidate model has been evaluated")
        lower_by_outcome = np.zeros((self.num_actions, self.num_observations), dtype=FP64)
        for program in self._evaluated:
            lower_by_outcome += 2.0 ** (-len(program.code)) * program.q_values
        unresolved_mass = max(0.0, 1.0 - sum(2.0 ** (-len(item.code)) for item in self._evaluated))
        upper_by_outcome = lower_by_outcome + unresolved_mass * self.max_return
        q_min = lower_by_outcome @ self.observation_probabilities
        q_max = q_min + unresolved_mass * self.max_return
        best_action = int(np.argmax(q_min))
        competing_upper = np.delete(q_max, best_action)
        separation = float(q_min[best_action] - np.max(competing_upper))
        bounds_separate = separation > 0.0

        action_ids = np.arange(self.num_actions) != best_action
        per_outcome_competitor = np.max(upper_by_outcome[action_ids], axis=0)
        outcome_gaps = lower_by_outcome[best_action] - per_outcome_competitor
        denominator = float(np.linalg.norm(outcome_gaps))
        safe_radius = separation / max(denominator, delta_eps) if bounds_separate else 0.0
        motor_pressure = np.zeros(self.num_actions, dtype=FP64)
        if bounds_separate:
            for action in range(self.num_actions):
                if action != best_action:
                    margin = q_min[best_action] - q_max[action]
                    motor_pressure[best_action] += 1.0 / (margin + delta_eps)
                    motor_pressure[action] -= 1.0 / (margin + delta_eps)
            norm = float(np.linalg.norm(motor_pressure))
            if norm > 0.0:
                motor_pressure *= safe_radius / norm

        return {
            "action": best_action,
            "q_min": q_min.copy(),
            "q_max": q_max.copy(),
            "unresolved_mass": unresolved_mass,
            "separation": separation,
            "safe_radius": float(safe_radius),
            "motor_pressure": motor_pressure,
            "bounds_separate": bool(bounds_separate),
            "evaluated_programs": len(self._evaluated),
            "pending_programs": len(self._pending),
        }