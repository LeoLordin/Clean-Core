"""Finite candidate-prefix prototype for Section 34.

The bounds and pressure are computed only over caller-supplied models and
their prefix-code weights. This is not universal search and does not establish
universal AIXI optimality.
"""

import numpy as np

from clean_core.config import FP64
from clean_core.cognition.polyhedral_aixi import PolyhedralAIXIPrototype


class Section34ExplicitProjection(PolyhedralAIXIPrototype):
    """Kraft-weighted Q bounds and pressure for a finite candidate prefix.

    ``m_k`` is the finite-sum normalizer in Section 34's lower-bound formula.
    The unresolved Kraft mass is bounded using ``r_max / (1 - discount)``.
    Pressure is emitted only when the lower and upper action bounds strictly
    separate; its Euclidean norm is then ``epsilon_safe``.
    """

    def __init__(
        self,
        num_actions,
        num_observations,
        r_max=1.0,
        discount=0.99,
        *,
        m_k=1.0,
        epsilon_safe=1.0,
    ):
        super().__init__(
            num_actions,
            num_observations,
            r_max=r_max,
            discount=discount,
        )
        self.m_k = self._positive_finite(m_k, "m_k")
        self.epsilon_safe = self._positive_finite(epsilon_safe, "epsilon_safe")

    @staticmethod
    def _positive_finite(value, name):
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be finite and positive") from error
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return value

    def decision(self, delta_eps=1e-12):
        """Return the finite-prefix bounds, strict separation, and pressure."""
        delta_eps = self._positive_finite(delta_eps, "delta_eps")
        if not self._evaluated:
            raise RuntimeError("no candidate model has been evaluated")

        lower_by_outcome = np.zeros(
            (self.num_actions, self.num_observations), dtype=FP64
        )
        for program in self._evaluated:
            lower_by_outcome += 2.0 ** (-len(program.code)) * program.q_values
        lower_by_outcome /= self.m_k

        unresolved_mass = max(
            0.0,
            1.0
            - sum(2.0 ** (-len(program.code)) for program in self._evaluated),
        )
        q_min = lower_by_outcome @ self.observation_probabilities
        q_max = q_min + unresolved_mass * self.max_return
        best_action = int(np.argmax(q_min))
        competing_upper = np.delete(q_max, best_action)
        separation = float(q_min[best_action] - np.max(competing_upper))
        bounds_separate = separation > 0.0

        motor_pressure = np.zeros(self.num_actions, dtype=FP64)
        if bounds_separate:
            competitors = np.arange(self.num_actions) != best_action
            margins = q_min[best_action] - q_max[competitors]

            # Scale reciprocal margins before summing to avoid overflow while
            # preserving the direction of the normalized pressure vector.
            denominators = margins + delta_eps
            reciprocal_weights = np.min(denominators) / denominators
            motor_pressure[best_action] = np.sum(reciprocal_weights)
            motor_pressure[competitors] = -reciprocal_weights
            norm = float(np.linalg.norm(motor_pressure))
            if not np.isfinite(norm) or norm == 0.0:
                raise ValueError("pressure vector cannot be represented")
            motor_pressure *= self.epsilon_safe / norm

        if not (
            np.all(np.isfinite(q_min))
            and np.all(np.isfinite(q_max))
            and np.all(np.isfinite(motor_pressure))
        ):
            raise ValueError("Q bounds or pressure are too large to represent")

        return {
            "action": best_action,
            "q_min": q_min.copy(),
            "q_max": q_max.copy(),
            "unresolved_mass": float(unresolved_mass),
            "separation": separation,
            "motor_pressure": motor_pressure,
            "bounds_separate": bool(bounds_separate),
            "evaluated_programs": len(self._evaluated),
            "pending_programs": len(self._pending),
        }


__all__ = ["Section34ExplicitProjection"]
