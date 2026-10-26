"""Finite-step implementation of Section 25's tail statistic and regulator.

This evaluates the stated equations for supplied errors and precision
matrices. It does not validate an SOC detector or establish unbiasedness.
"""

import numpy as np

from clean_core.config import FP64


class Section25SOCRegulator:
    """Track the precision-weighted tail log mean and bounded damping rate."""

    def __init__(
        self,
        epsilon_min=1e-3,
        gamma_soc=0.05,
        n_min=8,
        eta_alpha=0.01,
        alpha_damp=0.05,
        delta_soc=1e-6,
        delta_alpha_max=0.01,
    ):
        values = {
            "epsilon_min": epsilon_min,
            "gamma_soc": gamma_soc,
            "eta_alpha": eta_alpha,
            "alpha_damp": alpha_damp,
            "delta_soc": delta_soc,
            "delta_alpha_max": delta_alpha_max,
        }
        for name, value in values.items():
            values[name] = self._finite_scalar(name, value)
        if values["epsilon_min"] <= 0.0 or values["delta_soc"] <= 0.0:
            raise ValueError("epsilon_min and delta_soc must be positive")
        if not 0.0 <= values["gamma_soc"] <= 1.0:
            raise ValueError("gamma_soc must be in [0, 1]")
        if values["eta_alpha"] < 0.0 or values["delta_alpha_max"] < 0.0:
            raise ValueError("eta_alpha and delta_alpha_max must be non-negative")
        if values["alpha_damp"] < 0.05:
            raise ValueError("alpha_damp must be at least 0.05")
        if isinstance(n_min, (bool, np.bool_)) or not isinstance(
            n_min, (int, np.integer)
        ) or n_min < 1:
            raise ValueError("n_min must be a positive integer")

        self.epsilon_min = values["epsilon_min"]
        self.gamma_soc = values["gamma_soc"]
        self.n_min = int(n_min)
        self.eta_alpha = values["eta_alpha"]
        self.alpha_damp = values["alpha_damp"]
        self.delta_soc = values["delta_soc"]
        self.delta_alpha_max = values["delta_alpha_max"]
        self.log_tail_mean = 0.0
        self.alpha_measured = None
        self.tail_count = 0
        self.delta_alpha = 0.0

    @staticmethod
    def _finite_scalar(name, value):
        try:
            scalar = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite scalar") from exc
        if not np.isfinite(scalar):
            raise ValueError(f"{name} must be a finite scalar")
        return scalar

    def update(self, errors, precision_matrices, dt):
        """Apply one finite update and return the current damping coefficient."""
        dt = self._finite_scalar("dt", dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if len(errors) != len(precision_matrices):
            raise ValueError("errors and precision_matrices must have equal lengths")

        tail_log_ratios = []
        for error, precision in zip(errors, precision_matrices):
            error = np.asarray(error, dtype=FP64)
            precision = np.asarray(precision, dtype=FP64)
            if error.ndim != 1 or precision.shape != (error.size, error.size):
                raise ValueError("each precision matrix must match its error vector")
            if not np.all(np.isfinite(error)) or not np.all(np.isfinite(precision)):
                raise ValueError("errors and precision matrices must be finite")

            with np.errstate(over="ignore", invalid="ignore"):
                norm_sq = float(error @ precision @ error)
            if not np.isfinite(norm_sq):
                raise ValueError("precision-weighted squared error must be finite")
            if norm_sq < -1e-10:
                raise ValueError("precision-weighted squared error must be non-negative")
            norm = float(np.sqrt(max(0.0, norm_sq)))
            if norm >= self.epsilon_min:
                # Subtracting logs avoids overflow in norm / epsilon_min.
                tail_log_ratios.append(
                    float(np.log(norm) - np.log(self.epsilon_min))
                )

        self.tail_count = len(tail_log_ratios)
        if self.tail_count >= self.n_min:
            observed_log_tail_mean = float(np.mean(tail_log_ratios))
            next_log_tail_mean = (
                (1.0 - self.gamma_soc) * self.log_tail_mean
                + self.gamma_soc * observed_log_tail_mean
            )
            if not np.isfinite(next_log_tail_mean):
                raise ValueError("updated tail statistic must be finite")
            self.log_tail_mean = next_log_tail_mean

        alpha_measured = float(
            1.0 + 1.0 / max(self.delta_soc, self.log_tail_mean)
        )
        if not np.isfinite(alpha_measured):
            raise ValueError("measured alpha must be finite")
        self.alpha_measured = alpha_measured

        self.delta_alpha = 0.0
        if self.tail_count >= self.n_min:
            with np.errstate(over="ignore", invalid="ignore"):
                raw_delta = (
                    dt * (1.2 * self.eta_alpha) * (1.5 - self.alpha_measured)
                )
            self.delta_alpha = float(
                np.clip(raw_delta, -self.delta_alpha_max, self.delta_alpha_max)
            )
            next_alpha_damp = max(0.05, self.alpha_damp + self.delta_alpha)
            if not np.isfinite(next_alpha_damp):
                raise ValueError("updated alpha_damp must be finite")
            self.alpha_damp = float(next_alpha_damp)

        return self.alpha_damp


__all__ = ["Section25SOCRegulator"]
