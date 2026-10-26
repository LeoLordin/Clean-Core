"""Composed criticality and SoftVQ codebook dynamics for Algorithm 8.

The step combines bounded tail-statistic damping updates with usage decay,
pruning, and an optional novelty-gated code spawn. It is a prototype and does
not establish self-organized criticality.
"""

import operator

import numpy as np

from clean_core.cognition.criticality import SOCCriticalityRegulator
from clean_core.config import K_MAX
from clean_core.prototypes.sections.section_21 import SoftVQCodebookEvolution


def _finite_scalar(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite scalar")
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite scalar") from error
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


class Algorithm08Dynamics:
    """Maintain a bounded criticality regulator and SoftVQ codebook."""

    def __init__(
        self,
        centroids,
        usage=None,
        *,
        max_size=K_MAX,
        protected_indices=(0,),
        active=None,
        seed=None,
        epsilon_min=1e-3,
        gamma_soc=0.05,
        n_min=8,
        eta_alpha=0.01,
        alpha_damp=0.05,
        alpha_min=0.05,
        delta_soc=1e-6,
        delta_alpha_max=0.01,
        target_alpha=1.5,
    ):
        epsilon_min = _finite_scalar(epsilon_min, "epsilon_min")
        gamma_soc = _finite_scalar(gamma_soc, "gamma_soc")
        eta_alpha = _finite_scalar(eta_alpha, "eta_alpha")
        alpha_damp = _finite_scalar(alpha_damp, "alpha_damp")
        alpha_min = _finite_scalar(alpha_min, "alpha_min")
        delta_soc = _finite_scalar(delta_soc, "delta_soc")
        delta_alpha_max = _finite_scalar(delta_alpha_max, "delta_alpha_max")
        target_alpha = _finite_scalar(target_alpha, "target_alpha")
        if isinstance(n_min, (bool, np.bool_)):
            raise ValueError("n_min must be a positive integer")
        try:
            n_min = operator.index(n_min)
        except TypeError as error:
            raise ValueError("n_min must be a positive integer") from error
        if epsilon_min <= 0.0 or delta_soc <= 0.0 or n_min < 1:
            raise ValueError("epsilon_min, delta_soc, and n_min must be positive")
        if not 0.0 <= gamma_soc <= 1.0:
            raise ValueError("gamma_soc must be between zero and one")
        if eta_alpha < 0.0 or delta_alpha_max < 0.0:
            raise ValueError("eta_alpha and delta_alpha_max must be non-negative")
        if alpha_min < 0.0 or alpha_damp < alpha_min:
            raise ValueError("alpha_damp must be at least non-negative alpha_min")

        self.criticality = SOCCriticalityRegulator(
            epsilon_min=epsilon_min,
            gamma_soc=gamma_soc,
            n_min=n_min,
            eta_alpha=eta_alpha,
            alpha_damp=alpha_damp,
            alpha_min=alpha_min,
            delta_soc=delta_soc,
            delta_alpha_max=delta_alpha_max,
            target_alpha=target_alpha,
        )
        self.codebook = SoftVQCodebookEvolution(
            centroids,
            usage,
            max_size=max_size,
            protected_indices=protected_indices,
            active=active,
            seed=seed,
        )
        self.delta_alpha_max = delta_alpha_max

    def step(
        self,
        errors,
        precision_matrices,
        dt,
        assignments=(),
        *,
        gamma_usage=1.0,
        theta_prune=0.05,
        spawn_candidate=None,
        theta_spawn=15.0,
        d_min=0.1,
        noise_scale=0.0,
    ):
        """Update damping and codebook once; return a snapshot of the results.

        ``spawn_candidate``, when supplied, is a
        ``(mu_point, prediction_error, precision)`` tuple for the Section 21
        novelty/divergence gates. The criticality tail statistic is computed
        from the separate ``errors`` and ``precision_matrices`` collections.
        """
        previous_alpha = self.criticality.alpha_damp
        alpha_damp = self.criticality.update(errors, precision_matrices, dt)
        if abs(alpha_damp - previous_alpha) > self.delta_alpha_max + 1e-12:
            raise RuntimeError("criticality update exceeded delta_alpha_max")

        usage = self.codebook.update_usage(assignments, gamma_usage=gamma_usage)
        pruned_indices = self.codebook.prune(theta_prune=theta_prune)

        spawned = False
        if spawn_candidate is not None:
            try:
                mu_point, prediction_error, precision = spawn_candidate
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "spawn_candidate must contain a point, error, and precision"
                ) from error
            spawned = self.codebook.maybe_spawn(
                mu_point,
                prediction_error,
                precision,
                theta_spawn=theta_spawn,
                d_min=d_min,
                noise_scale=noise_scale,
            )

        return {
            "alpha_damp": float(self.criticality.alpha_damp),
            "alpha_measured": self.criticality.alpha_measured,
            "log_tail_mean": float(self.criticality.log_tail_mean),
            "usage": self.codebook.usage.copy(),
            "active": self.codebook.active.copy(),
            "pruned_indices": pruned_indices,
            "spawned": spawned,
        }


__all__ = ["Algorithm08Dynamics"]
