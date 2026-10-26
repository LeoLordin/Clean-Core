"""Numerical diagnostics for Section 24's Lyapunov stability formulation.

The candidate is non-negative relative to the supplied references, and the
decay check compares sampled finite-step change with the stated rate bound.
Neither helper proves a stability theorem or supplies its required assumptions.
"""

import numpy as np

from clean_core.cognition.stability import (
    lyapunov_candidate as _lyapunov_candidate,
)


def lyapunov_candidate(states, barycenter, actions=None, target_actions=None):
    """Evaluate Section 24's non-negative candidate using the stability API."""
    return _lyapunov_candidate(states, barycenter, actions, target_actions)


def finite_step_decay_diagnostic(
    value_before,
    value_after,
    dt,
    alpha_damp,
    sigma_noise=0.0,
    tolerance=0.0,
):
    """Compare measured finite-step decay with the stated differential bound.

    The returned pass flag is only a sampled-rate diagnostic; it is not a
    proof that the continuous-time inequality or a stability theorem holds.
    """
    values = {
        "value_before": value_before,
        "value_after": value_after,
        "dt": dt,
        "alpha_damp": alpha_damp,
        "sigma_noise": sigma_noise,
        "tolerance": tolerance,
    }
    scalars = {}
    for name, value in values.items():
        try:
            scalar = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite scalar") from exc
        if not np.isfinite(scalar):
            raise ValueError(f"{name} must be a finite scalar")
        scalars[name] = scalar

    if scalars["value_before"] < 0.0 or scalars["value_after"] < 0.0:
        raise ValueError("Lyapunov candidate values must be non-negative")
    if scalars["dt"] <= 0.0:
        raise ValueError("dt must be positive")
    if scalars["alpha_damp"] < 0.0:
        raise ValueError("alpha_damp must be non-negative")
    if scalars["sigma_noise"] < 0.0:
        raise ValueError("sigma_noise must be non-negative")
    if scalars["tolerance"] < 0.0:
        raise ValueError("tolerance must be non-negative")

    observed_rate = (
        scalars["value_after"] - scalars["value_before"]
    ) / scalars["dt"]
    allowed_rate = (
        -scalars["alpha_damp"] * scalars["value_before"]
        + scalars["sigma_noise"] ** 2
    )
    if not np.isfinite(observed_rate) or not np.isfinite(allowed_rate):
        raise ValueError("computed decay rates must be finite")

    return {
        "observed_rate": float(observed_rate),
        "allowed_rate": float(allowed_rate),
        "rate_residual": float(observed_rate - allowed_rate),
        "within_bound": bool(observed_rate <= allowed_rate + scalars["tolerance"]),
    }


__all__ = ["finite_step_decay_diagnostic", "lyapunov_candidate"]
