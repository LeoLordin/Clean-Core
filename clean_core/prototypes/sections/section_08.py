"""Finite Section 8 prototype for a projected PHR inequality step.

This covers one inequality ``constraint_value <= limit`` at a time and a
box-projected Euclidean state step. It does not implement or prove the full
master action functional.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def projected_phr_step(
    state: ArrayLike,
    objective_value: float,
    objective_gradient: ArrayLike,
    constraint_value: float,
    limit: float,
    constraint_gradient: ArrayLike,
    multiplier: float,
    penalty: float,
    step_size: float,
    lower_bound: ArrayLike,
    upper_bound: ArrayLike,
) -> dict[str, float | NDArray[np.float64]]:
    """Update one non-negative multiplier and take a box-projected PHR step.

    With ``g = constraint_value - limit``, the multiplier becomes
    ``max(0, multiplier + penalty * g)``. The returned objective is the supplied
    objective plus the shifted PHR term, and the state gradient adds the updated
    multiplier times the inequality gradient.
    """
    x = _finite_vector(state, "state")
    objective_grad = _matching_vector(objective_gradient, x.shape, "objective_gradient")
    constraint_grad = _matching_vector(constraint_gradient, x.shape, "constraint_gradient")
    lower = _matching_vector(lower_bound, x.shape, "lower_bound")
    upper = _matching_vector(upper_bound, x.shape, "upper_bound")
    objective = _finite_scalar(objective_value, "objective_value")
    value = _finite_scalar(constraint_value, "constraint_value")
    bound = _finite_scalar(limit, "limit")
    dual = _finite_scalar(multiplier, "multiplier")
    mu = _finite_scalar(penalty, "penalty")
    step = _finite_scalar(step_size, "step_size")

    if dual < 0.0 or mu <= 0.0 or step < 0.0:
        raise ValueError("multiplier and step_size must be non-negative; penalty must be positive")
    if np.any(lower > upper):
        raise ValueError("lower_bound must not exceed upper_bound")

    violation = value - bound
    updated_multiplier = max(0.0, dual + mu * violation)
    shifted_violation = max(0.0, violation + dual / mu)
    phr_term = 0.5 * mu * shifted_violation**2 - dual**2 / (2.0 * mu)
    augmented_gradient = objective_grad + updated_multiplier * constraint_grad
    updated_state = np.clip(x - step * augmented_gradient, lower, upper)

    return {
        "state": updated_state,
        "multiplier": updated_multiplier,
        "objective": objective + phr_term,
        "gradient": augmented_gradient,
        "violation": violation,
    }


def _finite_scalar(value: float, name: str) -> float:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 0 or not np.isfinite(array):
        raise ValueError(f"{name} must be a finite scalar")
    return float(array)


def _finite_vector(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return array


def _matching_vector(
    value: ArrayLike, shape: tuple[int, ...], name: str
) -> NDArray[np.float64]:
    array = _finite_vector(value, name)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return array
