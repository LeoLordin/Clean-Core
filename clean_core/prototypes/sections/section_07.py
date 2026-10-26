"""Bounded prototype for Section 7's microstate snapshot and motor loop."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from clean_core.interfaces.motor_efe import categorical_target_kl


FloatVector = NDArray[np.float64]
StatePredictor = Callable[[FloatVector], ArrayLike]
CausalInformation = Callable[[FloatVector], float]


def _finite_array(value: ArrayLike, name: str, *, dimensions: tuple[int, ...]) -> NDArray[np.float64]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if array.ndim not in dimensions or array.size == 0:
        expected = " or ".join(str(dimension) for dimension in dimensions)
        raise ValueError(f"{name} must be a non-empty array with {expected} dimensions")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.array(array, dtype=np.float64, copy=True)


def _vector(value: ArrayLike, name: str) -> FloatVector:
    return _finite_array(value, name, dimensions=(1,))


def _action(value: ArrayLike, name: str) -> FloatVector:
    action = _vector(value, name)
    if np.any(action < -1.0) or np.any(action > 1.0):
        raise ValueError(f"{name} values must be in [-1, 1]")
    return action


def _scalar(value: float, name: str, *, minimum: float | None = None) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(scalar) or (minimum is not None and scalar < minimum):
        if minimum is None:
            raise ValueError(f"{name} must be finite")
        raise ValueError(f"{name} must be finite and greater than or equal to {minimum}")
    return scalar


@dataclass(frozen=True)
class FullMicrostateSupervector:
    """Validated snapshot of the six groups named by the Section 7 formula."""

    mu: FloatVector
    p: FloatVector
    theta: Mapping[str, NDArray[np.float64]]
    pi: NDArray[np.float64]
    e_ji: NDArray[np.float64]
    a_scm: NDArray[np.float64]


def make_full_microstate_supervector(
    mu: ArrayLike,
    p: ArrayLike,
    theta: Mapping[str, ArrayLike],
    pi: ArrayLike,
    e_ji: ArrayLike,
    a_scm: ArrayLike,
) -> FullMicrostateSupervector:
    """Create a finite, shape-checked snapshot; arrays are copied and read-only."""
    if not isinstance(theta, Mapping):
        raise ValueError("theta must be a mapping with W1, W2, b1, and b2")
    required = {"W1", "W2", "b1", "b2"}
    if set(theta) != required:
        raise ValueError("theta must contain exactly W1, W2, b1, and b2")

    parameters = {
        name: _finite_array(theta[name], f"theta[{name!r}]", dimensions=(1, 2))
        for name in ("W1", "W2", "b1", "b2")
    }
    w1, w2 = parameters["W1"], parameters["W2"]
    b1, b2 = parameters["b1"], parameters["b2"]
    if w1.ndim != 2 or w2.ndim != 2 or b1.shape != (w1.shape[0],) or b2.shape != (w2.shape[0],):
        raise ValueError("theta requires W1 and W2 matrices with matching b1 and b2 vectors")
    if w2.shape[1] != w1.shape[0]:
        raise ValueError("theta W2 input dimension must match W1 output dimension")

    arrays = {
        "mu": _finite_array(mu, "mu", dimensions=(1,)),
        "p": _finite_array(p, "p", dimensions=(1,)),
        "pi": _finite_array(pi, "pi", dimensions=(1, 2)),
        "e_ji": _finite_array(e_ji, "e_ji", dimensions=(1, 2)),
        "a_scm": _finite_array(a_scm, "a_scm", dimensions=(1, 2)),
    }
    for array in (*arrays.values(), *parameters.values()):
        array.setflags(write=False)
    return FullMicrostateSupervector(
        mu=arrays["mu"],
        p=arrays["p"],
        theta=MappingProxyType(parameters),
        pi=arrays["pi"],
        e_ji=arrays["e_ji"],
        a_scm=arrays["a_scm"],
    )


def motor_target_action(
    geodesic_velocity: ArrayLike,
    polyhedral_pressure: ArrayLike,
    social_action: ArrayLike,
    geodesic_weights: ArrayLike,
    polyhedral_weights: ArrayLike,
    social_weights: ArrayLike,
    bias: ArrayLike,
) -> FloatVector:
    """Compute the Section 7 tanh-combined motor target from supplied sources."""
    sources = [
        _vector(geodesic_velocity, "geodesic_velocity"),
        _vector(polyhedral_pressure, "polyhedral_pressure"),
        _vector(social_action, "social_action"),
    ]
    weights = [
        _finite_array(geodesic_weights, "geodesic_weights", dimensions=(2,)),
        _finite_array(polyhedral_weights, "polyhedral_weights", dimensions=(2,)),
        _finite_array(social_weights, "social_weights", dimensions=(2,)),
    ]
    target_bias = _vector(bias, "bias")
    action_size = target_bias.size
    if any(weight.shape != (action_size, source.size) for weight, source in zip(weights, sources)):
        raise ValueError("each weight matrix must map its source vector to the action dimension")
    return np.tanh(
        weights[0] @ sources[0]
        + weights[1] @ sources[1]
        + weights[2] @ sources[2]
        + target_bias
    )


@dataclass(frozen=True)
class MotorLoopResult:
    action: FloatVector
    target_action: FloatVector
    gradient: FloatVector
    expected_free_energy: float
    target_kl: float
    causal_information: float
    policy_cost: float


def section_07_motor_loop_step(
    current_action: ArrayLike,
    target_distribution: ArrayLike,
    predict_state: StatePredictor,
    causal_information: CausalInformation,
    geodesic_velocity: ArrayLike,
    polyhedral_pressure: ArrayLike,
    social_action: ArrayLike,
    geodesic_weights: ArrayLike,
    polyhedral_weights: ArrayLike,
    social_weights: ArrayLike,
    bias: ArrayLike,
    *,
    dt: float = 0.01,
    learning_rate: float = 0.1,
    lambda_pearl: float = 1.0,
    lambda_policy: float = 1.0,
    finite_difference_step: float = 1e-5,
) -> MotorLoopResult:
    """Take one bounded finite-difference step on the stated EFE objective.

    State prediction and causal information are caller-supplied model outputs.
    This prototype does not implement a state predictor or a causal SCM.
    """
    action = _action(current_action, "current_action")
    target_action = motor_target_action(
        geodesic_velocity,
        polyhedral_pressure,
        social_action,
        geodesic_weights,
        polyhedral_weights,
        social_weights,
        bias,
    )
    if target_action.shape != action.shape:
        raise ValueError("motor target dimension must match current_action")
    target = _vector(target_distribution, "target_distribution")
    if np.any(target < 0.0) or np.sum(target) <= 0.0:
        raise ValueError("target_distribution must contain non-negative values with positive mass")
    if not callable(predict_state) or not callable(causal_information):
        raise ValueError("predict_state and causal_information must be callable")

    step_size = _scalar(dt, "dt", minimum=np.finfo(np.float64).tiny)
    rate = _scalar(learning_rate, "learning_rate", minimum=0.0)
    pearl_weight = _scalar(lambda_pearl, "lambda_pearl", minimum=0.0)
    policy_weight = _scalar(lambda_policy, "lambda_policy", minimum=0.0)
    epsilon = _scalar(finite_difference_step, "finite_difference_step", minimum=np.finfo(np.float64).tiny)

    def components(candidate: FloatVector) -> tuple[float, float, float, float]:
        posterior = predict_state(candidate.copy())
        divergence = categorical_target_kl(posterior, target)
        information = _scalar(causal_information(candidate.copy()), "causal information", minimum=0.0)
        policy_cost = 0.5 * policy_weight * float(np.sum((candidate - target_action) ** 2))
        objective = divergence - pearl_weight * information + policy_cost
        return objective, divergence, information, policy_cost

    objective_before, divergence, information, policy_cost = components(action)
    gradient = np.empty_like(action)
    for index in range(action.size):
        lower = max(-1.0, action[index] - epsilon)
        upper = min(1.0, action[index] + epsilon)
        left, right = action.copy(), action.copy()
        left[index], right[index] = lower, upper
        gradient[index] = (components(right)[0] - components(left)[0]) / (upper - lower)

    updated_action = np.clip(action - step_size * rate * gradient, -1.0, 1.0)
    for array in (target_action, gradient, updated_action):
        array.setflags(write=False)
    return MotorLoopResult(
        action=updated_action,
        target_action=target_action,
        gradient=gradient,
        expected_free_energy=objective_before,
        target_kl=divergence,
        causal_information=information,
        policy_cost=policy_cost,
    )
