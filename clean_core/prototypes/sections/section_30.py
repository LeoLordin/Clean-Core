"""Inference-only prototype for Section 30's fast geodesic relaxation."""

from collections.abc import Sequence

import numpy as np

from clean_core.config import D_LORENTZ, D_SPATIAL, ETA_L, FP64
from clean_core.core.neuron import NeuronMicrostate
from clean_core.geometry.lorentz import exp_map, get_boost_frame


def _finite_array(value, name):
    try:
        array = np.asarray(value, dtype=FP64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite numeric array") from error
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _positive_scalar(value, name):
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return scalar


def _validate_lorentz_point(mu, index):
    point = _finite_array(mu, f"neurons[{index}].mu")
    if point.shape != (D_LORENTZ,):
        raise ValueError(f"neurons[{index}].mu must have shape ({D_LORENTZ},)")
    norm_sq = float(np.sum(point * ETA_L * point))
    if (
        point[0] <= 0.0
        or not np.isfinite(norm_sq)
        or not np.isclose(norm_sq, -1.0, rtol=1e-9, atol=1e-9)
    ):
        raise ValueError(f"neurons[{index}].mu must lie on the future unit Lorentz hyperboloid")
    return point.copy()


def section_30_fast_geodesic_inference_relaxation(
    neurons: Sequence[NeuronMicrostate],
    bottom_up_velocities,
    pi_values,
    *,
    dt=0.01,
    epsilon_0=1e-3,
    v_max=3.0,
):
    """Relax neuron states for each supplied bottom-up velocity sample.

    ``bottom_up_velocities`` has shape ``(steps, neurons, D_SPATIAL)`` and
    contains frame-local velocities for each iteration. ``pi_values`` is a
    positive scalar or one positive scalar per neuron. Each neuron's
    ``Pi_precision`` supplies the local precision matrix. Only ``mu`` is
    updated; model weights and other neuron state are left unchanged.

    Returns the state trajectory, including initial states, with shape
    ``(steps + 1, neurons, D_LORENTZ)``, and the capped local relaxation
    velocities with shape ``(steps, neurons, D_SPATIAL)``.
    """
    try:
        neuron_list = tuple(neurons)
    except TypeError as error:
        raise ValueError("neurons must be a non-empty sequence") from error
    if (
        not neuron_list
        or any(not isinstance(neuron, NeuronMicrostate) for neuron in neuron_list)
        or len({id(neuron) for neuron in neuron_list}) != len(neuron_list)
    ):
        raise ValueError("neurons must contain distinct NeuronMicrostate instances")

    dt = _positive_scalar(dt, "dt")
    epsilon_0 = _positive_scalar(epsilon_0, "epsilon_0")
    v_max = _positive_scalar(v_max, "v_max")
    velocity_samples = _finite_array(bottom_up_velocities, "bottom_up_velocities")
    if velocity_samples.ndim != 3:
        raise ValueError(
            "bottom_up_velocities must have shape "
            f"(steps > 0, {len(neuron_list)}, {D_SPATIAL})"
        )
    expected_shape = (velocity_samples.shape[0], len(neuron_list), D_SPATIAL)
    if velocity_samples.shape != expected_shape:
        raise ValueError(
            "bottom_up_velocities must have shape "
            f"(steps > 0, {len(neuron_list)}, {D_SPATIAL})"
        )
    if velocity_samples.shape[0] == 0:
        raise ValueError("bottom_up_velocities must contain at least one step")

    precision_values = _finite_array(pi_values, "pi_values")
    if precision_values.ndim == 0:
        precision_values = np.full(len(neuron_list), precision_values, dtype=FP64)
    if (
        precision_values.shape != (len(neuron_list),)
        or np.any(precision_values <= 0.0)
    ):
        raise ValueError(f"pi_values must be positive and have shape ({len(neuron_list)},)")

    states = np.stack(
        [_validate_lorentz_point(neuron.mu, index) for index, neuron in enumerate(neuron_list)]
    )
    precision_matrices = []
    for index, neuron in enumerate(neuron_list):
        precision = _finite_array(neuron.Pi_precision, f"neurons[{index}].Pi_precision")
        if precision.shape != (D_SPATIAL, D_SPATIAL):
            raise ValueError(
                f"neurons[{index}].Pi_precision must have shape "
                f"({D_SPATIAL}, {D_SPATIAL})"
            )
        if not np.allclose(precision, precision.T, rtol=1e-9, atol=1e-9):
            raise ValueError(f"neurons[{index}].Pi_precision must be symmetric")
        try:
            np.linalg.cholesky(precision)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                f"neurons[{index}].Pi_precision must be positive definite"
            ) from error
        precision_matrices.append(precision)

    state_trajectory = [states.copy()]
    relaxed_velocities = np.empty_like(velocity_samples)
    with np.errstate(over="ignore", divide="ignore"):
        scale = dt / epsilon_0
    if not np.isfinite(scale):
        raise ValueError("dt / epsilon_0 must be finite")
    identity = np.eye(D_SPATIAL, dtype=FP64)

    for step, step_velocities in enumerate(velocity_samples):
        next_states = np.empty_like(states)
        for index, (neuron, velocity, precision) in enumerate(
            zip(neuron_list, step_velocities, precision_matrices)
        ):
            system = identity + scale * precision_values[index] * precision
            try:
                relaxation_velocity = np.linalg.solve(
                    system, scale * velocity
                )
            except np.linalg.LinAlgError as error:
                raise ValueError("relaxation system must be invertible") from error
            speed = float(np.linalg.norm(relaxation_velocity))
            if speed > v_max:
                relaxation_velocity = relaxation_velocity * (v_max / speed)
            if not np.all(np.isfinite(relaxation_velocity)):
                raise FloatingPointError("relaxation velocity must be finite")

            frame = get_boost_frame(states[index])
            next_states[index] = exp_map(
                states[index], frame @ relaxation_velocity
            )
            relaxed_velocities[step, index] = relaxation_velocity

        if not np.all(np.isfinite(next_states)):
            raise FloatingPointError("relaxed Lorentz states must be finite")
        states = next_states
        for neuron, point in zip(neuron_list, states):
            neuron.mu = point.copy()
        state_trajectory.append(states.copy())

    return np.stack(state_trajectory), relaxed_velocities


__all__ = ["section_30_fast_geodesic_inference_relaxation"]
