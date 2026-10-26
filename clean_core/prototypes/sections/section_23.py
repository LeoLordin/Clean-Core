"""Executable NumPy prototype for Section 23 associative attention.

The routines evaluate the stated manifold attention and transported linear
crossbar-current equations. They are numerical prototypes, not hardware models.
"""

import numpy as np

from clean_core.config import D_LORENTZ, D_SPATIAL, ETA_L, E_BASIS, FP64, POLE
from clean_core.geometry.lorentz import (
    chordal_distance,
    exp_map,
    get_boost_frame,
    log_map,
    parallel_transport,
)


def _finite_scalar(name, value):
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite scalar")
    return scalar


def _hyperboloid_point(name, value):
    point = np.asarray(value, dtype=FP64)
    if point.shape != (D_LORENTZ,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be a finite vector with shape ({D_LORENTZ},)")
    norm_sq = float(np.dot(ETA_L * point, point))
    if point[0] <= 0.0 or not np.isclose(norm_sq, -1.0, atol=1e-8):
        raise ValueError(f"{name} must lie on the future unit hyperboloid")
    return point


def _hyperboloid_points(name, values):
    points = np.asarray(values, dtype=FP64)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != D_LORENTZ:
        raise ValueError(f"{name} must have shape (n > 0, {D_LORENTZ})")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain only finite values")
    for point in points:
        _hyperboloid_point(name, point)
    return points


def section_23_project_memory(memory, key_weights, value_weights):
    """Project Euclidean memory vectors onto the Section 23 key/value manifolds."""
    memory = np.asarray(memory, dtype=FP64)
    if memory.ndim != 2 or memory.shape[0] == 0 or memory.shape[1] == 0:
        raise ValueError("memory must have shape (n > 0, h_dim > 0)")
    if not np.all(np.isfinite(memory)):
        raise ValueError("memory must contain only finite values")

    key_weights = np.asarray(key_weights, dtype=FP64)
    value_weights = np.asarray(value_weights, dtype=FP64)
    expected_shape = (D_SPATIAL, memory.shape[1])
    if key_weights.shape != expected_shape:
        raise ValueError(f"key_weights must have shape {expected_shape}")
    if value_weights.shape != expected_shape:
        raise ValueError(f"value_weights must have shape {expected_shape}")
    if not np.all(np.isfinite(key_weights)) or not np.all(np.isfinite(value_weights)):
        raise ValueError("projection weights must contain only finite values")

    keys = np.stack(
        [
            exp_map(POLE.copy(), E_BASIS.T @ (key_weights @ item))
            for item in memory
        ]
    )
    values = np.stack(
        [
            exp_map(POLE.copy(), E_BASIS.T @ (value_weights @ item))
            for item in memory
        ]
    )
    return {
        "keys": _hyperboloid_points("projected keys", keys),
        "values": _hyperboloid_points("projected values", values),
    }


def section_23_associative_attention(query, keys, values, temperature, dopamine=1.0):
    """Compute Section 23 manifold scores, softmax weights, and local update."""
    query = _hyperboloid_point("query", query)
    keys = _hyperboloid_points("keys", keys)
    values = _hyperboloid_points("values", values)
    if keys.shape[0] != values.shape[0]:
        raise ValueError("keys and values must contain the same number of points")

    temperature = _finite_scalar("temperature", temperature)
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    dopamine = _finite_scalar("dopamine", dopamine)
    if dopamine < 0.0:
        raise ValueError("dopamine must be non-negative")

    scale = temperature * np.sqrt(D_SPATIAL)
    scores = np.array(
        [-(chordal_distance(query, key) ** 2) / scale for key in keys],
        dtype=FP64,
    )
    unnormalized = np.exp(scores - np.max(scores))
    weights = unnormalized / np.sum(unnormalized)

    local_frame = get_boost_frame(query)
    local_logs = np.stack(
        [
            local_frame.T @ (ETA_L * log_map(query, value))
            for value in values
        ]
    )
    update = dopamine * np.sum(weights[:, None] * local_logs, axis=0)
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(weights)):
        raise ValueError("attention scores and weights must be finite")
    if not np.all(np.isfinite(update)):
        raise ValueError("attention update must contain only finite values")
    return {
        "scores": scores,
        "weights": weights,
        "update": np.asarray(update, dtype=FP64),
    }


def section_23_crossbar_current(query, analog_weights, state):
    """Calculate the Section 23 linear current and express it locally at query."""
    query = _hyperboloid_point("query", query)
    analog_weights = np.asarray(analog_weights, dtype=FP64)
    if analog_weights.shape != (D_SPATIAL, D_SPATIAL):
        raise ValueError(
            f"analog_weights must have shape ({D_SPATIAL}, {D_SPATIAL})"
        )
    if not np.all(np.isfinite(analog_weights)):
        raise ValueError("analog_weights must contain only finite values")
    state = np.asarray(state, dtype=FP64)
    if state.shape != (D_SPATIAL,):
        raise ValueError(f"state must have shape ({D_SPATIAL},)")
    if not np.all(np.isfinite(state)):
        raise ValueError("state must contain only finite values")

    origin_tangent = E_BASIS.T @ (analog_weights @ state)
    transported = parallel_transport(POLE, query, origin_tangent)
    local_frame = get_boost_frame(query)
    current = local_frame.T @ (ETA_L * transported)
    if not np.all(np.isfinite(current)):
        raise ValueError("crossbar current must contain only finite values")
    return np.asarray(current, dtype=FP64)


def section_23_update(
    query, keys, values, temperature, analog_weights, state, dopamine=1.0
):
    """Combine associative attention and transported crossbar current."""
    attention = section_23_associative_attention(
        query, keys, values, temperature, dopamine=dopamine
    )
    current = section_23_crossbar_current(query, analog_weights, state)
    total = attention["update"] + current
    if not np.all(np.isfinite(total)):
        raise ValueError("combined update must contain only finite values")
    return {
        **attention,
        "crossbar_current": current,
        "total_update": np.asarray(total, dtype=FP64),
    }


__all__ = [
    "section_23_associative_attention",
    "section_23_crossbar_current",
    "section_23_project_memory",
    "section_23_update",
]
