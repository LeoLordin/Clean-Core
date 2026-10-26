"""Executable tangent-space prototype for Section 10.

This module combines parent context in local tangent coordinates, applies an
active-code SoftVQ distribution, and retracts a bounded perceptron prediction
onto the future unit Lorentz hyperboloid. It does not implement perceptron
training or a complete learning system.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from clean_core.config import D_LORENTZ, D_SPATIAL, ETA_L, FP64, POLE
from clean_core.geometry.lorentz import (
    chordal_distance,
    exp_map,
    get_boost_frame,
    log_map,
    minkowski_norm_sq,
)


@dataclass(frozen=True)
class RetractionPrediction:
    """Intermediate and final values produced by the Section 10 perceptron."""

    hidden: NDArray[np.float64]
    raw_tangent_coordinates: NDArray[np.float64]
    tangent_coordinates: NDArray[np.float64]
    ambient_tangent: NDArray[np.float64]
    predicted_point: NDArray[np.float64]
    error: NDArray[np.float64]


def _finite_vector(values, shape, name):
    vector = np.asarray(values, dtype=FP64)
    if vector.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _hyperboloid_point(values, name):
    point = _finite_vector(values, (D_LORENTZ,), name)
    if point[0] <= 0.0 or not np.isclose(
        minkowski_norm_sq(point), -1.0, rtol=1e-8, atol=1e-8
    ):
        raise ValueError(f"{name} must lie on the future unit hyperboloid")
    return point


def aggregate_parent_context(parent_codes, edge_weights):
    """Sum parent tangent codes after applying their scalar edge weights."""
    codes = np.asarray(parent_codes, dtype=FP64)
    if codes.ndim != 2 or codes.shape[0] == 0 or codes.shape[1] != D_SPATIAL:
        raise ValueError(f"parent_codes must have shape (n > 0, {D_SPATIAL})")
    if not np.all(np.isfinite(codes)):
        raise ValueError("parent_codes must contain only finite values")

    weights = _finite_vector(edge_weights, (codes.shape[0],), "edge_weights")
    return weights @ codes


def introspection_tangent_context(reference_mu, introspection_mu):
    """Project the Lorentz-log introspection signal into the reference frame."""
    reference = _hyperboloid_point(reference_mu, "reference_mu")
    introspection = _hyperboloid_point(introspection_mu, "introspection_mu")
    tangent = log_map(reference, introspection)
    return get_boost_frame(reference).T @ (ETA_L * tangent)


def _softvq_inputs(mu_point, centroids, active_mask, temperature, epsilon, base_point):
    point = _hyperboloid_point(mu_point, "mu_point")
    origin = _hyperboloid_point(base_point, "base_point")
    codebook = np.asarray(centroids, dtype=FP64)
    if codebook.ndim != 2 or codebook.shape[0] == 0 or codebook.shape[1] != D_SPATIAL:
        raise ValueError(f"centroids must have shape (n > 0, {D_SPATIAL})")
    if not np.all(np.isfinite(codebook)):
        raise ValueError("centroids must contain only finite values")
    if not np.allclose(codebook[0], 0.0, atol=1e-12):
        raise ValueError("centroids[0] must be the zero (null) code")

    if active_mask is None:
        mask = np.ones(codebook.shape[0], dtype=bool)
    else:
        supplied_mask = np.asarray(active_mask)
        if supplied_mask.shape != (codebook.shape[0],):
            raise ValueError(f"active_mask must have shape ({codebook.shape[0]},)")
        if supplied_mask.dtype.kind not in "biuf":
            raise ValueError("active_mask must contain only boolean or binary values")
        if not np.all(np.isfinite(supplied_mask)) or not np.all(
            np.isin(supplied_mask, (0, 1))
        ):
            raise ValueError("active_mask must contain only boolean or binary values")
        mask = supplied_mask.astype(bool)
    if not mask[0]:
        raise ValueError("active_mask must preserve the zero (null) code")

    temperature = float(temperature)
    epsilon = float(epsilon)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    return point, origin, codebook, mask, temperature, epsilon


def softvq_tangent_weights(
    mu_point,
    centroids,
    active_mask=None,
    temperature=0.1,
    epsilon=1e-12,
    base_point=POLE,
):
    """Return active-code SoftVQ weights, keeping the null code active.

    Centroid vectors are coordinates in the tangent space at ``base_point``.
    The stabilizing offset is included in the denominator as in the formula.
    """
    point, origin, codebook, mask, temperature, epsilon = _softvq_inputs(
        mu_point, centroids, active_mask, temperature, epsilon, base_point
    )
    frame = get_boost_frame(origin)
    active_indices = np.flatnonzero(mask)
    with np.errstate(over="ignore", invalid="ignore"):
        code_points = [
            exp_map(origin.copy(), frame @ codebook[index])
            for index in active_indices
        ]
    if not all(np.all(np.isfinite(candidate)) for candidate in code_points):
        raise ValueError("active centroids must map to finite hyperboloid points")
    with np.errstate(over="ignore", invalid="ignore"):
        active_distances_sq = np.asarray(
            [chordal_distance(point, candidate) ** 2 for candidate in code_points],
            dtype=FP64,
        )
    if not np.all(np.isfinite(active_distances_sq)):
        raise ValueError("active centroid distances must be finite")

    minimum_active = float(np.min(active_distances_sq))
    logits = -(active_distances_sq - minimum_active) / temperature
    numerators = np.zeros_like(mask, dtype=FP64)
    numerators[active_indices] = np.exp(logits)
    return numerators / (float(np.sum(numerators)) + epsilon)


def softvq_tangent(
    mu_point,
    centroids,
    active_mask=None,
    temperature=0.1,
    epsilon=1e-12,
    base_point=POLE,
):
    """Quantize a hyperboloid point to the weighted tangent-space code."""
    _, _, codebook, _, _, _ = _softvq_inputs(
        mu_point, centroids, active_mask, temperature, epsilon, base_point
    )
    weights = softvq_tangent_weights(
        mu_point,
        codebook,
        active_mask=active_mask,
        temperature=temperature,
        epsilon=epsilon,
        base_point=base_point,
    )
    return weights @ codebook


def _stable_sigmoid(values):
    result = np.empty_like(values)
    nonnegative = values >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exp_values = np.exp(values[~nonnegative])
    result[~nonnegative] = exp_values / (1.0 + exp_values)
    return result


def retraction_perceptron(
    context,
    reference_mu,
    actual_mu,
    w1,
    b1,
    w2,
    b2,
    max_radius=3.0,
):
    """Predict on the hyperboloid and return the logarithmic prediction error.

    ``w1`` has shape ``(hidden_dim, D_SPATIAL)`` and ``w2`` has shape
    ``(D_SPATIAL, hidden_dim)``. The output tangent coordinates are clipped to
    ``max_radius`` before being mapped through the local Lorentz frame.
    """
    context = _finite_vector(context, (D_SPATIAL,), "context")
    reference = _hyperboloid_point(reference_mu, "reference_mu")
    actual = _hyperboloid_point(actual_mu, "actual_mu")
    weights1 = np.asarray(w1, dtype=FP64)
    if (
        weights1.ndim != 2
        or weights1.shape[0] == 0
        or weights1.shape[1] != D_SPATIAL
    ):
        raise ValueError(f"w1 must have shape (hidden_dim > 0, {D_SPATIAL})")
    if not np.all(np.isfinite(weights1)):
        raise ValueError("w1 must contain only finite values")
    hidden_bias = _finite_vector(b1, (weights1.shape[0],), "b1")
    weights2 = np.asarray(w2, dtype=FP64)
    if weights2.shape != (D_SPATIAL, weights1.shape[0]):
        raise ValueError(
            f"w2 must have shape ({D_SPATIAL}, {weights1.shape[0]})"
        )
    if not np.all(np.isfinite(weights2)):
        raise ValueError("w2 must contain only finite values")
    output_bias = _finite_vector(b2, (D_SPATIAL,), "b2")
    max_radius = float(max_radius)
    if not np.isfinite(max_radius) or max_radius <= 0.0:
        raise ValueError("max_radius must be finite and positive")

    hidden = _stable_sigmoid(weights1 @ context + hidden_bias)
    raw_coordinates = weights2 @ hidden + output_bias
    if not np.all(np.isfinite(raw_coordinates)):
        raise ValueError("perceptron output must contain only finite values")
    raw_norm = float(np.hypot.reduce(raw_coordinates))
    if raw_norm > max_radius:
        coordinates = raw_coordinates * (max_radius / raw_norm)
    else:
        coordinates = raw_coordinates.copy()

    ambient_tangent = get_boost_frame(reference) @ coordinates
    predicted_point = exp_map(reference.copy(), ambient_tangent)
    error = log_map(actual, predicted_point)
    return RetractionPrediction(
        hidden=hidden,
        raw_tangent_coordinates=raw_coordinates,
        tangent_coordinates=coordinates,
        ambient_tangent=ambient_tangent,
        predicted_point=predicted_point,
        error=error,
    )


__all__ = [
    "RetractionPrediction",
    "aggregate_parent_context",
    "introspection_tangent_context",
    "retraction_perceptron",
    "softvq_tangent",
    "softvq_tangent_weights",
]
