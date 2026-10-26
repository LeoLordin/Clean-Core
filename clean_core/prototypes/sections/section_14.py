"""Validated Section 14 operators composed from the Lorentz geometry API."""

import numpy as np

from clean_core.config import D_LORENTZ, ETA_L, FP64
from clean_core.geometry.lorentz import (
    chordal_distance,
    exp_map,
    log_map as _log_map,
    minkowski_dot,
    minkowski_norm as _minkowski_norm,
    minkowski_norm_sq,
    parallel_transport as _parallel_transport,
)


def _vector(value, name):
    vector = np.asarray(value, dtype=FP64)
    if vector.shape != (D_LORENTZ,):
        raise ValueError(f"{name} must have shape ({D_LORENTZ},)")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _point(value, name):
    point = _vector(value, name)
    if point[0] <= 0.0 or not np.isclose(
        minkowski_norm_sq(point), -1.0, rtol=0.0, atol=1e-8
    ):
        raise ValueError(f"{name} must lie on the future unit hyperboloid")
    return point


def _tangent(base, value, name):
    tangent = _vector(value, name)
    scale = max(1.0, float(np.linalg.norm(tangent)))
    if abs(minkowski_dot(base, tangent)) > 1e-8 * scale:
        raise ValueError(f"{name} must be tangent at the base point")
    return tangent


def minkowski_metric():
    """Return the diagonal Lorentz metric in the configured dimension."""
    return np.diag(ETA_L).astype(FP64, copy=True)


def minkowski_inner_product(x, y):
    """Evaluate the Lorentz inner product of two ambient vectors."""
    x = _vector(x, "x")
    y = _vector(y, "y")
    return minkowski_dot(x, y)


def minkowski_norm(vector):
    """Return the nonnegative Minkowski norm used by the Section 14 maps."""
    vector = _vector(vector, "vector")
    return _minkowski_norm(vector)


def geodesic_distance(x, y):
    """Compute hyperbolic distance between two future unit points."""
    x = _point(x, "x")
    y = _point(y, "y")
    return chordal_distance(x, y)


def geodesic_exp(mu, tangent):
    """Map a tangent vector at ``mu`` onto the future unit hyperboloid."""
    mu = _point(mu, "mu")
    tangent = _tangent(mu, tangent, "tangent")
    return _point(exp_map(mu.copy(), tangent.copy()), "result")


def geodesic_log(mu, y):
    """Map a future unit point ``y`` into the tangent space at ``mu``."""
    mu = _point(mu, "mu")
    y = _point(y, "y")
    u = y + minkowski_dot(mu, y) * mu
    norm_u = _minkowski_norm(u)
    if norm_u < 1e-4:
        logarithm = (
            1.0 - norm_u**2 / 6.0 + 3.0 * norm_u**4 / 40.0
        ) * u
    else:
        logarithm = _log_map(mu, y)
    return _tangent(mu, logarithm, "logarithm")


def parallel_transport(x, y, tangent):
    """Transport a tangent vector from ``x`` to ``y`` along their geodesic."""
    x = _point(x, "x")
    y = _point(y, "y")
    tangent = _tangent(x, tangent, "tangent")
    transported = _parallel_transport(x, y, tangent)
    if abs(minkowski_dot(y, transported)) > 1e-8 * max(
        1.0, float(np.linalg.norm(transported))
    ):
        # The geometry helper returns the input unchanged for very short
        # geodesics; use the closed form here to preserve target tangency.
        denominator = 1.0 - minkowski_dot(x, y)
        transported = tangent + (
            minkowski_dot(tangent, y) / denominator
        ) * (x + y)
    return _tangent(y, transported, "transported tangent")


__all__ = [
    "geodesic_distance",
    "geodesic_exp",
    "geodesic_log",
    "minkowski_inner_product",
    "minkowski_metric",
    "minkowski_norm",
    "parallel_transport",
]
