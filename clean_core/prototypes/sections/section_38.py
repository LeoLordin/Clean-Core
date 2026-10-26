"""Zero-padding isometric embeddings for Section 38.

The embedding preserves the Lorentz inner products by appending spatial zero
coordinates. It does not implement or imply the conjectural capacity scaling.
"""

import numpy as np

from clean_core.config import FP64


def _validated_points(points, *, batch, name="points"):
    try:
        result = np.asarray(points, dtype=FP64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain finite numeric coordinates") from error

    allowed_ndim = (2,) if batch else (1, 2)
    if result.ndim not in allowed_ndim:
        if batch:
            raise ValueError(f"{name} must be a batch of ambient vectors")
        raise ValueError(
            f"{name} must be an ambient vector or a batch of ambient vectors"
        )
    if result.shape[-1] < 2:
        raise ValueError(f"{name} must have at least two ambient coordinates")
    if result.ndim == 2 and result.shape[0] == 0:
        raise ValueError(f"{name} batch must not be empty")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite coordinates")
    return result


def _validated_tolerance(tolerance):
    try:
        tolerance = float(tolerance)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("tolerance must be finite and positive") from error
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    return tolerance


def expand_hyperboloid_embedding(points, extra_spatial_dimensions, tolerance=1e-10):
    """Embed one or more future-unit points by appending spatial zeros.

    ``points`` may be a single ambient vector or a non-empty batch with shape
    ``(batch_size, ambient_dimension)``. The time coordinate remains first,
    and the returned ambient dimension is increased by ``k``.
    """
    points = _validated_points(points, batch=False)
    tolerance = _validated_tolerance(tolerance)
    if (
        isinstance(extra_spatial_dimensions, (bool, np.bool_))
        or not isinstance(extra_spatial_dimensions, (int, np.integer))
        or extra_spatial_dimensions < 1
    ):
        raise ValueError("extra_spatial_dimensions must be a positive integer")

    with np.errstate(over="ignore", invalid="ignore"):
        lorentz_norm = -points[..., 0] ** 2 + np.sum(points[..., 1:] ** 2, axis=-1)
    if (
        not np.all(np.isfinite(lorentz_norm))
        or np.any(points[..., 0] <= 0.0)
        or not np.allclose(lorentz_norm, -1.0, atol=tolerance, rtol=0.0)
    ):
        raise ValueError("every point must lie on the future unit hyperboloid")

    padding_shape = points.shape[:-1] + (int(extra_spatial_dimensions),)
    padding = np.zeros(padding_shape, dtype=FP64)
    return np.concatenate((points, padding), axis=-1)


def minkowski_gram_matrix(points):
    """Return the pairwise Lorentz inner products of a non-empty point batch."""
    points = _validated_points(points, batch=True)
    metric_signs = np.ones(points.shape[1], dtype=FP64)
    metric_signs[0] = -1.0
    with np.errstate(over="ignore", invalid="ignore"):
        gram = (points * metric_signs) @ points.T
    if not np.all(np.isfinite(gram)):
        raise ValueError("Minkowski Gram matrix is too large to represent")
    return gram


__all__ = ["expand_hyperboloid_embedding", "minkowski_gram_matrix"]
