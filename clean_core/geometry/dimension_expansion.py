import numpy as np

from clean_core.config import FP64


def expand_hyperboloid_embedding(points, extra_spatial_dimensions, tolerance=1e-10):
    points = np.asarray(points, dtype=FP64)
    if points.ndim not in (1, 2) or points.shape[-1] < 2:
        raise ValueError("points must be an ambient vector or a batch of ambient vectors")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must contain only finite coordinates")
    if not isinstance(extra_spatial_dimensions, (int, np.integer)) or extra_spatial_dimensions < 1:
        raise ValueError("extra_spatial_dimensions must be a positive integer")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    minkowski_norm = -points[..., 0] ** 2 + np.sum(points[..., 1:] ** 2, axis=-1)
    if np.any(points[..., 0] <= 0.0) or not np.allclose(minkowski_norm, -1.0, atol=tolerance, rtol=0.0):
        raise ValueError("every point must lie on the upper unit hyperboloid")
    padding_shape = points.shape[:-1] + (extra_spatial_dimensions,)
    padding = np.zeros(padding_shape, dtype=FP64)
    return np.concatenate((points, padding), axis=-1)


def minkowski_gram_matrix(points):
    points = np.asarray(points, dtype=FP64)
    if points.ndim != 2 or points.shape[1] < 2 or not np.all(np.isfinite(points)):
        raise ValueError("points must be a finite matrix of ambient coordinates")
    metric_signs = np.ones(points.shape[1], dtype=FP64)
    metric_signs[0] = -1.0
    return (points * metric_signs) @ points.T