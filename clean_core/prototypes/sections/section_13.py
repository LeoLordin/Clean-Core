"""Standalone prototype for Section 13 spectral Lipschitz regularization."""

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["spectral_norm_estimate", "spectral_lipschitz_penalty"]


def spectral_norm_estimate(weights: ArrayLike) -> float:
    """Return the largest singular value of a finite, nonempty weight matrix."""
    matrix = np.asarray(weights, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) == 0:
        raise ValueError("each weight matrix must be a non-empty 2D array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("weight matrices must contain only finite values")

    try:
        singular_values = np.linalg.svd(matrix, compute_uv=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("spectral norm could not be estimated") from exc
    largest_singular_value = float(singular_values[0])
    if not np.isfinite(largest_singular_value):
        raise ValueError("spectral norm estimate is too large to represent")
    return largest_singular_value


def spectral_lipschitz_penalty(
    weight_matrices: Iterable[ArrayLike], lambda_lip: float = 1.0
) -> float:
    """Evaluate Section 13's hinge-squared penalty over all supplied weights.

    ``weight_matrices`` contains the ``W_i^(1, l)`` and ``W_i^(2, l)``
    matrices across layers. This penalty alone does not establish a
    training-wide Lipschitz guarantee.
    """
    coefficient = float(lambda_lip)
    if not np.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("lambda_lip must be finite and non-negative")

    penalty_sum = 0.0
    try:
        matrices = iter(weight_matrices)
    except TypeError as exc:
        raise ValueError("weight_matrices must be an iterable of matrices") from exc

    for weights in matrices:
        largest_singular_value = spectral_norm_estimate(weights)
        excess = max(0.0, largest_singular_value - 1.0)
        penalty_sum += excess * excess
        if not np.isfinite(penalty_sum):
            raise ValueError("spectral penalty is too large to represent")

    penalty = coefficient * penalty_sum
    if not np.isfinite(penalty):
        raise ValueError("spectral penalty is too large to represent")
    return float(penalty)
