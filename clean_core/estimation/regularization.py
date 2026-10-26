import numpy as np


def spectral_lipschitz_penalty(weight_matrices, limit=1.0):
    """Return the squared spectral-norm excess penalty for a collection of weights."""
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("limit must be finite and positive")

    penalty = 0.0
    for weights in weight_matrices:
        matrix = np.asarray(weights, dtype=np.float64)
        if matrix.ndim != 2 or min(matrix.shape) == 0:
            raise ValueError("each weight matrix must be a non-empty 2D array")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("weight matrices must contain only finite values")
        largest_singular_value = np.linalg.svd(matrix, compute_uv=False)[0]
        penalty += max(0.0, largest_singular_value - limit) ** 2
    return float(penalty)
