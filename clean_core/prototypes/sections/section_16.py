"""Small numerical primitives for Section 16's NOTEARS update.

This prototype implements the log-determinant acyclicity penalty and one
gradient step followed by a real-Schur spectral-radius cap. The cap is not a
DAG projection, and these primitives do not identify causal effects.
"""

import numpy as np
from scipy.linalg import expm, schur


def _as_finite_square_matrix(value: object, name: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a real numeric matrix") from error
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _acyclicity_exponential(adjacency: object) -> tuple[np.ndarray, np.ndarray]:
    matrix = _as_finite_square_matrix(adjacency, "adjacency")
    exponential = expm(matrix * matrix)
    if not np.all(np.isfinite(exponential)):
        raise ValueError("adjacency is too large for a finite acyclicity calculation")
    trace = float(np.trace(exponential))
    if not np.isfinite(trace) or trace <= 0.0:
        raise ValueError("acyclicity exponential must have a finite positive trace")
    return matrix, exponential


def log_acyclicity(adjacency: object) -> float:
    """Return ``log(trace(expm(A ∘ A)) / d)`` for a real adjacency matrix."""
    matrix, exponential = _acyclicity_exponential(adjacency)
    return float(np.log(np.trace(exponential) / matrix.shape[0]))


def log_acyclicity_gradient(adjacency: object) -> np.ndarray:
    """Return the Section 16 elementwise gradient of :func:`log_acyclicity`."""
    matrix, exponential = _acyclicity_exponential(adjacency)
    gradient = (2.0 / np.trace(exponential)) * (exponential.T * matrix)
    if not np.all(np.isfinite(gradient)):
        raise ValueError("acyclicity gradient is not finite")
    return gradient


def schur_capped_intervention(
    adjacency: object,
    causal_gradient: object,
    *,
    dt: float,
    epsilon1: float,
    mu_dag: float = 1.0,
    epsilon_gap: float = 0.05,
) -> np.ndarray:
    """Take the Section 16 gradient step and cap its spectral radius.

    ``causal_gradient`` is the gradient of the causal objective. After the
    specified update, a real Schur decomposition is used to uniformly scale
    the matrix if needed, keeping its spectral radius just below
    ``1 - epsilon_gap``. This cap does not enforce acyclicity or preserve the
    causal objective's descent direction after scaling.
    """
    matrix = _as_finite_square_matrix(adjacency, "adjacency")
    causal_grad = _as_finite_square_matrix(causal_gradient, "causal_gradient")
    if causal_grad.shape != matrix.shape:
        raise ValueError("causal_gradient must have the same shape as adjacency")
    scalar_parameters = {}
    for name, value in (
        ("dt", dt),
        ("epsilon1", epsilon1),
        ("mu_dag", mu_dag),
        ("epsilon_gap", epsilon_gap),
    ):
        if not np.isscalar(value):
            raise ValueError(f"{name} must be finite")
        try:
            scalar = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be finite") from error
        if not np.isfinite(scalar):
            raise ValueError(f"{name} must be finite")
        scalar_parameters[name] = scalar
    dt = scalar_parameters["dt"]
    epsilon1 = scalar_parameters["epsilon1"]
    mu_dag = scalar_parameters["mu_dag"]
    epsilon_gap = scalar_parameters["epsilon_gap"]
    if dt <= 0.0 or epsilon1 <= 0.0:
        raise ValueError("dt and epsilon1 must be positive")
    if mu_dag < 0.0:
        raise ValueError("mu_dag must be non-negative")
    if not 0.0 < epsilon_gap < 1.0:
        raise ValueError("epsilon_gap must be between 0 and 1")

    updated = matrix - (dt / epsilon1) * (
        -causal_grad + mu_dag * log_acyclicity_gradient(matrix)
    )
    if not np.all(np.isfinite(updated)):
        raise ValueError("gradient step produced non-finite adjacency values")

    schur_form, schur_vectors = schur(updated, output="real")
    eigenvalues = np.linalg.eigvals(schur_form)
    radius = float(np.max(np.abs(eigenvalues)))
    cap = 1.0 - epsilon_gap
    strict_cap = cap * (1.0 - 1e-12)
    if radius > strict_cap:
        schur_form *= strict_cap / radius
    result = schur_vectors @ schur_form @ schur_vectors.T
    if not np.all(np.isfinite(result)):
        raise ValueError("Schur-capped intervention produced non-finite values")
    return result
