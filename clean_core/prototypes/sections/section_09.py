"""Finite numeric objective for the local Section 9 formula.

This evaluates one error vector and one DLR precision model. It is a bounded
algebraic prototype, not an implementation of the full free-energy theory.
"""

import numpy as np
from numpy.typing import ArrayLike


def _finite_array(name: str, value: ArrayLike) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty array of finite values")
    return array


def _positive_scalar(name: str, value: float) -> float:
    try:
        scalar_array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite positive scalar") from exc
    if scalar_array.ndim != 0:
        raise ValueError(f"{name} must be a finite positive scalar")
    scalar = float(scalar_array)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return scalar


def section_09_pragmatic_free_energy(
    prediction_error: ArrayLike,
    precision_scalar: float,
    d_diag: ArrayLike,
    low_rank_basis: ArrayLike,
    low_rank_values: ArrayLike,
    trace_probes: ArrayLike,
) -> float:
    """Evaluate the Section 9 summand for a single local state.

    ``d_diag`` and ``low_rank_values`` define the covariance
    ``diag(d_diag) + V diag(low_rank_values) V.T`` used by the existing DLR
    estimator. Its inverse is computed with the Woodbury identity. Probe rows
    must contain Rademacher values (-1 or +1); their average estimates the
    trace correction in the stated formula.
    """
    error = _finite_array("prediction_error", prediction_error)
    diagonal = _finite_array("d_diag", d_diag)
    basis = _finite_array("low_rank_basis", low_rank_basis)
    values = _finite_array("low_rank_values", low_rank_values)
    probes = _finite_array("trace_probes", trace_probes)
    scalar = _positive_scalar("precision_scalar", precision_scalar)

    if error.ndim != 1:
        raise ValueError("prediction_error must be a vector")
    dimension = error.size
    if diagonal.shape != (dimension,):
        raise ValueError(f"d_diag must have shape ({dimension},)")
    if basis.ndim != 2 or basis.shape[0] != dimension:
        raise ValueError(f"low_rank_basis must have {dimension} rows")
    rank = basis.shape[1]
    if rank < 1 or rank > dimension or values.shape != (rank,):
        raise ValueError("low_rank_values must match a non-empty basis rank")
    if probes.ndim != 2 or probes.shape[0] < 1 or probes.shape[1] != dimension:
        raise ValueError(f"trace_probes must have shape (M, {dimension}) with M >= 1")
    if np.any(diagonal <= 0.0):
        raise ValueError("d_diag entries must be positive")
    if np.any(values <= 0.0):
        raise ValueError("low_rank_values entries must be positive")
    if np.any((probes != -1.0) & (probes != 1.0)):
        raise ValueError("trace_probes must contain only -1 or +1")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        d_inv = 1.0 / diagonal
        d_inv_v = d_inv[:, None] * basis
        middle = np.diag(1.0 / values) + basis.T @ d_inv_v
        sign, logdet_middle = np.linalg.slogdet(middle)
        if sign <= 0.0 or not np.isfinite(logdet_middle):
            raise FloatingPointError("DLR log-determinant is not finite")

        precision = np.diag(d_inv) - d_inv_v @ np.linalg.solve(
            middle, d_inv_v.T
        )
        quadratic = scalar * 0.5 * float(error @ precision @ error)
        log_precision = -0.5 * (dimension - 1) * np.log(scalar)
        log_covariance = 0.5 * (
            float(np.sum(np.log(diagonal)))
            + float(np.sum(np.log(values)))
            + float(logdet_middle)
        )
        trace_correction = 0.5 * float(
            np.mean(np.einsum("mi,ij,mj->m", probes, precision - np.eye(dimension), probes))
        )
        objective = quadratic + log_precision + log_covariance + trace_correction

    if not np.isfinite(objective):
        raise FloatingPointError("Section 9 objective must be finite")
    return float(objective)
