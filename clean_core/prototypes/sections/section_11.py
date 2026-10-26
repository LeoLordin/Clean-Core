"""Forward-only, independent prototype for Section 11's activation mask.

Section 11 specifies the mask equation but no straight-through estimator (STE)
or backward rule.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _finite_scalar(value: float, name: str) -> float:
    scalar_array = np.asarray(value)
    if scalar_array.ndim != 0 or np.iscomplexobj(scalar_array):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        scalar = float(scalar_array)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real scalar") from error
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def sigmoidal_spike_activation_mask(
    error: ArrayLike,
    precision: ArrayLike,
    threshold: float,
    beta: float = 5.0,
) -> float:
    """Evaluate Section 11's scalar sigmoid mask in the forward pass only."""
    try:
        error_array = np.asarray(error, dtype=np.float64)
        precision_array = np.asarray(precision, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("error and precision must be finite real arrays") from error

    if (
        error_array.ndim != 1
        or error_array.size == 0
        or precision_array.shape != (error_array.size, error_array.size)
    ):
        raise ValueError("precision must be square and match the error vector")
    if not np.all(np.isfinite(error_array)) or not np.all(np.isfinite(precision_array)):
        raise ValueError("error and precision must contain only finite values")
    if not np.allclose(precision_array, precision_array.T, rtol=0.0, atol=1e-10):
        raise ValueError("precision must be symmetric")
    threshold_value = _finite_scalar(threshold, "threshold")
    beta_value = _finite_scalar(beta, "beta")
    if beta_value <= 0.0:
        raise ValueError("beta must be finite and positive")

    symmetric_precision = 0.5 * (precision_array + precision_array.T)
    try:
        eigenvalues = np.linalg.eigvalsh(symmetric_precision)
    except np.linalg.LinAlgError as error:
        raise ValueError("precision must be positive semidefinite") from error
    psd_tolerance = 1e-10 * max(1.0, float(np.max(np.abs(eigenvalues))))
    if eigenvalues[0] < -psd_tolerance:
        raise ValueError("precision must be positive semidefinite")

    with np.errstate(over="ignore", invalid="ignore"):
        energy = float(error_array @ symmetric_precision @ error_array)
        logit = float(beta_value * (energy - threshold_value))
    if not np.isfinite(energy) or not np.isfinite(logit):
        raise ValueError("weighted error energy and sigmoid input must be finite")

    # Choose the exponent's sign so exp is never evaluated on a large positive
    # input, avoiding overflow while retaining precision in both tails.
    if logit >= 0.0:
        return float(1.0 / (1.0 + np.exp(-logit)))
    exp_logit = np.exp(logit)
    return float(exp_logit / (1.0 + exp_logit))
