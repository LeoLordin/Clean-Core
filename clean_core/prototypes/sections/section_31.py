"""Standalone prototype for Section 31's conformal Christoffel correction."""

import numpy as np

from clean_core.config import ETA_L, FP64


def _finite_array(value, name):
    try:
        array = np.asarray(value, dtype=FP64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite numeric array") from error
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def conformal_christoffel_correction(
    gradient_log_omega,
    lorentz_metric=ETA_L,
):
    """Compute the conformal contribution for a diagonal Lorentz metric.

    The result is indexed as ``correction[A, B, C]`` and evaluates
    ``delta_B^A q_C + delta_C^A q_B - eta_BC eta^{AK} q_K``, where
    ``q`` is ``gradient_log_omega``. ``lorentz_metric`` contains the
    non-zero diagonal entries of ``eta``. This computes only the conformal
    correction; it does not provide a general connection implementation.
    """
    gradient = _finite_array(gradient_log_omega, "gradient_log_omega")
    metric = _finite_array(lorentz_metric, "lorentz_metric")
    if gradient.ndim != 1 or gradient.size == 0:
        raise ValueError("gradient_log_omega must be a non-empty vector")
    if metric.shape != gradient.shape:
        raise ValueError("lorentz_metric must have one diagonal entry per gradient component")
    if np.any(metric == 0.0):
        raise ValueError("lorentz_metric must contain finite non-zero diagonal entries")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        identity = np.eye(gradient.size, dtype=FP64)
        raised_gradient = gradient / metric
        metric_tensor = np.diag(metric)
        correction = (
            np.einsum("ab,c->abc", identity, gradient)
            + np.einsum("ac,b->abc", identity, gradient)
            - np.einsum("bc,a->abc", metric_tensor, raised_gradient)
        )
    if not np.all(np.isfinite(correction)):
        raise ValueError("conformal Christoffel correction is too large to represent")
    return correction


def conformal_christoffel_symbols(
    base_connection,
    gradient_log_omega,
    lorentz_metric=ETA_L,
):
    """Add Section 31's conformal correction to a supplied base connection.

    The base and result use ``[A, B, C]`` indexing. This helper only applies
    the documented conformal formula to a supplied rank-three array.
    """
    base = _finite_array(base_connection, "base_connection")
    if base.ndim != 3 or not (base.shape[0] == base.shape[1] == base.shape[2]):
        raise ValueError("base_connection must have shape (n, n, n)")
    correction = conformal_christoffel_correction(
        gradient_log_omega,
        lorentz_metric=lorentz_metric,
    )
    if base.shape != correction.shape:
        raise ValueError("base_connection dimensions must match the gradient")

    with np.errstate(over="ignore", invalid="ignore"):
        symbols = base + correction
    if not np.all(np.isfinite(symbols)):
        raise ValueError("conformal Christoffel symbols are too large to represent")
    return symbols


__all__ = [
    "conformal_christoffel_correction",
    "conformal_christoffel_symbols",
]
