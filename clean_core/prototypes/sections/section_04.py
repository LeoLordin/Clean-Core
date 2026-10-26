"""Bounded graph prototype for Section 4's linearized Cole-Hopf equation.

The graph Laplacian is a discrete stand-in for the hyperbolic
Laplace-Beltrami operator. This is not a complete HJB solver.
"""

import numpy as np

from clean_core.cognition.cole_hopf import (
    cole_hopf_graph_diffusion_step,
    cole_hopf_optimal_velocity,
    superpose_goal_wavefunctions as _superpose_goal_wavefunctions,
    value_to_wavefunction as _value_to_wavefunction,
    wavefunction_to_value as _wavefunction_to_value,
)
from clean_core.config import FP64


def _validated_scale(lambda_value, sigma):
    try:
        lambda_value = float(lambda_value)
        sigma = float(sigma)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("lambda_value and sigma must be finite positive scalars") from error
    if not np.isfinite(lambda_value) or lambda_value <= 0.0:
        raise ValueError("lambda_value must be finite and positive")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    with np.errstate(over="ignore", invalid="ignore"):
        scale = np.float64(lambda_value) * np.square(np.float64(sigma))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("lambda_value * sigma**2 must be finite and positive")
    return lambda_value, sigma


def value_to_wavefunction(value, lambda_value, sigma):
    """Apply the positive Cole-Hopf transform to finite value samples."""
    lambda_value, sigma = _validated_scale(lambda_value, sigma)
    result = _value_to_wavefunction(value, lambda_value, sigma)
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise FloatingPointError("Cole-Hopf transform must produce finite positive values")
    return result


def wavefunction_to_value(wavefunction, lambda_value, sigma):
    """Recover values from a finite, strictly positive wavefunction."""
    lambda_value, sigma = _validated_scale(lambda_value, sigma)
    result = _wavefunction_to_value(wavefunction, lambda_value, sigma)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("inverse Cole-Hopf transform must be finite")
    return result


def superpose_goal_wavefunctions(wavefunctions, weights=None):
    """Linearly combine finite positive goal wavefunctions."""
    result = _superpose_goal_wavefunctions(wavefunctions, weights)
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise FloatingPointError("wavefunction superposition must be finite and positive")
    return result


def linearized_hjb_graph_step(
    wavefunction,
    edge_weights,
    pragmatic_potential,
    dt,
    lambda_value,
    sigma,
):
    """Advance a graph-discretized linearized HJB equation by one step."""
    lambda_value, sigma = _validated_scale(lambda_value, sigma)
    result = cole_hopf_graph_diffusion_step(
        wavefunction,
        edge_weights,
        pragmatic_potential,
        dt,
        lambda_value,
        sigma,
    )
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise FloatingPointError("graph diffusion step must remain finite and positive")
    return result


def optimal_geodesic_velocity(
    log_wavefunction_gradient,
    frame,
    lorentz_metric,
    lambda_value,
    sigma,
):
    """Project the log-wavefunction gradient and scale it per Section 4."""
    lambda_value, sigma = _validated_scale(lambda_value, sigma)
    gradient = np.asarray(log_wavefunction_gradient, dtype=FP64)
    frame = np.asarray(frame, dtype=FP64)
    metric = np.asarray(lorentz_metric, dtype=FP64)
    if gradient.ndim != 1 or gradient.size == 0 or not np.all(np.isfinite(gradient)):
        raise ValueError("log_wavefunction_gradient must be a non-empty finite vector")
    if frame.ndim != 2 or frame.shape[0] != gradient.size or frame.shape[1] == 0:
        raise ValueError("frame must have one row per gradient component and at least one column")
    if metric.shape != gradient.shape:
        raise ValueError("lorentz_metric must have one diagonal entry per gradient component")
    if not np.all(np.isfinite(frame)) or not np.all(np.isfinite(metric)):
        raise ValueError("frame and lorentz_metric must contain only finite values")

    with np.errstate(over="ignore", invalid="ignore"):
        projected_gradient = frame.T @ (metric * gradient)
    if not np.all(np.isfinite(projected_gradient)):
        raise ValueError("projected log-wavefunction gradient must be finite")
    velocity = cole_hopf_optimal_velocity(projected_gradient, lambda_value, sigma)
    if not np.all(np.isfinite(velocity)):
        raise FloatingPointError("optimal geodesic velocity must be finite")
    return velocity


__all__ = [
    "linearized_hjb_graph_step",
    "optimal_geodesic_velocity",
    "superpose_goal_wavefunctions",
    "value_to_wavefunction",
    "wavefunction_to_value",
]
