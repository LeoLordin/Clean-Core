"""Bounded numerical prototype for selected Section 2 conformal equations.

This evaluates the stated conformal factor, metric, connection correction, and
local-frame drift for supplied inputs. It is not a complete geometric proof.
"""

import numpy as np

from clean_core.config import D_LORENTZ, D_SPATIAL, ETA_L, FP64
from clean_core.core.inference import compute_maupertuis_conformal_drift
from clean_core.geometry.lorentz import minkowski_dot, minkowski_norm_sq


def _finite_scalar(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _finite_vector(value, shape, name):
    try:
        vector = np.asarray(value, dtype=FP64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if vector.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _conformal_parameters(threshold, delta_eps):
    threshold = _finite_scalar(threshold, "threshold")
    delta_eps = _finite_scalar(delta_eps, "delta_eps")
    if delta_eps <= 0.0:
        raise ValueError("delta_eps must be positive")
    return threshold, delta_eps


def conformal_factor(pragmatic_potential, threshold, delta_eps=1e-6):
    """Evaluate Omega=max(delta_eps, threshold-pragmatic_potential)."""
    pragmatic_potential = _finite_scalar(pragmatic_potential, "pragmatic_potential")
    threshold, delta_eps = _conformal_parameters(threshold, delta_eps)
    raw_factor = threshold - pragmatic_potential
    if not np.isfinite(raw_factor):
        raise ValueError("conformal factor must be finite")
    return float(max(delta_eps, raw_factor))


def conformal_metric(omega):
    """Return the conformal metric and its inverse in Lorentz coordinates."""
    omega = _finite_scalar(omega, "omega")
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        omega_array = np.float64(omega)
        metric = (omega_array**2) * np.diag(ETA_L)
        inverse_metric = (omega_array**-2) * np.diag(ETA_L)
    if not np.all(np.isfinite(metric)) or not np.all(np.isfinite(inverse_metric)):
        raise ValueError("conformal metric and inverse must be finite")
    return metric, inverse_metric


def conformal_connection(gradient_ln_omega, background_connection=None):
    """Apply the stated Levi-Civita conformal connection transformation."""
    gradient = _finite_vector(
        gradient_ln_omega, (D_LORENTZ,), "gradient_ln_omega"
    )
    if background_connection is None:
        connection = np.zeros((D_LORENTZ, D_LORENTZ, D_LORENTZ), dtype=FP64)
    else:
        try:
            connection = np.asarray(background_connection, dtype=FP64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("background_connection must be a finite numeric array") from exc
        if connection.shape != (D_LORENTZ, D_LORENTZ, D_LORENTZ):
            raise ValueError(
                "background_connection must have shape "
                f"({D_LORENTZ}, {D_LORENTZ}, {D_LORENTZ})"
            )
        if not np.all(np.isfinite(connection)):
            raise ValueError("background_connection must contain only finite values")

    identity = np.eye(D_LORENTZ, dtype=FP64)
    correction = (
        np.einsum("ab,c->abc", identity, gradient)
        + np.einsum("ac,b->abc", identity, gradient)
        - np.einsum("bc,a->abc", np.diag(ETA_L), ETA_L * gradient)
    )
    transformed = connection + correction
    if not np.all(np.isfinite(transformed)):
        raise ValueError("transformed connection must contain only finite values")
    return transformed


def conformal_drift(
    mu,
    w,
    pragmatic_potential,
    gradient_pragmatic_potential,
    threshold,
    delta_eps=1e-6,
):
    """Evaluate the Section 2 tetrad drift for a tangent potential gradient."""
    mu = _finite_vector(mu, (D_LORENTZ,), "mu")
    w = _finite_vector(w, (D_SPATIAL,), "w")
    gradient = _finite_vector(
        gradient_pragmatic_potential,
        (D_LORENTZ,),
        "gradient_pragmatic_potential",
    )
    if mu[0] <= 0.0 or not np.isclose(minkowski_norm_sq(mu), -1.0, atol=1e-8):
        raise ValueError("mu must lie on the future unit hyperboloid")
    tangent_scale = max(1.0, float(np.linalg.norm(gradient)))
    if abs(minkowski_dot(mu, gradient)) > 1e-8 * tangent_scale:
        raise ValueError("gradient_pragmatic_potential must be tangent at mu")

    pragmatic_potential = _finite_scalar(
        pragmatic_potential, "pragmatic_potential"
    )
    threshold, delta_eps = _conformal_parameters(threshold, delta_eps)
    raw_factor = threshold - pragmatic_potential
    if not np.isfinite(raw_factor):
        raise ValueError("conformal factor must be finite")
    omega = max(delta_eps, raw_factor)
    # The floor is constant on this branch; at the kink use zero gradient.
    if raw_factor > delta_eps:
        gradient_ln_omega = -gradient / omega
    else:
        gradient_ln_omega = np.zeros_like(gradient)
    drift = compute_maupertuis_conformal_drift(mu, w, gradient_ln_omega)
    if drift.shape != (D_SPATIAL,) or not np.all(np.isfinite(drift)):
        raise ValueError("conformal drift must be a finite spatial vector")
    return drift
