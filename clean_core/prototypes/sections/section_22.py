"""Executable frame-noise prototype for Section 22."""

import numpy as np

from clean_core.config import D_LORENTZ, D_SPATIAL, ETA_L, FP64
from clean_core.core.inference import thermodynamic_langevin_noise
from clean_core.geometry.lorentz import get_boost_frame, minkowski_norm_sq


T_MAX = 1.0


def _finite_vector(value, shape, name):
    vector = np.asarray(value, dtype=FP64)
    if vector.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def thermodynamic_temperature(error, precision, alpha_temp):
    """Evaluate the bounded Section 22 temperature from local-frame error."""
    error = np.asarray(error, dtype=FP64)
    if error.shape != (D_SPATIAL,):
        raise ValueError(f"error must have shape ({D_SPATIAL},)")
    if not np.all(np.isfinite(error)):
        raise ValueError("error must contain only finite values")

    precision = np.asarray(precision, dtype=FP64)
    if precision.shape != (D_SPATIAL, D_SPATIAL):
        raise ValueError(
            f"precision must have shape ({D_SPATIAL}, {D_SPATIAL})"
        )
    if not np.all(np.isfinite(precision)):
        raise ValueError("precision must contain only finite values")
    if not np.allclose(precision, precision.T, atol=1e-10):
        raise ValueError("precision must be symmetric")

    alpha_temp = float(alpha_temp)
    if not np.isfinite(alpha_temp) or alpha_temp < 0.0:
        raise ValueError("alpha_temp must be finite and non-negative")

    energy = max(0.0, float(error @ precision @ error))
    return float(T_MAX * np.tanh((alpha_temp / T_MAX) * energy))


def thermodynamic_langevin_frame_noise(
    mu,
    error,
    precision,
    alpha_temp,
    dt,
    epsilon_0=1e-3,
    annealing=1.0,
    rng=None,
):
    """Sample Section 22 noise and lift it from the local tangent frame."""
    mu = _finite_vector(mu, (D_LORENTZ,), "mu")
    if mu[0] <= 0.0 or not np.isclose(minkowski_norm_sq(mu), -1.0, atol=1e-8):
        raise ValueError("mu must lie on the future unit hyperboloid")

    temperature = thermodynamic_temperature(error, precision, alpha_temp)
    local_noise = thermodynamic_langevin_noise(
        D_SPATIAL,
        temperature,
        dt,
        epsilon_0=epsilon_0,
        annealing=annealing,
        rng=rng,
    )
    frame_noise = get_boost_frame(mu) @ local_noise
    if not np.all(np.isfinite(frame_noise)):
        raise ValueError("frame noise must contain only finite values")
    return np.asarray(frame_noise, dtype=FP64)
