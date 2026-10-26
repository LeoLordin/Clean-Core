"""Standalone NumPy prototype for Section 29's spike encoder.

The surrogate helper represents only the specified derivative with respect to
spike probability; this prototype does not implement hardware STE training.
"""

import numpy as np

from clean_core.config import D_SPATIAL, ETA_L, FP64, POLE
from clean_core.geometry.lorentz import get_boost_frame, log_map


def section_29_spike_probabilities(mu, W_enc, b_enc, f_max, dt):
    """Return per-crossbar spike probabilities for one Lorentz microstate.

    The microstate is mapped from the hyperboloid origin into its spatial
    tangent coordinates before applying the learned encoder projection.
    """
    mu = np.asarray(mu, dtype=FP64)
    weights = np.asarray(W_enc, dtype=FP64)
    bias = np.asarray(b_enc, dtype=FP64)

    if mu.shape != POLE.shape or not np.all(np.isfinite(mu)):
        raise ValueError(f"mu must be a finite Lorentz vector with shape {POLE.shape}")
    minkowski_norm_sq = float(np.dot(mu * ETA_L, mu))
    if mu[0] <= 0.0 or not np.isclose(minkowski_norm_sq, -1.0, atol=1e-8):
        raise ValueError("mu must lie on the future unit Lorentz hyperboloid")
    if weights.ndim != 2 or weights.shape[0] == 0 or weights.shape[1] != D_SPATIAL:
        raise ValueError(f"W_enc must have shape (M > 0, {D_SPATIAL})")
    if bias.shape != (weights.shape[0],):
        raise ValueError("b_enc must have one value per row of W_enc")
    if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(bias)):
        raise ValueError("W_enc and b_enc must contain only finite values")
    if not np.isfinite(f_max) or f_max <= 0.0:
        raise ValueError("f_max must be finite and positive")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")

    origin_frame = get_boost_frame(POLE)
    tangent_features = origin_frame.T @ (ETA_L * log_map(POLE, mu))
    logits = weights @ tangent_features + bias
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    rates = float(f_max) * sigmoid
    with np.errstate(over="ignore"):
        probabilities = -np.expm1(-rates * float(dt))
    return probabilities


def section_29_spike_encoder(
    mu, W_enc, b_enc, f_max, dt, uniform=None, rng=None
):
    """Encode ``mu`` as binary crossbar spikes using the Section 29 threshold.

    Pass ``uniform`` for deterministic evaluation; otherwise values are drawn
    from ``rng`` (or a new NumPy generator). The returned spikes are uint8.
    """
    if uniform is not None and rng is not None:
        raise ValueError("provide either uniform or rng, not both")

    probabilities = section_29_spike_probabilities(mu, W_enc, b_enc, f_max, dt)
    if uniform is None:
        random_generator = np.random.default_rng() if rng is None else rng
        uniform = random_generator.random(probabilities.shape)
    else:
        uniform = np.asarray(uniform, dtype=FP64)
        if uniform.shape != probabilities.shape:
            raise ValueError("uniform must have one value per spike probability")
        if not np.all(np.isfinite(uniform)) or np.any(uniform < 0.0) or np.any(uniform > 1.0):
            raise ValueError("uniform values must be finite and in [0, 1]")

    return (probabilities > uniform).astype(np.uint8)


def section_29_ste_surrogate_gradient(upstream_gradient):
    """Apply Section 29's ``d spike / d probability = 1`` surrogate derivative.

    Section 29's ``-sg[p] + p`` terms make the backward derivative of the
    thresholded spike equal to one, so the incoming gradient passes through.
    """
    gradient = np.asarray(upstream_gradient, dtype=FP64)
    if not np.all(np.isfinite(gradient)):
        raise ValueError("upstream_gradient must contain only finite values")
    return gradient.copy()
