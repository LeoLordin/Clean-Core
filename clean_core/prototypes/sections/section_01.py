"""Bounded stochastic Lorentz-manifold step from theory formula Section 1.

This prototype implements the local-frame DLR precision, adaptive scalar
precision, temperature-scaled Itô correction, semi-implicit velocity solve,
Langevin noise, speed clamp, and one geodesic update. The caller supplies the
already assembled local force; it does not implement the full theory or its
other timescales.
"""

from dataclasses import dataclass

import numpy as np

from clean_core.config import D_LORENTZ, D_SPATIAL, ETA_L, FP64
from clean_core.geometry.lorentz import (
    exp_map,
    get_boost_frame,
    minkowski_norm_sq,
)


@dataclass(frozen=True)
class LorentzStepResult:
    """Inspectable outputs from one stochastic manifold step."""

    mu: np.ndarray
    velocity: np.ndarray
    raw_velocity: np.ndarray
    local_error: np.ndarray
    precision: np.ndarray
    adaptive_precision: float
    ito_drift: np.ndarray


def _finite_array(name, value, shape=None):
    try:
        array = np.asarray(value, dtype=FP64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _positive_scalar(name, value, *, allow_zero=False):
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    lower_bound_fails = scalar < 0.0 if allow_zero else scalar <= 0.0
    if not np.isfinite(scalar) or lower_bound_fails:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return scalar


def stochastic_lie_euler_maruyama_step(
    mu,
    prediction_error,
    force,
    d_diag,
    low_rank_basis,
    low_rank_values,
    *,
    dt,
    temperature,
    sigma_beta=1.0,
    alpha_temp=1.0,
    temperature_max=1.0,
    epsilon_0=1e-3,
    precision_min=1e-3,
    precision_max=1e3,
    max_speed=5.0,
    annealing=1.0,
    delta_eps=1e-12,
    rng=None,
):
    """Advance one state using the bounded Section 1 inference sub-equations.

    ``prediction_error`` is ambient Lorentz-space error and ``force`` is the
    fully assembled force in the orthonormal local frame. DLR arguments encode
    covariance ``diag(d_diag) + V diag(low_rank_values) V.T``. The generated
    local velocity is clamped and used for one exponential-map step (the
    velocity equation already includes its stated ``dt`` factors).
    """
    mu = _finite_array("mu", mu, (D_LORENTZ,))
    prediction_error = _finite_array(
        "prediction_error", prediction_error, (D_LORENTZ,)
    )
    force = _finite_array("force", force, (D_SPATIAL,))
    d_diag = _finite_array("d_diag", d_diag, (D_SPATIAL,))
    low_rank_basis = _finite_array("low_rank_basis", low_rank_basis)
    low_rank_values = _finite_array("low_rank_values", low_rank_values)

    if low_rank_basis.ndim != 2 or low_rank_basis.shape[0] != D_SPATIAL:
        raise ValueError(f"low_rank_basis must have shape ({D_SPATIAL}, rank)")
    rank = low_rank_basis.shape[1]
    if rank < 1 or low_rank_values.shape != (rank,):
        raise ValueError("low_rank_values must match a non-empty basis rank")
    if np.any(d_diag <= 0.0):
        raise ValueError("d_diag entries must be positive")
    if np.any(low_rank_values <= 0.0):
        raise ValueError("low_rank_values entries must be positive")
    if mu[0] <= 0.0 or not np.isclose(
        minkowski_norm_sq(mu), -1.0, rtol=1e-9, atol=1e-9
    ):
        raise ValueError("mu must lie on the future unit Lorentz hyperboloid")

    dt = _positive_scalar("dt", dt)
    temperature = _positive_scalar("temperature", temperature, allow_zero=True)
    sigma_beta = _positive_scalar("sigma_beta", sigma_beta, allow_zero=True)
    alpha_temp = _positive_scalar("alpha_temp", alpha_temp, allow_zero=True)
    temperature_max = _positive_scalar("temperature_max", temperature_max)
    epsilon_0 = _positive_scalar("epsilon_0", epsilon_0)
    precision_min = _positive_scalar("precision_min", precision_min)
    precision_max = _positive_scalar("precision_max", precision_max)
    max_speed = _positive_scalar("max_speed", max_speed)
    annealing = _positive_scalar("annealing", annealing, allow_zero=True)
    delta_eps = _positive_scalar("delta_eps", delta_eps)
    if precision_min > precision_max:
        raise ValueError("precision_min must not exceed precision_max")
    if rng is not None and not callable(getattr(rng, "normal", None)):
        raise ValueError("rng must provide a normal method")

    # Woodbury inversion of D + V diag(s) V.T.
    d_inv = 1.0 / d_diag
    d_inv_v = d_inv[:, None] * low_rank_basis
    middle = np.diag(1.0 / low_rank_values) + low_rank_basis.T @ d_inv_v
    precision = np.diag(d_inv) - d_inv_v @ np.linalg.solve(
        middle, d_inv_v.T
    )

    frame = get_boost_frame(mu)
    local_error = frame.T @ (ETA_L * prediction_error)
    error_energy = max(0.0, float(local_error @ precision @ local_error))
    adaptive_precision = float(
        np.clip(
            D_SPATIAL / max(delta_eps, error_energy),
            precision_min,
            precision_max,
        )
    )

    sech_squared = 1.0 - np.tanh(
        (alpha_temp / temperature_max) * error_energy
    ) ** 2
    ito_drift = (
        -0.5
        * alpha_temp
        * temperature
        * sech_squared
        * (precision @ local_error)
    )

    system = (
        np.eye(D_SPATIAL, dtype=FP64)
        + (dt / epsilon_0) * adaptive_precision * sigma_beta * precision
    )
    deterministic_velocity = np.linalg.solve(
        system, (dt / epsilon_0) * force + dt * ito_drift
    )
    noise_scale = (
        np.sqrt(2.0 * temperature * min(1.0, epsilon_0 / dt))
        * 0.05
        * annealing
    )
    generator = np.random.default_rng() if rng is None else rng
    raw_velocity = deterministic_velocity + generator.normal(
        0.0, noise_scale, size=D_SPATIAL
    )
    speed = float(np.linalg.norm(raw_velocity))
    velocity = raw_velocity / max(1.0, speed / max_speed)

    if not np.all(np.isfinite(velocity)) or np.linalg.norm(velocity) > 350.0:
        raise ValueError("clamped velocity is outside the finite exp-map range")
    mu_next = exp_map(mu.copy(), frame @ velocity)
    if (
        not np.all(np.isfinite(mu_next))
        or mu_next[0] <= 0.0
        or not np.isclose(minkowski_norm_sq(mu_next), -1.0, rtol=1e-9, atol=1e-9)
    ):
        raise FloatingPointError("manifold step produced an invalid Lorentz point")

    return LorentzStepResult(
        mu=mu_next,
        velocity=velocity,
        raw_velocity=raw_velocity,
        local_error=local_error,
        precision=precision,
        adaptive_precision=adaptive_precision,
        ito_drift=ito_drift,
    )
