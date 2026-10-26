import numpy as np
from clean_core.config import D_SPATIAL, ETA_L, POLE, FP64, FP32
from clean_core.geometry.lorentz import get_boost_frame, exp_map, log_map, minkowski_norm


def thermodynamic_langevin_noise(
    dimension,
    temperature,
    dt,
    epsilon_0=1e-3,
    annealing=1.0,
    rng=None,
):
    if dimension < 1:
        raise ValueError("dimension must be positive")
    if not np.isfinite(temperature) or temperature < 0.0:
        raise ValueError("temperature must be finite and non-negative")
    if not np.isfinite(dt) or dt <= 0.0 or not np.isfinite(epsilon_0) or epsilon_0 <= 0.0:
        raise ValueError("dt and epsilon_0 must be finite and positive")
    if not np.isfinite(annealing) or annealing < 0.0:
        raise ValueError("annealing must be finite and non-negative")

    sigma = np.sqrt(2.0 * temperature * min(1.0, epsilon_0 / dt)) * 0.05 * annealing
    random_generator = np.random.default_rng() if rng is None else rng
    return random_generator.normal(0.0, sigma, size=dimension).astype(FP64)


def compute_maupertuis_conformal_drift(mu, w, grad_ln_omega):
    """Return conformal drift for an ambient tangent gradient at ``mu``."""
    E = get_boost_frame(mu)
    grad_ln_omega = np.asarray(grad_ln_omega, dtype=FP64)
    w = np.asarray(w, dtype=FP64)
    if grad_ln_omega.shape != mu.shape:
        raise ValueError(f"grad_ln_omega must have shape {mu.shape}")
    if w.shape != (E.shape[1],):
        raise ValueError(f"w must have shape ({E.shape[1]},)")
    internal_grad = E.T @ (ETA_L * grad_ln_omega)
    speed_sq = float(np.dot(w, w))
    return -speed_sq * internal_grad + 2.0 * np.dot(w, internal_grad) * w


def precision_weighted_spike_mask(error, precision, threshold, beta=5.0):
    error = np.asarray(error, dtype=FP64)
    precision = np.asarray(precision, dtype=FP64)
    if error.ndim != 1 or precision.shape != (error.size, error.size):
        raise ValueError("precision must be square and match the error vector")
    if not np.all(np.isfinite(error)) or not np.all(np.isfinite(precision)):
        raise ValueError("error and precision must contain only finite values")
    if not np.allclose(precision, precision.T, atol=1e-10):
        raise ValueError("precision must be symmetric")
    if not np.isfinite(threshold) or beta <= 0.0:
        raise ValueError("threshold must be finite and beta must be positive")

    energy = float(error @ precision @ error)
    logit = np.clip(beta * (energy - threshold), -60.0, 60.0)
    return float(1.0 / (1.0 + np.exp(-logit)))

def system1_fast_inference_step(
    neuron,
    parent_ctx_mu,
    barycenter_mu,
    alpha_damp=0.05,
    dt=0.01,
    noise_temperature=0.0,
    noise_annealing=1.0,
    rng=None,
):
    mu_curr = neuron.mu.copy()
    E = get_boost_frame(mu_curr)
    r = 2.0 * np.arcsinh(0.5 * minkowski_norm(mu_curr - POLE))
    f_drift_safe = 1.0 / (1.0 + r * np.tanh(r))
    log_parent = log_map(mu_curr, parent_ctx_mu)
    v_parent_internal = E.T @ (ETA_L * log_parent)
    w_raw = neuron.w + alpha_damp * f_drift_safe * v_parent_internal
    w_raw += thermodynamic_langevin_noise(
        D_SPATIAL,
        noise_temperature,
        dt,
        annealing=noise_annealing,
        rng=rng,
    )
    norm_w = np.linalg.norm(w_raw)
    w_speed = w_raw / max(1.0, norm_w / 3.0)
    neuron.w = w_speed.copy()
    v_tangent = E @ w_speed
    mu_next = exp_map(mu_curr, dt * v_tangent)
    prediction_error = log_map(mu_next, parent_ctx_mu)
    local_frame = get_boost_frame(mu_next)
    eps_tilde = local_frame.T @ (ETA_L * prediction_error)
    neuron.eps_tilde = eps_tilde.astype(FP32)
    neuron.mu = mu_next.copy()
    return mu_next, eps_tilde


def system1_query_relaxation(
    neurons,
    query_mu,
    barycenter_mu,
    steps=4,
    dt=0.01,
    alpha_damp=0.05,
):
    """Run inference-only relaxation; no weights or graph structure are trained."""
    if not neurons:
        raise ValueError("neurons must not be empty")
    if steps < 0 or dt <= 0.0:
        raise ValueError("steps must be non-negative and dt must be positive")

    errors = []
    for _ in range(steps):
        errors = []
        for neuron in neurons:
            _, error = system1_fast_inference_step(
                neuron,
                query_mu,
                barycenter_mu,
                alpha_damp=alpha_damp,
                dt=dt,
            )
            errors.append(error)
    return np.stack([neuron.mu.copy() for neuron in neurons]), errors
