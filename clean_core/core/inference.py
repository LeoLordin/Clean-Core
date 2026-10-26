import numpy as np
from clean_core.config import ETA_L, POLE, FP64, FP32
from clean_core.geometry.lorentz import get_boost_frame, exp_map, log_map, minkowski_norm

def compute_maupertuis_conformal_drift(mu, w, grad_ln_omega):
    E = get_boost_frame(mu)
    internal_grad = E.T @ (ETA_L * np.pad(grad_ln_omega, (1, 0)))
    speed_sq = float(np.dot(w, w))
    return -speed_sq * internal_grad + 2.0 * np.dot(w, internal_grad) * w

def system1_fast_inference_step(neuron, parent_ctx_mu, barycenter_mu, alpha_damp=0.05, dt=0.01):
    mu_curr = neuron.mu.copy()
    E = get_boost_frame(mu_curr)
    r = 2.0 * np.arcsinh(0.5 * minkowski_norm(mu_curr - POLE))
    f_drift_safe = 1.0 / (1.0 + r * np.tanh(r))
    log_parent = log_map(mu_curr, parent_ctx_mu)
    v_parent_internal = E.T @ (ETA_L * log_parent)
    w_raw = neuron.w + alpha_damp * f_drift_safe * v_parent_internal
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
