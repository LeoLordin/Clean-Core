import numpy as np
from clean_core.config import FP64
from clean_core.geometry.lorentz import minkowski_dot, exp_map

def implicit_midpoint_step(mu, p, grad_F, dt=0.01):
    p_mid = p - 0.5 * dt * grad_F
    v_mid = p_mid - minkowski_dot(mu, p_mid) * mu
    mu_next = exp_map(mu, dt * v_mid)
    mu_next[0] = np.sqrt(1.0 + np.sum(mu_next[1:]**2))
    p_next = p_mid - 0.5 * dt * grad_F
    mu_mid_dot = minkowski_dot(mu_next, p_next)
    p_next = p_next - mu_mid_dot * mu_next
    return mu_next, p_next
