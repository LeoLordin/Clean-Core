import numpy as np

from clean_core.config import D_LORENTZ, ETA_L, FP64
from clean_core.geometry.lorentz import minkowski_dot


def project_tangent(mu, vector):
    mu = np.asarray(mu, dtype=FP64)
    vector = np.asarray(vector, dtype=FP64)
    if mu.shape != (D_LORENTZ,) or vector.shape != mu.shape:
        raise ValueError(f"mu and vector must have shape ({D_LORENTZ},)")
    return vector + minkowski_dot(mu, vector) * mu


def conformal_hamiltonian(mu, momentum, omega, pragmatic_potential, kappa_v=0.0):
    mu = np.asarray(mu, dtype=FP64)
    momentum = np.asarray(momentum, dtype=FP64)
    if mu.shape != (D_LORENTZ,) or momentum.shape != mu.shape:
        raise ValueError(f"mu and momentum must have shape ({D_LORENTZ},)")
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(momentum)):
        raise ValueError("state must contain only finite values")
    if not np.isfinite(omega) or omega <= 0.0 or not np.isfinite(kappa_v):
        raise ValueError("omega must be positive and kappa_v finite")
    if not np.isfinite(pragmatic_potential):
        raise ValueError("pragmatic_potential must be finite")
    kinetic_metric = minkowski_dot(momentum, momentum)
    if kinetic_metric < -1e-10:
        raise ValueError("momentum must be space-like or zero and tangent")
    kinetic_scale = 1.0 + kappa_v
    return float(kinetic_scale * kinetic_metric / (2.0 * omega**2) + pragmatic_potential)


def conformal_hamiltonian_vector_field(
    mu,
    momentum,
    omega,
    gradient_potential,
    gradient_omega,
    kappa_v=0.0,
):
    mu = np.asarray(mu, dtype=FP64)
    momentum = project_tangent(mu, momentum)
    gradient_potential = project_tangent(mu, gradient_potential)
    gradient_omega = project_tangent(mu, gradient_omega)
    if not np.isfinite(omega) or omega <= 0.0 or not np.isfinite(kappa_v):
        raise ValueError("omega must be positive and kappa_v finite")

    kinetic_scale = 1.0 + kappa_v
    momentum_norm_sq = max(0.0, minkowski_dot(momentum, momentum))
    mu_dot = kinetic_scale * momentum / omega**2
    p_dot = -gradient_potential + kinetic_scale * momentum_norm_sq * gradient_omega / omega**3
    normal = ETA_L * mu
    normal_norm = minkowski_dot(mu, normal)
    constraint_multiplier = -minkowski_dot(mu_dot, momentum) / normal_norm
    p_dot = p_dot + constraint_multiplier * normal
    return mu_dot, p_dot, float(constraint_multiplier)