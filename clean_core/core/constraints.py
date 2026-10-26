import numpy as np


def projected_phr_multiplier_update(
    multiplier,
    constraint_value,
    limit,
    rho=1.0,
    eta_lambda=0.01,
    dt=0.01,
    gamma_prox=0.1,
):
    """Projected PHR multiplier update for an inequality constraint ``c <= limit``."""
    values = np.asarray([multiplier, constraint_value, limit, rho, eta_lambda, dt, gamma_prox], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("multiplier update inputs must be finite")
    if multiplier < 0.0 or rho <= 0.0 or eta_lambda < 0.0 or dt < 0.0 or gamma_prox < 0.0:
        raise ValueError("multiplier/rates must be non-negative and rho must be positive")

    violation = constraint_value - limit
    proximal_violation = max(-multiplier / rho, violation)
    step = dt * eta_lambda / (1.0 + gamma_prox * dt)
    return float(max(0.0, multiplier + step * proximal_violation))