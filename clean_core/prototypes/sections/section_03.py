"""Bounded numerical prototype for Section 3's constrained midpoint update.

The section states two multipliers but displays only one multiplier term.
This prototype includes a scalar normal-drift correction so both listed
endpoint constraints can be solved; it is not a proof of symplecticity.
"""

import numpy as np
from scipy.optimize import root

from clean_core.config import D_LORENTZ, ETA_L, FP64
from clean_core.geometry.lorentz import minkowski_dot, minkowski_norm_sq


def _finite_vector(value, name):
    vector = np.asarray(value, dtype=FP64)
    if vector.shape != (D_LORENTZ,):
        raise ValueError(f"{name} must have shape ({D_LORENTZ},)")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def implicit_midpoint_step(
    mu,
    p,
    grad_F,
    dt=0.01,
    *,
    tolerance=1e-10,
    max_iterations=200,
):
    """Solve one constrained implicit midpoint step on the Lorentz hyperboloid.

    ``grad_F`` is called as ``grad_F(mu_mid)``. The returned phase point
    satisfies the unit-hyperboloid and cotangent constraints within
    ``tolerance``. The additional normal-drift multiplier completes the
    section's stated two-multiplier constraint solve; symplecticity is not
    established by this bounded prototype.
    """
    mu = _finite_vector(mu, "mu")
    p = _finite_vector(p, "p")
    dt = float(dt)
    tolerance = float(tolerance)
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("dt must be finite and non-negative")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, (int, np.integer)):
        raise ValueError("max_iterations must be a positive integer")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")
    if not callable(grad_F):
        raise ValueError("grad_F must be callable")

    mu_scale = max(1.0, float(np.linalg.norm(mu)) ** 2)
    if mu[0] <= 0.0 or not np.isclose(
        minkowski_norm_sq(mu), -1.0, atol=1e-8 * mu_scale, rtol=0.0
    ):
        raise ValueError("mu must lie on the future unit hyperboloid")
    if abs(minkowski_dot(mu, p)) > 1e-8 * max(1.0, float(np.linalg.norm(p))):
        raise ValueError("p must be tangent to the hyperboloid at mu")
    if dt == 0.0:
        return mu.copy(), p.copy()

    def residual(unknowns):
        mu_next = unknowns[:D_LORENTZ]
        p_next = unknowns[D_LORENTZ : 2 * D_LORENTZ]
        lambda_mid, drift_multiplier = unknowns[-2:]
        mu_mid = 0.5 * (mu + mu_next)
        p_mid = 0.5 * (p + p_next)
        gradient = _finite_vector(grad_F(mu_mid.copy()), "grad_F(mu_mid)")

        momentum_residual = (
            p_next - p + dt * (gradient + lambda_mid * ETA_L * mu_mid)
        )
        position_residual = (
            mu_next - mu - dt * ETA_L * p_mid - drift_multiplier * ETA_L * mu_mid
        )
        constraints = np.array(
            [
                minkowski_norm_sq(mu_next) + 1.0,
                minkowski_dot(mu_next, p_next),
            ],
            dtype=FP64,
        )
        return np.concatenate((momentum_residual, position_residual, constraints))

    initial_mu_next = mu + dt * ETA_L * p
    initial_gradient = _finite_vector(grad_F(mu.copy()), "grad_F(mu)")
    initial_p_next = p - dt * initial_gradient
    initial = np.concatenate((initial_mu_next, initial_p_next, [0.0, 0.0]))
    solution = root(
        residual,
        initial,
        method="hybr",
        options={"maxfev": int(max_iterations)},
        tol=tolerance,
    )
    final_residual = residual(solution.x)
    if not np.all(np.isfinite(solution.x)) or np.linalg.norm(final_residual, ord=np.inf) > max(
        1e-8, 10.0 * tolerance
    ):
        raise RuntimeError(
            "constrained midpoint solve did not converge "
            f"(residual={np.linalg.norm(final_residual, ord=np.inf):.3g})"
        )

    mu_next = solution.x[:D_LORENTZ].copy()
    p_next = solution.x[D_LORENTZ : 2 * D_LORENTZ].copy()
    if mu_next[0] <= 0.0:
        raise RuntimeError("constrained midpoint solve left the future hyperboloid sheet")
    return mu_next, p_next


__all__ = ["implicit_midpoint_step"]
