"""Constrained finite-step prototype for Section 32 Hamiltonian mechanics."""

from collections.abc import Callable

import numpy as np

from clean_core.config import D_LORENTZ, ETA_L, FP64
from clean_core.geometry.hamiltonian import conformal_hamiltonian_vector_field
from clean_core.geometry.lorentz import minkowski_dot


def _validated_state(mu, momentum):
    mu = np.asarray(mu, dtype=FP64)
    momentum = np.asarray(momentum, dtype=FP64)
    expected_shape = (D_LORENTZ,)
    if mu.shape != expected_shape or momentum.shape != expected_shape:
        raise ValueError(f"mu and momentum must have shape {expected_shape}")
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(momentum)):
        raise ValueError("state must contain only finite values")
    if mu[0] <= 0.0 or not np.isclose(
        minkowski_dot(mu, mu), -1.0, rtol=1e-8, atol=1e-8
    ):
        raise ValueError("mu must lie on the future unit Lorentz hyperboloid")
    tangent_residual = minkowski_dot(mu, momentum)
    tangent_tolerance = 1e-8 * max(
        1.0, float(np.linalg.norm(mu) * np.linalg.norm(momentum))
    )
    if abs(tangent_residual) > tangent_tolerance:
        raise ValueError("momentum must be cotangent-tangent to mu")
    return mu, momentum


def section_32_vector_field(
    mu,
    momentum,
    omega,
    gradient_potential,
    gradient_omega,
    kappa_v=0.0,
):
    """Return ``(mu_dot, momentum_dot, lambda)`` for a constrained state.

    Momentum and gradients use ambient Lorentz-vector components.  The
    constraint multiplier makes the vector field tangent to the cotangent
    constraint at the supplied state.
    """
    mu, momentum = _validated_state(mu, momentum)
    gradient_potential = np.asarray(gradient_potential, dtype=FP64)
    gradient_omega = np.asarray(gradient_omega, dtype=FP64)
    if (
        gradient_potential.shape != mu.shape
        or gradient_omega.shape != mu.shape
        or not np.all(np.isfinite(gradient_potential))
        or not np.all(np.isfinite(gradient_omega))
    ):
        raise ValueError("gradients must be finite Lorentz vectors")

    return conformal_hamiltonian_vector_field(
        mu,
        momentum,
        omega,
        gradient_potential,
        gradient_omega,
        kappa_v=kappa_v,
    )


def _evaluate_at(value, mu):
    return value(mu.copy()) if isinstance(value, Callable) else value


def _retract_state(mu, momentum):
    norm_sq = minkowski_dot(mu, mu)
    if mu[0] <= 0.0 or not np.isfinite(norm_sq) or norm_sq >= 0.0:
        raise ValueError("finite step left the future timelike region")
    mu = mu / np.sqrt(-norm_sq)
    momentum = momentum + minkowski_dot(mu, momentum) * mu
    return mu, momentum


def section_32_step(
    mu,
    momentum,
    dt,
    omega,
    gradient_potential,
    gradient_omega,
    kappa_v=0.0,
):
    """Advance one projected explicit-midpoint step.

    ``omega``, ``gradient_potential``, and ``gradient_omega`` may be fixed
    values or callables evaluated as ``value(mu)``.  The endpoint is retracted
    onto the hyperboloid and its cotangent space; this finite-step prototype
    does not claim exact symplecticity or exact energy conservation.
    """
    mu, momentum = _validated_state(mu, momentum)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")

    initial_field = section_32_vector_field(
        mu,
        momentum,
        _evaluate_at(omega, mu),
        _evaluate_at(gradient_potential, mu),
        _evaluate_at(gradient_omega, mu),
        kappa_v=kappa_v,
    )
    mu_dot, momentum_dot, _ = initial_field
    midpoint_mu, midpoint_momentum = _retract_state(
        mu + 0.5 * dt * mu_dot,
        momentum + 0.5 * dt * momentum_dot,
    )

    midpoint_field = section_32_vector_field(
        midpoint_mu,
        midpoint_momentum,
        _evaluate_at(omega, midpoint_mu),
        _evaluate_at(gradient_potential, midpoint_mu),
        _evaluate_at(gradient_omega, midpoint_mu),
        kappa_v=kappa_v,
    )
    midpoint_mu_dot, midpoint_momentum_dot, _ = midpoint_field
    return _retract_state(
        mu + dt * midpoint_mu_dot,
        momentum + dt * midpoint_momentum_dot,
    )


__all__ = ["section_32_vector_field", "section_32_step"]
