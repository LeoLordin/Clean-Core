"""Finite-dimensional resolvent prototype for Section 19.

This implements the affine ego fixed point and its adjoint-resolvent parameter
updates. It does not implement the hyperbolic maps or establish any Gödelian
claim.
"""

from dataclasses import dataclass

import numpy as np

from clean_core.config import FP64


@dataclass(frozen=True)
class EgoReflectionStep:
    """Values produced by one finite-dimensional ego-reflection update."""

    v_self_star: np.ndarray
    W_self_v: np.ndarray
    W_self_hist: np.ndarray
    b_self: np.ndarray


def _finite_scalar(value, name):
    try:
        raw = np.asarray(value)
        if raw.ndim != 0 or np.iscomplexobj(raw):
            raise ValueError
        scalar = float(raw)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite scalar") from error
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _finite_array(value, shape, name):
    try:
        raw = np.asarray(value)
        if np.iscomplexobj(raw):
            raise ValueError
        array = np.asarray(value, dtype=FP64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite array with shape {shape}") from error
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _regularization_parameters(delta_reg, epsilon_gap):
    delta_reg = _finite_scalar(delta_reg, "delta_reg")
    epsilon_gap = _finite_scalar(epsilon_gap, "epsilon_gap")
    if delta_reg <= 0.0:
        raise ValueError("delta_reg must be positive")
    if not 0.0 < epsilon_gap < 1.0:
        raise ValueError("epsilon_gap must be between zero and one")
    return delta_reg, epsilon_gap


def _clip_spectral_norm(matrix, cap):
    scale = float(np.max(np.abs(matrix)))
    if scale == 0.0:
        return matrix.copy()
    normalized = matrix / scale
    norm = float(np.linalg.norm(normalized, ord=2))
    if scale <= cap / norm:
        return matrix.copy()
    return normalized * (cap / norm)


def _bounded_resolvent(W_self_v, delta_reg, epsilon_gap):
    try:
        candidate = np.asarray(W_self_v)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("W_self_v must be a non-empty square matrix") from error
    if (
        candidate.ndim != 2
        or candidate.shape[0] == 0
        or candidate.shape[0] != candidate.shape[1]
    ):
        raise ValueError("W_self_v must be a non-empty square matrix")
    matrix = _finite_array(candidate, candidate.shape, "W_self_v")
    delta_reg, epsilon_gap = _regularization_parameters(delta_reg, epsilon_gap)
    cap = 1.0 - epsilon_gap
    bounded_matrix = _clip_spectral_norm(matrix, cap)
    system = (1.0 + delta_reg) * np.eye(matrix.shape[0], dtype=FP64) - bounded_matrix
    resolvent = np.linalg.solve(system, np.eye(matrix.shape[0], dtype=FP64))
    if not np.all(np.isfinite(resolvent)):
        raise ValueError("regularized resolvent must be finite")
    return bounded_matrix, resolvent


def regularized_resolvent(W_self_v, delta_reg=0.1, epsilon_gap=1e-3):
    """Return ``((1 + delta_reg) I - W_self_v)^-1`` after spectral clipping.

    Clipping ``W_self_v`` to spectral norm at most ``1 - epsilon_gap`` gives
    the resolvent the operator-norm bound ``1 / (delta_reg + epsilon_gap)``.
    """
    _, resolvent = _bounded_resolvent(W_self_v, delta_reg, epsilon_gap)
    return resolvent


def ego_reflection_step(
    W_self_v,
    W_self_hist,
    feat_hist,
    b_self,
    g_tilde,
    *,
    delta_reg=0.1,
    dt=1.0,
    eta_self=1e-2,
    epsilon_gap=1e-3,
):
    """Compute the fixed point and one adjoint-resolvent parameter update."""
    try:
        W_self_v_input = np.asarray(W_self_v)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("W_self_v must be a non-empty square matrix") from error
    if (
        W_self_v_input.ndim != 2
        or W_self_v_input.shape[0] == 0
        or W_self_v_input.shape[0] != W_self_v_input.shape[1]
    ):
        raise ValueError("W_self_v must be a non-empty square matrix")
    dimension = W_self_v_input.shape[0]
    W_self_v = _finite_array(
        W_self_v_input, (dimension, dimension), "W_self_v"
    )

    try:
        W_self_hist_input = np.asarray(W_self_hist)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "W_self_hist must have one row per self dimension and at least one column"
        ) from error
    if (
        W_self_hist_input.ndim != 2
        or W_self_hist_input.shape[0] != dimension
        or W_self_hist_input.shape[1] == 0
    ):
        raise ValueError("W_self_hist must have one row per self dimension and at least one column")
    history_dimension = W_self_hist_input.shape[1]
    W_self_hist = _finite_array(
        W_self_hist_input, (dimension, history_dimension), "W_self_hist"
    )
    feat_hist = _finite_array(feat_hist, (history_dimension,), "feat_hist")
    b_self = _finite_array(b_self, (dimension,), "b_self")
    g_tilde = _finite_array(g_tilde, (dimension,), "g_tilde")
    delta_reg, epsilon_gap = _regularization_parameters(delta_reg, epsilon_gap)
    dt = _finite_scalar(dt, "dt")
    eta_self = _finite_scalar(eta_self, "eta_self")
    if dt < 0.0:
        raise ValueError("dt must be non-negative")
    if eta_self < 0.0:
        raise ValueError("eta_self must be non-negative")

    bounded_W_self_v, resolvent = _bounded_resolvent(
        W_self_v, delta_reg, epsilon_gap
    )
    v_self_star = resolvent @ (W_self_hist @ feat_hist + b_self)
    adjoint = np.linalg.solve(
        ((1.0 + delta_reg) * np.eye(dimension, dtype=FP64) - bounded_W_self_v).T,
        g_tilde,
    )
    step_size = dt * eta_self
    if not np.isfinite(step_size):
        raise ValueError("dt * eta_self must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        W_self_v_candidate = bounded_W_self_v - step_size * np.outer(
            adjoint, v_self_star
        )
        W_self_hist_next = W_self_hist - step_size * np.outer(adjoint, feat_hist)
        b_self_next = b_self - step_size * adjoint
    if not all(
        np.all(np.isfinite(array))
        for array in (W_self_v_candidate, W_self_hist_next, b_self_next)
    ):
        raise ValueError("ego-reflection update must remain finite")
    W_self_v_next = _clip_spectral_norm(
        W_self_v_candidate, 1.0 - epsilon_gap
    )
    result_arrays = (
        v_self_star,
        W_self_v_next,
        W_self_hist_next,
        b_self_next,
    )
    if not all(np.all(np.isfinite(array)) for array in result_arrays):
        raise ValueError("ego-reflection update must remain finite")
    return EgoReflectionStep(*result_arrays)


__all__ = ["EgoReflectionStep", "ego_reflection_step", "regularized_resolvent"]
