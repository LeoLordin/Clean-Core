"""Finite-dimensional regularized ego-reflection prototype for Algorithm 6.

Historical features are supplied as an explicit finite array (including any
detached coordinates and motor features). This composes Section 19's
adjoint-resolvent update with validated Lorentz maps; it does not establish a
formal Gödel self-proof.
"""

from dataclasses import dataclass

import numpy as np

from clean_core.config import D_LORENTZ, D_SPATIAL, ETA_L, FP64
from clean_core.geometry.lorentz import get_boost_frame
from clean_core.prototypes.sections.section_14 import (
    geodesic_exp,
    geodesic_log,
    parallel_transport,
)
from clean_core.prototypes.sections.section_19 import (
    ego_reflection_step,
    regularized_resolvent,
)


@dataclass(frozen=True)
class Algorithm06Step:
    """Finite-array values produced by one ego-reflection step."""

    mu_self_star: np.ndarray
    introspection: np.ndarray
    gradient_self: np.ndarray
    gradient_tilde: np.ndarray
    W_self_v: np.ndarray
    W_self_hist: np.ndarray
    b_self: np.ndarray


def _finite_scalar(value, name):
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite scalar") from error
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _finite_array(value, shape, name):
    try:
        array = np.asarray(value, dtype=FP64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite array with shape {shape}") from error
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _hyperboloid_point(value, name):
    point = _finite_array(value, (D_LORENTZ,), name)
    norm_sq = float(np.sum(point * ETA_L * point))
    if (
        point[0] <= 0.0
        or not np.isfinite(norm_sq)
        or not np.isclose(norm_sq, -1.0, rtol=1e-9, atol=1e-9)
    ):
        raise ValueError(f"{name} must lie on the future unit Lorentz hyperboloid")
    return point.copy()


def algorithm_06_step(
    W_self_v,
    W_self_hist,
    feat_hist,
    b_self,
    barycenter,
    introspection,
    *,
    delta_reg=0.1,
    dt=1.0,
    eta_self=1e-2,
    gamma_ema=0.1,
    epsilon_gap=1e-3,
):
    """Compute one regularized resolvent, geometric reflection, and update.

    ``W_self_v`` has shape ``(D_SPATIAL, D_SPATIAL)`` and ``W_self_hist``
    has shape ``(D_SPATIAL, D_hist)``. ``feat_hist`` is the already assembled
    Section 19 historical macro-vector. ``barycenter`` and ``introspection``
    are future-unit Lorentz points. The returned introspection point is the
    geodesic EMA of its input and the newly reflected self point.
    """
    dimension = D_SPATIAL
    W_self_v = _finite_array(
        W_self_v, (dimension, dimension), "W_self_v"
    )
    try:
        W_self_hist = np.asarray(W_self_hist, dtype=FP64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("W_self_hist must be a finite numeric matrix") from error
    if (
        W_self_hist.ndim != 2
        or W_self_hist.shape[0] != dimension
        or W_self_hist.shape[1] == 0
    ):
        raise ValueError(
            "W_self_hist must have D_SPATIAL rows and at least one column"
        )
    history_dimension = W_self_hist.shape[1]
    W_self_hist = _finite_array(
        W_self_hist, (dimension, history_dimension), "W_self_hist"
    )
    feat_hist = _finite_array(feat_hist, (history_dimension,), "feat_hist")
    b_self = _finite_array(b_self, (dimension,), "b_self")
    barycenter = _hyperboloid_point(barycenter, "barycenter")
    introspection = _hyperboloid_point(introspection, "introspection")

    delta_reg = _finite_scalar(delta_reg, "delta_reg")
    dt = _finite_scalar(dt, "dt")
    eta_self = _finite_scalar(eta_self, "eta_self")
    gamma_ema = _finite_scalar(gamma_ema, "gamma_ema")
    epsilon_gap = _finite_scalar(epsilon_gap, "epsilon_gap")
    if delta_reg <= 0.0:
        raise ValueError("delta_reg must be positive")
    if dt < 0.0:
        raise ValueError("dt must be non-negative")
    if eta_self < 0.0:
        raise ValueError("eta_self must be non-negative")
    if not 0.0 <= gamma_ema <= 1.0:
        raise ValueError("gamma_ema must be between zero and one")
    if not 0.0 < epsilon_gap < 1.0:
        raise ValueError("epsilon_gap must be between zero and one")

    resolvent = regularized_resolvent(
        W_self_v, delta_reg=delta_reg, epsilon_gap=epsilon_gap
    )
    v_self_star = resolvent @ (W_self_hist @ feat_hist + b_self)
    if not np.all(np.isfinite(v_self_star)):
        raise ValueError("self-reflection tangent must remain finite")

    barycenter_frame = get_boost_frame(barycenter)
    ambient_velocity = barycenter_frame @ v_self_star
    mu_self_star = geodesic_exp(barycenter, ambient_velocity)

    introspection_log = geodesic_log(introspection, mu_self_star)
    introspection_next = geodesic_exp(
        introspection, gamma_ema * introspection_log
    )

    residual = geodesic_log(introspection_next, mu_self_star)
    frame = get_boost_frame(introspection_next)
    gradient_self = frame.T @ (ETA_L * residual)
    transported_residual = parallel_transport(
        introspection_next, barycenter, frame @ gradient_self
    )
    gradient_tilde = barycenter_frame.T @ (ETA_L * transported_residual)
    if not np.all(np.isfinite(gradient_tilde)):
        raise ValueError("transported reflection gradient must remain finite")

    update = ego_reflection_step(
        W_self_v,
        W_self_hist,
        feat_hist,
        b_self,
        gradient_tilde,
        delta_reg=delta_reg,
        dt=dt,
        eta_self=eta_self,
        epsilon_gap=epsilon_gap,
    )
    cap = 1.0 - epsilon_gap
    if np.linalg.norm(update.W_self_v, ord=2) > cap + 1e-10:
        raise ValueError("updated W_self_v exceeded its spectral norm bound")
    arrays = (
        mu_self_star,
        introspection_next,
        gradient_self,
        gradient_tilde,
        update.W_self_v,
        update.W_self_hist,
        update.b_self,
    )
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("Algorithm 6 step must remain finite")
    return Algorithm06Step(*arrays)


__all__ = ["Algorithm06Step", "algorithm_06_step"]
