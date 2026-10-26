"""Deterministic bounded inference step for Algorithm 2.

The supplied context is the already assembled SoftVQ context vector, and the
layer barycenter is supplied by the caller. This prototype performs the
feed-forward prediction, local-frame error/precision update, and one bounded
deterministic Lorentz step. It updates inference state only, never MLP weights.
"""

from dataclasses import dataclass

import numpy as np

from clean_core.config import D_LORENTZ, D_SPATIAL, ETA_L, FP32, FP64
from clean_core.core.neuron import NeuronMicrostate
from clean_core.geometry.lorentz import exp_map, get_boost_frame, log_map

_MAX_DT = 0.1
_MAX_TANGENT_NORM = 3.0
_MAX_SPEED = 5.0
_DLR_RATE = 0.05
_DLR_FLOOR = 1e-5
_DLR_CEILING = 1e5


@dataclass(frozen=True)
class Algorithm02Result:
    """Outputs from one deterministic inference-only manifold step."""

    mu: np.ndarray
    velocity: np.ndarray
    prediction: np.ndarray
    local_error: np.ndarray
    precision: np.ndarray
    adaptive_precision: float


def _finite_array(name, value, shape=None):
    try:
        array = np.asarray(value, dtype=FP64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite numeric array") from error
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _finite_scalar(name, value, *, minimum=None, maximum=None):
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and scalar < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and scalar > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return scalar


def _hyperboloid_point(name, value):
    point = _finite_array(name, value, (D_LORENTZ,))
    with np.errstate(over="ignore", invalid="ignore"):
        norm_sq = float(np.sum(point * ETA_L * point))
    if (
        point[0] <= 0.0
        or not np.isfinite(norm_sq)
        or not np.isclose(norm_sq, -1.0, rtol=1e-9, atol=1e-9)
    ):
        raise ValueError(f"{name} must lie on the future unit Lorentz hyperboloid")
    return point


def _orthonormalize(matrix, rank):
    """Deterministically complete a possibly rank-deficient basis."""
    columns = []
    for column in [*matrix.T, *np.eye(D_SPATIAL, dtype=FP64)]:
        candidate = column.copy()
        for basis_column in columns:
            candidate -= np.dot(basis_column, candidate) * basis_column
        norm = float(np.linalg.norm(candidate))
        if norm > 1e-10:
            columns.append(candidate / norm)
            if len(columns) == rank:
                break
    if len(columns) != rank:
        raise ValueError("V_basis cannot be completed to an orthonormal basis")
    return np.column_stack(columns)


def _clip_norm(name, vector, maximum):
    if not np.all(np.isfinite(vector)):
        raise FloatingPointError(f"{name} must remain finite")
    scale = float(np.max(np.abs(vector)))
    if scale == 0.0:
        return vector.copy()
    scaled_norm = float(np.linalg.norm(vector / scale))
    if scale <= maximum / scaled_norm:
        return vector.copy()
    return (vector / scale) * (maximum / scaled_norm)


def _woodbury_precision(diagonal, basis, low_rank_values):
    inverse_diagonal = 1.0 / diagonal
    inverse_diagonal_basis = inverse_diagonal[:, None] * basis
    middle = (
        np.diag(1.0 / low_rank_values)
        + basis.T @ inverse_diagonal_basis
    )
    precision = np.diag(inverse_diagonal) - inverse_diagonal_basis @ np.linalg.solve(
        middle, inverse_diagonal_basis.T
    )
    return 0.5 * (precision + precision.T)


def run_algorithm_02(
    neuron,
    context,
    barycenter_mu,
    *,
    dt=0.01,
    alpha_damp=0.05,
):
    """Advance ``neuron`` once using fixed weights and bounded geodesic motion.

    ``context`` is the finite ``D_SPATIAL`` SoftVQ context vector. Inference
    state (coordinates, velocity, local error, and DLR precision statistics)
    is updated only after all outputs validate; ``W1`` and ``W2`` are read-only.
    Langevin noise is omitted so the step is deterministic.
    """
    if not isinstance(neuron, NeuronMicrostate):
        raise ValueError("neuron must be a NeuronMicrostate")

    mu = _hyperboloid_point("neuron.mu", neuron.mu)
    barycenter = _hyperboloid_point("barycenter_mu", barycenter_mu)
    context = _finite_array("context", context, (D_SPATIAL,))
    velocity = _finite_array("neuron.w", neuron.w, (D_SPATIAL,))
    weights_1 = _finite_array("neuron.W1", neuron.W1)
    if weights_1.ndim != 2 or weights_1.shape[1] != D_SPATIAL:
        raise ValueError(f"neuron.W1 must have shape (hidden_dim, {D_SPATIAL})")
    if weights_1.shape[0] < 1:
        raise ValueError("neuron.W1 must have at least one hidden unit")
    hidden_dim = weights_1.shape[0]
    weights_2 = _finite_array(
        "neuron.W2", neuron.W2, (D_SPATIAL, hidden_dim)
    )
    basis = _finite_array("neuron.V_basis", neuron.V_basis)
    if (
        basis.ndim != 2
        or basis.shape[0] != D_SPATIAL
        or not 1 <= basis.shape[1] <= D_SPATIAL
    ):
        raise ValueError(f"neuron.V_basis must have shape ({D_SPATIAL}, rank)")
    rank = basis.shape[1]
    diagonal = _finite_array("neuron.d_diag", neuron.d_diag, (D_SPATIAL,))
    if np.any(diagonal <= 0.0):
        raise ValueError("neuron.d_diag entries must be positive")
    _finite_array("neuron.Pi_precision", neuron.Pi_precision, (D_SPATIAL, D_SPATIAL))
    local_gradient = _finite_array(
        "neuron.grad_ln_Omega", neuron.grad_ln_Omega, (D_SPATIAL,)
    )
    momentum = _finite_array("neuron.p", neuron.p, (D_LORENTZ,))
    omega = _finite_scalar("neuron.Omega", neuron.Omega, minimum=np.finfo(FP64).tiny)
    dt = _finite_scalar("dt", dt, minimum=np.finfo(FP64).tiny, maximum=_MAX_DT)
    alpha_damp = _finite_scalar("alpha_damp", alpha_damp, minimum=0.0)

    first_frame = get_boost_frame(mu)
    barycenter_frame = get_boost_frame(barycenter)
    with np.errstate(over="ignore", invalid="ignore"):
        hidden = np.tanh(weights_1 @ context)
        output = weights_2 @ hidden
    if not np.all(np.isfinite(hidden)) or not np.all(np.isfinite(output)):
        raise ValueError("feed-forward prediction must remain finite")

    safe_output = _clip_norm("prediction tangent", output, _MAX_TANGENT_NORM)
    prediction = exp_map(barycenter.copy(), barycenter_frame @ safe_output)
    prediction = _hyperboloid_point("prediction", prediction)
    prediction_error = log_map(mu, prediction)
    local_error = first_frame.T @ (ETA_L * prediction_error)
    if not np.all(np.isfinite(local_error)):
        raise ValueError("prediction error must remain finite")

    projected_error = basis.T @ local_error
    updated_basis = basis + _DLR_RATE * (
        np.outer(local_error, projected_error)
        - basis @ np.outer(projected_error, projected_error)
    )
    updated_basis = _orthonormalize(updated_basis, rank)
    residual = local_error - updated_basis @ (updated_basis.T @ local_error)
    updated_diagonal = np.clip(
        (1.0 - _DLR_RATE) * diagonal
        + _DLR_RATE * (residual**2 + _DLR_FLOOR),
        _DLR_FLOOR,
        _DLR_CEILING,
    )
    low_rank_values = np.clip(
        (1.0 - _DLR_RATE) * _DLR_FLOOR
        + _DLR_RATE * (projected_error**2 + _DLR_FLOOR),
        _DLR_FLOOR,
        _DLR_CEILING,
    )
    precision = _woodbury_precision(updated_diagonal, updated_basis, low_rank_values)
    error_energy = max(0.0, float(local_error @ precision @ local_error))
    adaptive_precision = float(
        np.clip(
            D_SPATIAL / max(_DLR_FLOOR, error_energy),
            1e-3,
            1e3,
        )
    )

    speed_sq = float(np.dot(velocity, velocity))
    conformal_drift = (
        -speed_sq * local_gradient
        + 2.0 * np.dot(velocity, local_gradient) * velocity
    )
    momentum_norm_sq = float(np.sum(momentum * ETA_L * momentum))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        momentum_drift = (
            momentum_norm_sq / np.float64(omega) ** 3
        ) * local_gradient
    force = adaptive_precision * (precision @ local_error)
    force += conformal_drift + momentum_drift
    system = (
        np.eye(D_SPATIAL, dtype=FP64)
        + dt * alpha_damp * adaptive_precision * precision
    )
    raw_velocity = np.linalg.solve(system, velocity + dt * force)
    next_velocity = _clip_norm("updated velocity", raw_velocity, _MAX_SPEED)
    next_mu = exp_map(mu.copy(), first_frame @ (dt * next_velocity))
    next_mu = _hyperboloid_point("updated coordinates", next_mu)
    next_error = get_boost_frame(next_mu).T @ (
        ETA_L * log_map(next_mu, prediction)
    )
    if (
        not np.all(np.isfinite(next_error))
        or not np.all(np.isfinite(precision))
        or not np.all(np.isfinite(next_velocity))
        or not np.isfinite(adaptive_precision)
    ):
        raise FloatingPointError("inference step produced a non-finite output")

    neuron.mu = next_mu.copy()
    neuron.w = next_velocity.copy()
    neuron.eps_tilde = next_error.astype(FP32)
    neuron.V_basis = updated_basis.astype(FP32)
    neuron.d_diag = updated_diagonal.astype(FP32)
    neuron.Pi_precision = precision.astype(FP32)
    return Algorithm02Result(
        mu=next_mu.copy(),
        velocity=next_velocity.copy(),
        prediction=prediction.copy(),
        local_error=next_error.copy(),
        precision=precision.copy(),
        adaptive_precision=adaptive_precision,
    )


__all__ = ["Algorithm02Result", "run_algorithm_02"]
