"""Finite-step K-FAC natural-gradient update using Section 12 statistics."""

import operator

import numpy as np

from clean_core.config import FP32

_INFERENCE_PERIOD = 4
_MAX_STEP_NORM = 1.0


def _matrix(name, value, shape):
    try:
        result = np.asarray(value, dtype=FP32)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite numeric matrix of shape {shape}") from error
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result.copy()


def _positive_scalar(name, value):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be positive and finite") from error
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def apply_algorithm_07(
    section12,
    weights_1,
    weights_2,
    grad_w1,
    grad_w2,
    *,
    inference_tick,
    dt,
    epsilon_1=1.0,
):
    """Return one scheduled, norm-bounded K-FAC weight update.

    K-FAC statistics are read from ``section12``; this function does not mutate
    its DLR/K-FAC state or the supplied weights. Updates occur on positive
    inference ticks divisible by four. The returned mapping contains copied
    weights and, on update ticks, the clipped natural gradients.
    """
    try:
        tick = operator.index(inference_tick)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("inference_tick must be a positive integer") from error
    if isinstance(inference_tick, (bool, np.bool_)) or tick <= 0:
        raise ValueError("inference_tick must be a positive integer")

    dt = _positive_scalar("dt", dt)
    epsilon_1 = _positive_scalar("epsilon_1", epsilon_1)
    scale = dt / epsilon_1
    if not np.isfinite(scale):
        raise ValueError("dt / epsilon_1 must be finite")

    expected_shapes = (
        (section12.kfac_layer1.B.shape[0], section12.kfac_layer1.A.shape[0]),
        (section12.kfac_layer2.B.shape[0], section12.kfac_layer2.A.shape[0]),
    )
    weights_1 = _matrix("weights_1", weights_1, expected_shapes[0])
    weights_2 = _matrix("weights_2", weights_2, expected_shapes[1])
    grad_w1 = _matrix("grad_w1", grad_w1, expected_shapes[0])
    grad_w2 = _matrix("grad_w2", grad_w2, expected_shapes[1])

    result = {
        "updated": tick % _INFERENCE_PERIOD == 0,
        "weights_1": weights_1.copy(),
        "weights_2": weights_2.copy(),
        "natural_gradient_1": None,
        "natural_gradient_2": None,
    }
    if not result["updated"]:
        return result

    natural_gradients = (
        section12.compute_natural_gradient(1, grad_w1),
        section12.compute_natural_gradient(2, grad_w2),
    )
    clipped_gradients = []
    for name, gradient, shape in zip(
        ("natural_gradient_1", "natural_gradient_2"),
        natural_gradients,
        expected_shapes,
    ):
        gradient = _matrix(name, gradient, shape)
        norm = float(np.linalg.norm(gradient.astype(np.float64)))
        if not np.isfinite(norm):
            raise ValueError(f"{name} must have a finite Frobenius norm")
        if norm > _MAX_STEP_NORM:
            gradient /= norm
        clipped_gradients.append(gradient)

    updated_weights = []
    for weights, gradient in zip(
        (weights_1, weights_2), clipped_gradients
    ):
        updated = weights.astype(np.float64) - scale * gradient.astype(np.float64)
        if not np.all(np.isfinite(updated)):
            raise ValueError("updated weights must remain finite")
        updated = updated.astype(FP32)
        if not np.all(np.isfinite(updated)):
            raise ValueError("updated weights must be representable as finite FP32 values")
        updated_weights.append(updated)

    result.update(
        weights_1=updated_weights[0],
        weights_2=updated_weights[1],
        natural_gradient_1=clipped_gradients[0].copy(),
        natural_gradient_2=clipped_gradients[1].copy(),
    )
    return result
