import numpy as np

from clean_core.config import FP64


def polyhedral_cole_hopf_motor_step(
    current_action,
    geodesic_velocity,
    polyhedral_pressure,
    social_action,
    geodesic_weights,
    polyhedral_weights,
    social_weights,
    bias,
    dt=0.01,
    learning_rate=0.1,
):
    vectors = [
        np.asarray(current_action, dtype=FP64),
        np.asarray(geodesic_velocity, dtype=FP64),
        np.asarray(polyhedral_pressure, dtype=FP64),
        np.asarray(social_action, dtype=FP64),
        np.asarray(bias, dtype=FP64),
    ]
    current, geodesic, polyhedral, social, bias = vectors
    weights = [
        np.asarray(geodesic_weights, dtype=FP64),
        np.asarray(polyhedral_weights, dtype=FP64),
        np.asarray(social_weights, dtype=FP64),
    ]
    if current.ndim != 1 or current.size == 0:
        raise ValueError("current_action must be a non-empty vector")
    if any(not np.all(np.isfinite(value)) for value in vectors + weights):
        raise ValueError("all vectors and matrices must be finite")
    expected = (current.size,)
    if bias.shape != expected or any(matrix.ndim != 2 or matrix.shape[0] != current.size for matrix in weights):
        raise ValueError("bias and weight matrices must target the action dimension")
    if any(matrix.shape[1] != vector.size for matrix, vector in zip(weights, (geodesic, polyhedral, social))):
        raise ValueError("weight matrix input dimensions must match their source vectors")
    if not np.isfinite(dt) or dt <= 0.0 or not np.isfinite(learning_rate) or learning_rate < 0.0:
        raise ValueError("dt must be positive and learning_rate non-negative")

    target = np.tanh(
        weights[0] @ geodesic + weights[1] @ polyhedral + weights[2] @ social + bias
    )
    surrogate_efe_gradient = current - target
    updated = current - learning_rate * dt * surrogate_efe_gradient
    return np.clip(updated, -1.0, 1.0), target