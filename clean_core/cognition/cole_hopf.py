import numpy as np

from clean_core.config import FP64


def _positive_scale(lambda_value, sigma):
    if not np.isfinite(lambda_value) or lambda_value <= 0.0:
        raise ValueError("lambda_value must be finite and positive")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    return float(lambda_value * sigma**2)


def value_to_wavefunction(value, lambda_value, sigma):
    scale = _positive_scale(lambda_value, sigma)
    value = np.asarray(value, dtype=FP64)
    if not np.all(np.isfinite(value)):
        raise ValueError("value must contain only finite values")
    with np.errstate(over="raise", invalid="raise"):
        return np.exp(-value / scale)


def wavefunction_to_value(wavefunction, lambda_value, sigma):
    scale = _positive_scale(lambda_value, sigma)
    wavefunction = np.asarray(wavefunction, dtype=FP64)
    if not np.all(np.isfinite(wavefunction)) or np.any(wavefunction <= 0.0):
        raise ValueError("wavefunction must contain finite positive values")
    return -scale * np.log(wavefunction)


def superpose_goal_wavefunctions(wavefunctions, weights=None):
    wavefunctions = np.asarray(wavefunctions, dtype=FP64)
    if wavefunctions.ndim != 2 or wavefunctions.shape[0] == 0:
        raise ValueError("wavefunctions must have shape (goals, states)")
    if not np.all(np.isfinite(wavefunctions)) or np.any(wavefunctions <= 0.0):
        raise ValueError("wavefunctions must contain finite positive values")
    if weights is None:
        weights = np.ones(wavefunctions.shape[0], dtype=FP64)
    weights = np.asarray(weights, dtype=FP64)
    if weights.shape != (wavefunctions.shape[0],) or not np.all(np.isfinite(weights)):
        raise ValueError("weights must be a finite vector with one entry per goal")
    if np.any(weights < 0.0) or not np.any(weights > 0.0):
        raise ValueError("weights must be non-negative with positive total mass")
    return weights @ wavefunctions


def cole_hopf_graph_diffusion_step(
    wavefunction,
    edge_weights,
    pragmatic_potential,
    dt,
    lambda_value,
    sigma,
):
    """Take one positivity-preserving explicit step on a weighted graph.

    The graph Laplacian is a discrete approximation, not an exact
    Laplace-Beltrami operator on the continuous hyperboloid.
    """
    scale = _positive_scale(lambda_value, sigma)
    wavefunction = np.asarray(wavefunction, dtype=FP64)
    edge_weights = np.asarray(edge_weights, dtype=FP64)
    potential = np.asarray(pragmatic_potential, dtype=FP64)
    if wavefunction.ndim != 1 or wavefunction.size == 0:
        raise ValueError("wavefunction must be a non-empty vector")
    if edge_weights.shape != (wavefunction.size, wavefunction.size):
        raise ValueError("edge_weights must be square and match wavefunction")
    if potential.shape != wavefunction.shape:
        raise ValueError("pragmatic_potential must match wavefunction")
    if not all(np.all(np.isfinite(value)) for value in (wavefunction, edge_weights, potential)):
        raise ValueError("inputs must contain only finite values")
    if np.any(wavefunction <= 0.0):
        raise ValueError("wavefunction must be strictly positive")
    if np.any(edge_weights < 0.0) or not np.allclose(edge_weights, edge_weights.T, atol=1e-12):
        raise ValueError("edge_weights must be symmetric and non-negative")
    if not np.allclose(np.diag(edge_weights), 0.0, atol=1e-12):
        raise ValueError("edge_weights diagonal must be zero")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")

    diffusion = scale / 2.0
    degree = np.sum(edge_weights, axis=1)
    loss_rate = diffusion * degree + np.maximum(potential, 0.0) / scale
    max_loss_rate = float(np.max(loss_rate))
    if max_loss_rate > 0.0 and dt * max_loss_rate > 1.0:
        raise ValueError("dt violates the explicit-step positivity bound")

    graph_laplacian = edge_weights @ wavefunction - degree * wavefunction
    derivative = diffusion * graph_laplacian - (potential / scale) * wavefunction
    result = wavefunction + dt * derivative
    if np.any(result <= 0.0) or not np.all(np.isfinite(result)):
        raise FloatingPointError("diffusion step did not preserve positivity")
    return result


def cole_hopf_optimal_velocity(log_wavefunction_gradient, lambda_value, sigma):
    scale = _positive_scale(lambda_value, sigma)
    gradient = np.asarray(log_wavefunction_gradient, dtype=FP64)
    if gradient.ndim != 1 or not np.all(np.isfinite(gradient)):
        raise ValueError("log_wavefunction_gradient must be a finite vector")
    return scale * gradient