import numpy as np
from clean_core.config import D_LORENTZ, D_SPATIAL, ETA_L, POLE, E_BASIS, FP64

def minkowski_dot(x, y):
    return float(np.dot(x * ETA_L, y))

def minkowski_norm_sq(x):
    return minkowski_dot(x, x)

def minkowski_norm(x):
    return float(np.sqrt(max(0.0, minkowski_norm_sq(x))))

def chordal_distance(x, y):
    diff = x - y
    chord = np.sqrt(max(0.0, minkowski_norm_sq(diff)))
    return float(2.0 * np.arcsinh(0.5 * chord))

def get_boost_frame(mu):
    mu_0 = mu[0]
    mu_spatial = mu[1:]
    o_plus_mu = POLE + mu
    E_transposed = E_BASIS + (mu_spatial[:, None] / (1.0 + mu_0)) * o_plus_mu[None, :]
    return E_transposed.T

def exp_map(mu, v):
    norm_v = minkowski_norm(v)
    if norm_v < 1e-4:
        res = (1.0 + 0.5 * norm_v**2) * mu + (1.0 + norm_v**2 / 6.0) * v
    else:
        res = np.cosh(norm_v) * mu + (np.sinh(norm_v) / norm_v) * v
    res[0] = np.sqrt(1.0 + np.sum(res[1:]**2))
    return res

def log_map(mu, y):
    u = y + minkowski_dot(mu, y) * mu
    norm_u = minkowski_norm(u)
    if norm_u < 1e-4:
        return (1.0 - norm_u**2 / 6.0) * u
    diff = mu - y
    d_H = 2.0 * np.arcsinh(0.5 * minkowski_norm(diff))
    return (d_H / norm_u) * u

def parallel_transport(x, y, v):
    log_xy = log_map(x, y)
    d_xy = minkowski_norm(log_xy)
    if d_xy < 1e-6:
        return v.copy()
    denominator = 1.0 - minkowski_dot(x, y)
    return v + (minkowski_dot(v, y) / denominator) * (x + y)


def streaming_frechet_barycenter(previous, points, step_size=0.2):
    points = np.asarray(points, dtype=FP64)
    previous = np.asarray(previous, dtype=FP64)
    if points.ndim != 2 or points.shape[1] != D_LORENTZ or points.shape[0] == 0:
        raise ValueError(f"points must have shape (n > 0, {D_LORENTZ})")
    if previous.shape != (D_LORENTZ,):
        raise ValueError(f"previous must have shape ({D_LORENTZ},)")
    if not 0.0 <= step_size <= 1.0:
        raise ValueError("step_size must be between 0 and 1")

    mean_tangent = np.mean([log_map(previous, point) for point in points], axis=0)
    return exp_map(previous, step_size * mean_tangent)
