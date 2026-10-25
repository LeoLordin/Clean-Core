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
    return float(np.sqrt(max(0.0, minkowski_norm_sq(diff))))

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
    u = log_xy / d_xy
    proj_u = minkowski_dot(u, v)
    return v + (np.cosh(d_xy) - 1.0) * proj_u * u - np.sinh(d_xy) * proj_u * x
