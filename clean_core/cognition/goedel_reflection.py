import numpy as np
from clean_core.config import D_SPATIAL, FP64, POLE
from clean_core.geometry.lorentz import exp_map, get_boost_frame, log_map

class GoedelEgoReflectionEngine:
    def __init__(self, dim=D_SPATIAL):
        self.dim = dim
        self.W_self_v = np.random.randn(dim, dim).astype(FP64) * 0.01
        self.introspection_mu = POLE.copy()

    def step_ego_reflection(self, eps_history, barycenter_mu, delta_reg=0.1):
        if len(eps_history) == 0:
            mean_eps = np.zeros(self.dim, dtype=FP64)
        else:
            mean_eps = np.mean(eps_history, axis=0).astype(FP64)
            if len(mean_eps) < self.dim:
                mean_eps = np.pad(mean_eps, (0, self.dim - len(mean_eps)))
            else:
                mean_eps = mean_eps[:self.dim]
        I_mat = np.eye(self.dim, dtype=FP64)
        resolvent = np.linalg.solve((1.0 + delta_reg) * I_mat - self.W_self_v.T, I_mat)
        v_self_star = resolvent @ mean_eps
        v_tangent = get_boost_frame(barycenter_mu) @ v_self_star
        mu_self = exp_map(barycenter_mu, v_tangent)
        introspection_tangent = log_map(self.introspection_mu, mu_self)
        self.introspection_mu = exp_map(self.introspection_mu, 0.1 * introspection_tangent)
        return mu_self, self.introspection_mu.copy()
