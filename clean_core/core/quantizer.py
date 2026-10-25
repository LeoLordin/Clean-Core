import numpy as np
from clean_core.config import K_MAX, D_SPATIAL, FP64, POLE
from clean_core.geometry.lorentz import log_map, exp_map, chordal_distance

class SoftVQQuantizer:
    def __init__(self, k_max=K_MAX, dim=D_SPATIAL):
        self.k_max = k_max
        self.dim = dim
        self.centroids = np.random.randn(4, dim).astype(FP64) * 0.05
        self.usage_counts = np.ones(4, dtype=FP64)

    def quantize_tangent(self, mu_point, temp_vq=0.1):
        v_tangent = log_map(POLE, mu_point)
        dists = np.linalg.norm(self.centroids - v_tangent[1:], axis=1)
        k_best = np.argmin(dists)
        self.usage_counts[k_best] += 1.0
        c_k = self.centroids[k_best]
        v_centroid = np.array([0.0] + list(c_k), dtype=FP64)
        mu_centroid = exp_map(POLE, v_centroid)
        q_soft = np.exp(-dists / temp_vq) / np.sum(np.exp(-dists / temp_vq))
        return k_best, mu_centroid, q_soft

    def prune_and_spawn(self, mu_err_point, err_norm_sq, theta_spawn=15.0, theta_prune=0.05):
        if err_norm_sq > theta_spawn and len(self.centroids) < self.k_max:
            v_err = log_map(POLE, mu_err_point)[1:]
            self.centroids = np.vstack([self.centroids, v_err])
            self.usage_counts = np.append(self.usage_counts, 1.0)
            return True, "SPAWNED"
        total_usage = np.sum(self.usage_counts)
        usage_ratios = self.usage_counts / max(1.0, total_usage)
        keep_mask = usage_ratios >= theta_prune
        if np.sum(keep_mask) > 1 and np.sum(keep_mask) < len(self.centroids):
            self.centroids = self.centroids[keep_mask]
            self.usage_counts = self.usage_counts[keep_mask]
            return True, "PRUNED"
        return False, "UNCHANGED"
