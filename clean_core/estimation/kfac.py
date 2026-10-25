import numpy as np
from clean_core.config import FP32

class KFACNaturalGradient:
    def __init__(self, in_dim, out_dim, alpha=0.95, damping=0.2):
        self.alpha = alpha
        self.damping = damping
        self.A = np.eye(in_dim, dtype=FP32)
        self.B = np.eye(out_dim, dtype=FP32)

    def update_factors(self, a_in, g_out):
        self.A = self.alpha * self.A + (1.0 - self.alpha) * np.outer(a_in, a_in)
        self.B = self.alpha * self.B + (1.0 - self.alpha) * np.outer(g_out, g_out)

    def compute_natural_gradient(self, grad_W):
        A_inv = np.linalg.inv(self.A + self.damping * np.eye(self.A.shape[0], dtype=FP32))
        B_inv = np.linalg.inv(self.B + self.damping * np.eye(self.B.shape[0], dtype=FP32))
        nat_grad = B_inv @ grad_W @ A_inv
        norm_val = np.linalg.norm(nat_grad)
        if norm_val > 1.0:
            nat_grad /= norm_val
        return nat_grad
