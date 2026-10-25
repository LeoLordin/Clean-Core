import numpy as np
from clean_core.config import R_RANK, D_SPATIAL, FP32

class FisherRaoDLREstimator:
    def __init__(self, rank=R_RANK, dim=D_SPATIAL):
        self.rank = rank
        self.dim = dim
        self.d_diag = np.ones(dim, dtype=FP32)
        self.V_basis = np.random.randn(dim, rank).astype(FP32) * 0.01
        self.gamma_lr = 0.01

    def update_error(self, eps_tilde):
        eps = eps_tilde.astype(FP32)
        self.d_diag = 0.99 * self.d_diag + 0.01 * (eps**2 + 1e-4)
        y = self.V_basis.T @ eps
        self.V_basis += self.gamma_lr * (np.outer(eps, y) - self.V_basis @ np.outer(y, y))

    def get_precision_matrix(self):
        D_inv = np.diag(1.0 / self.d_diag)
        VDV = self.V_basis.T @ D_inv @ self.V_basis
        M = np.linalg.inv(np.eye(self.rank, dtype=FP32) + VDV)
        return D_inv - D_inv @ self.V_basis @ M @ self.V_basis.T @ D_inv
