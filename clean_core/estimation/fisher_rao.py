import numpy as np
from clean_core.config import R_RANK, D_SPATIAL, FP32

class FisherRaoDLREstimator:
    def __init__(self, rank=R_RANK, dim=D_SPATIAL):
        self.rank = rank
        self.dim = dim
        self.d_diag = np.ones(dim, dtype=FP32)
        basis, _ = np.linalg.qr(np.random.randn(dim, rank))
        self.V_basis = basis.astype(FP32)
        self.s_values = np.full(rank, 1e-4, dtype=FP32)
        self.gamma_lr = 0.05

    def update_error(self, eps_tilde):
        eps = np.asarray(eps_tilde, dtype=FP32)
        if eps.shape != (self.dim,):
            raise ValueError(f"Expected error shape {(self.dim,)}, got {eps.shape}")

        eta = self.gamma_lr
        y = self.V_basis.T @ eps
        self.s_values = (1.0 - eta) * self.s_values + eta * (y**2 + 1e-5)
        v_raw = self.V_basis + eta * (
            np.outer(eps, y) - self.V_basis @ np.outer(y, y)
        )
        basis, _ = np.linalg.qr(v_raw, mode="reduced")
        self.V_basis = basis.astype(FP32)

        residual = eps - self.V_basis @ (self.V_basis.T @ eps)
        self.d_diag = (1.0 - eta) * self.d_diag + eta * (residual**2 + 1e-5)
        self.d_diag = np.maximum(self.d_diag, 1e-5)
        self.s_values = np.maximum(self.s_values, 1e-5)

    def get_precision_matrix(self):
        D_inv = np.diag(1.0 / self.d_diag)
        S_inv = np.diag(1.0 / self.s_values)
        middle = S_inv + self.V_basis.T @ D_inv @ self.V_basis
        M = np.linalg.solve(middle, np.eye(self.rank, dtype=FP32))
        return D_inv - D_inv @ self.V_basis @ M @ self.V_basis.T @ D_inv
