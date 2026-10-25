import numpy as np
from scipy.linalg import expm, schur
from clean_core.config import D_SCM, FP64

class NOTEARSCausalSCM:
    def __init__(self, d_scm=D_SCM):
        self.d = d_scm
        self.A_scm = np.zeros((d_scm, d_scm), dtype=FP64)

    def compute_acyclicity_h(self):
        E_mat = expm(self.A_scm * self.A_scm)
        return float(np.trace(E_mat) - self.d)

    def update_scm_and_clip(self, data_batch, alpha=0.01):
        cov = np.cov(data_batch, rowvar=False) if len(data_batch) > 1 else np.eye(self.d)
        grad = cov - self.A_scm
        self.A_scm += alpha * grad
        np.fill_diagonal(self.A_scm, 0.0)
        T_mat, Z_mat = schur(self.A_scm, output='real')
        max_diag = np.max(np.abs(np.diag(T_mat)))
        if max_diag > 0.95:
            T_mat = T_mat * (0.95 / max_diag)
            self.A_scm = Z_mat @ T_mat @ Z_mat.T
