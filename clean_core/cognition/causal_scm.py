import numpy as np
from scipy.linalg import expm
from clean_core.config import D_SCM, FP64

class NOTEARSCausalSCM:
    def __init__(self, d_scm=D_SCM):
        self.d = d_scm
        self.A_scm = np.zeros((d_scm, d_scm), dtype=FP64)

    def compute_acyclicity_h(self):
        E_mat = expm(self.A_scm * self.A_scm)
        return float(np.log(np.trace(E_mat) / self.d))

    def compute_acyclicity_gradient(self):
        E_mat = expm(self.A_scm * self.A_scm)
        return (2.0 / np.trace(E_mat)) * (E_mat.T * self.A_scm)

    def update_scm_and_clip(
        self,
        data_batch,
        alpha=0.01,
        dag_penalty=1.0,
        l1_penalty=0.0,
        max_spectral_radius=0.95,
    ):
        data = np.asarray(data_batch, dtype=FP64)
        if data.ndim != 2 or data.shape[1] != self.d or data.shape[0] < 2:
            raise ValueError(f"data_batch must have shape (n >= 2, {self.d})")
        if not np.all(np.isfinite(data)):
            raise ValueError("data_batch must contain only finite values")
        if alpha <= 0.0 or dag_penalty < 0.0 or l1_penalty < 0.0:
            raise ValueError("learning rate must be positive and penalties non-negative")
        if not 0.0 < max_spectral_radius < 1.0:
            raise ValueError("max_spectral_radius must be between 0 and 1")

        residual = data @ self.A_scm - data
        grad_data = data.T @ residual / data.shape[0]
        grad = grad_data + dag_penalty * self.compute_acyclicity_gradient()
        if l1_penalty:
            grad += l1_penalty * np.sign(self.A_scm)
        self.A_scm -= alpha * grad
        np.fill_diagonal(self.A_scm, 0.0)
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(self.A_scm))))
        if spectral_radius > max_spectral_radius:
            self.A_scm *= max_spectral_radius / spectral_radius
        return self.compute_acyclicity_h()
