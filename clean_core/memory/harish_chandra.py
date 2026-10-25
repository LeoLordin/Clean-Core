import numpy as np
from clean_core.config import H_DIM, D_SPATIAL, FP64, COMPLEX_MEM, POLE
from clean_core.geometry.lorentz import log_map

class HarishChandraHolographicMemory:
    def __init__(self, h_dim=H_DIM):
        self.h_dim = h_dim
        self.lambda_freqs = np.linspace(0.1, 5.0, h_dim, dtype=FP64)

    def plancherel_measure(self, lam):
        return float(lam * np.tanh(np.pi * lam))

    def transform_to_phase_space(self, mu_point):
        v_log = log_map(POLE, mu_point)
        r_dist = float(np.linalg.norm(v_log[1:]))
        phi_pos = np.exp(1j * self.lambda_freqs * r_dist)
        phi_neg = np.exp(-1j * self.lambda_freqs * r_dist)
        psi_wave = (phi_pos + phi_neg) / np.sqrt(2.0)
        return psi_wave.astype(COMPLEX_MEM)
