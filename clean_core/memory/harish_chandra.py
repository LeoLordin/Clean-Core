import numpy as np
from clean_core.config import H_DIM, D_SPATIAL, FP64, COMPLEX_MEM, POLE
from clean_core.geometry.lorentz import log_map

class HarishChandraHolographicMemory:
    def __init__(self, h_dim=H_DIM, capacity=256):
        if h_dim < 1 or capacity < 1:
            raise ValueError("h_dim and capacity must be positive")
        self.h_dim = h_dim
        self.capacity = capacity
        self.lambda_freqs = np.linspace(0.1, 5.0, h_dim, dtype=FP64)
        self._keys = []
        self._values = []

    def plancherel_measure(self, lam):
        return float(lam * np.tanh(np.pi * lam))

    def transform_to_phase_space(self, mu_point):
        mu_point = np.asarray(mu_point, dtype=FP64)
        if mu_point.shape != POLE.shape or not np.all(np.isfinite(mu_point)):
            raise ValueError(f"mu_point must be a finite vector with shape {POLE.shape}")
        v_log = log_map(POLE, mu_point)
        r_dist = float(np.linalg.norm(v_log[1:]))
        phi_pos = np.exp(1j * self.lambda_freqs * r_dist)
        phi_neg = np.exp(-1j * self.lambda_freqs * r_dist)
        psi_wave = (phi_pos + phi_neg) / np.sqrt(2.0)
        return psi_wave.astype(COMPLEX_MEM)

    def store(self, mu_point, value):
        mu_point = np.asarray(mu_point, dtype=FP64)
        value = np.asarray(value, dtype=FP64)
        if mu_point.shape != POLE.shape or not np.all(np.isfinite(mu_point)):
            raise ValueError(f"mu_point must be a finite vector with shape {POLE.shape}")
        if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
            raise ValueError("value must be a non-empty finite vector")
        if self._values and value.shape != self._values[0].shape:
            raise ValueError("all stored values must have the same shape")
        self._keys.append(self.transform_to_phase_space(mu_point))
        self._values.append(value.copy())
        if len(self._keys) > self.capacity:
            self._keys.pop(0)
            self._values.pop(0)

    def retrieve(self, query_mu, temperature=0.2):
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        query = self.transform_to_phase_space(query_mu).astype(np.complex128)
        if not self._values:
            return np.zeros(0, dtype=FP64)
        frequencies = self.lambda_freqs.astype(FP64)
        density = np.array([self.plancherel_measure(value) for value in frequencies])
        density = np.maximum(density, np.finfo(FP64).tiny)
        scores = np.array([
            np.real(np.sum(density * np.conj(query) * key.astype(np.complex128))) / np.sum(density)
            for key in self._keys
        ], dtype=FP64)
        logits = (scores - np.max(scores)) / temperature
        weights = np.exp(logits)
        weights /= np.sum(weights)
        return np.sum(np.stack(self._values) * weights[:, None], axis=0)
