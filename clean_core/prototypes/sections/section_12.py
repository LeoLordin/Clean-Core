"""Prototype for Section 12's DLR/Oja and K-FAC updates."""

import numpy as np

from clean_core.config import ETA_L, FP32
from clean_core.estimation.fisher_rao import FisherRaoDLREstimator
from clean_core.estimation.kfac import KFACNaturalGradient
from clean_core.geometry.lorentz import get_boost_frame, parallel_transport

__all__ = ["Section12Prototype"]


class Section12Prototype:
    """Stream DLR statistics, transported sensitivities, and two K-FAC factors."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        rank,
        *,
        eta_m=0.05,
        eta_k=0.05,
        damping=0.2,
        seed=0,
    ):
        self.error_dim = ETA_L.size - 1
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.rank = int(rank)
        if self.input_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if not 0 < self.rank <= self.error_dim:
            raise ValueError(f"rank must be between 1 and {self.error_dim}")
        if not np.isfinite(eta_m) or not 0.0 <= eta_m <= 1.0:
            raise ValueError("eta_m must be between 0 and 1")
        if not np.isfinite(eta_k) or not 0.0 <= eta_k <= 1.0:
            raise ValueError("eta_k must be between 0 and 1")
        if not np.isfinite(damping) or damping <= 0.0:
            raise ValueError("damping must be positive and finite")

        self.eta_m = float(eta_m)
        self.eta_k = float(eta_k)
        self.delta_min = 1e-5
        self.dlr = FisherRaoDLREstimator(rank=self.rank, dim=self.error_dim)
        self.dlr.gamma_lr = self.eta_m
        basis, _ = np.linalg.qr(
            np.random.default_rng(seed).standard_normal((self.error_dim, self.rank)),
            mode="reduced",
        )
        self.dlr.V_basis = basis.astype(FP32)
        self.kfac_layer1 = KFACNaturalGradient(
            self.input_dim, self.hidden_dim, alpha=1.0 - self.eta_k, damping=damping
        )
        self.kfac_layer2 = KFACNaturalGradient(
            self.hidden_dim, self.error_dim, alpha=1.0 - self.eta_k, damping=damping
        )

    @staticmethod
    def _vector(name, value, length):
        result = np.asarray(value, dtype=FP32)
        if result.shape != (length,):
            raise ValueError(f"{name} must have shape ({length},)")
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain only finite values")
        return result

    @staticmethod
    def _matrix(name, value, shape):
        result = np.asarray(value, dtype=FP32)
        if result.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain only finite values")
        return result

    @staticmethod
    def _hyperboloid_point(name, point):
        result = np.asarray(point, dtype=np.float64)
        if result.shape != ETA_L.shape or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be a finite Lorentz vector of shape {ETA_L.shape}")
        if result[0] <= 0.0 or not np.isclose(result @ (ETA_L * result), -1.0, atol=1e-6):
            raise ValueError(f"{name} must be a future-unit hyperboloid point")
        return result

    @staticmethod
    def _normalized_factor_input(value):
        value = np.asarray(value, dtype=np.float64)
        normalized = value / np.sqrt(1.0 + np.dot(value, value))
        return normalized.astype(FP32)

    def step(
        self,
        error,
        activation,
        hidden_activation,
        hidden_derivative,
        weights_1,
        weights_2,
        mu_current,
        mu_mean,
        *,
        precision_scalar=1.0,
    ):
        """Apply one Section 12 streaming update and return its intermediates."""
        error = self._vector("error", error, self.error_dim)
        activation = self._vector("activation", activation, self.input_dim)
        hidden_activation = self._vector("hidden_activation", hidden_activation, self.hidden_dim)
        hidden_derivative = self._vector("hidden_derivative", hidden_derivative, self.hidden_dim)
        weights_1 = self._matrix(
            "weights_1", weights_1, (self.hidden_dim, self.input_dim)
        )
        weights_2 = self._matrix(
            "weights_2", weights_2, (self.error_dim, self.hidden_dim)
        )
        mu_current = self._hyperboloid_point("mu_current", mu_current)
        mu_mean = self._hyperboloid_point("mu_mean", mu_mean)
        if not np.isfinite(precision_scalar) or precision_scalar < 0.0:
            raise ValueError("precision_scalar must be nonnegative and finite")

        old_basis = self.dlr.V_basis.copy()
        y = old_basis.T @ error
        precision = self.dlr.get_precision_matrix()
        g2_local = float(precision_scalar) * (precision @ error)

        local_frame = get_boost_frame(mu_current)
        mean_frame = get_boost_frame(mu_mean)
        transported = parallel_transport(
            mu_current, mu_mean, local_frame @ g2_local
        )
        g2 = mean_frame.T @ (ETA_L * transported)
        g1 = (weights_2.T @ g2) * hidden_derivative

        self.dlr.update_error(error)
        residual = error - self.dlr.V_basis @ (self.dlr.V_basis.T @ error)
        self.kfac_layer1.update_factors(
            self._normalized_factor_input(activation),
            self._normalized_factor_input(g1),
        )
        self.kfac_layer2.update_factors(
            self._normalized_factor_input(hidden_activation),
            self._normalized_factor_input(g2),
        )

        return {
            "y": y.copy(),
            "residual": residual.copy(),
            "precision": precision.copy(),
            "g2_local": g2_local.copy(),
            "g2": g2.copy(),
            "g1": g1.copy(),
            "basis": self.dlr.V_basis.copy(),
            "s_values": self.dlr.s_values.copy(),
            "d_diag": self.dlr.d_diag.copy(),
            "A1": self.kfac_layer1.A.copy(),
            "B1": self.kfac_layer1.B.copy(),
            "A2": self.kfac_layer2.A.copy(),
            "B2": self.kfac_layer2.B.copy(),
        }

    def compute_natural_gradient(self, layer, grad_w):
        """Apply the existing K-FAC natural-gradient primitive for a layer."""
        if layer == 1:
            estimator = self.kfac_layer1
        elif layer == 2:
            estimator = self.kfac_layer2
        else:
            raise ValueError("layer must be 1 or 2")
        gradient = np.asarray(grad_w, dtype=FP32)
        expected = (estimator.B.shape[0], estimator.A.shape[0])
        if gradient.shape != expected or not np.all(np.isfinite(gradient)):
            raise ValueError(f"grad_w must be finite and have shape {expected}")
        return estimator.compute_natural_gradient(gradient)
