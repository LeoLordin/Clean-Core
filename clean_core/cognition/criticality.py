import numpy as np

from clean_core.config import FP64


class SOCCriticalityRegulator:
    def __init__(
        self,
        epsilon_min=1e-3,
        gamma_soc=0.05,
        n_min=8,
        eta_alpha=0.01,
        alpha_damp=0.05,
        alpha_min=0.05,
        delta_soc=1e-6,
        delta_alpha_max=0.01,
        target_alpha=1.5,
    ):
        if epsilon_min <= 0.0 or n_min < 1 or delta_soc <= 0.0:
            raise ValueError("epsilon_min, n_min, and delta_soc must be positive")
        if not 0.0 <= gamma_soc <= 1.0 or eta_alpha < 0.0 or delta_alpha_max < 0.0:
            raise ValueError("gamma_soc must be in [0, 1]; rates must be non-negative")
        if alpha_damp < alpha_min:
            raise ValueError("alpha_damp must be at least alpha_min")

        self.epsilon_min = epsilon_min
        self.gamma_soc = gamma_soc
        self.n_min = n_min
        self.eta_alpha = eta_alpha
        self.alpha_damp = alpha_damp
        self.alpha_min = alpha_min
        self.delta_soc = delta_soc
        self.delta_alpha_max = delta_alpha_max
        self.target_alpha = target_alpha
        self.log_tail_mean = 0.0
        self.alpha_measured = None

    def update(self, errors, precision_matrices, dt):
        if len(errors) != len(precision_matrices):
            raise ValueError("errors and precision_matrices must have equal lengths")
        if dt <= 0.0 or not np.isfinite(dt):
            raise ValueError("dt must be finite and positive")

        tail_norms = []
        for error, precision in zip(errors, precision_matrices):
            error = np.asarray(error, dtype=FP64)
            precision = np.asarray(precision, dtype=FP64)
            if precision.shape != (error.size, error.size):
                raise ValueError("each precision matrix must match its error vector")
            if not np.all(np.isfinite(error)) or not np.all(np.isfinite(precision)):
                raise ValueError("errors and precision matrices must be finite")
            norm_sq = float(error @ precision @ error)
            if norm_sq < -1e-10:
                raise ValueError("precision-weighted squared error must be non-negative")
            norm = np.sqrt(max(0.0, norm_sq))
            if norm >= self.epsilon_min:
                tail_norms.append(norm)

        if len(tail_norms) < self.n_min:
            return self.alpha_damp

        observed_log_tail_mean = float(np.mean(np.log(np.asarray(tail_norms) / self.epsilon_min)))
        self.log_tail_mean = (
            (1.0 - self.gamma_soc) * self.log_tail_mean
            + self.gamma_soc * observed_log_tail_mean
        )
        self.alpha_measured = 1.0 + 1.0 / max(self.delta_soc, self.log_tail_mean)
        delta_alpha = dt * (1.2 * self.eta_alpha) * (self.target_alpha - self.alpha_measured)
        delta_alpha = float(
            np.clip(delta_alpha, -self.delta_alpha_max, self.delta_alpha_max)
        )
        self.alpha_damp = max(self.alpha_min, self.alpha_damp + delta_alpha)
        return self.alpha_damp