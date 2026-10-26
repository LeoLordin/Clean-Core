import numpy as np
from clean_core.config import D_ACT, FP64


def categorical_target_kl(posterior, target_prior):
    posterior = np.asarray(posterior, dtype=FP64)
    target_prior = np.asarray(target_prior, dtype=FP64)
    if posterior.ndim != 1 or target_prior.shape != posterior.shape or posterior.size == 0:
        raise ValueError("posterior and target_prior must be non-empty vectors of equal shape")
    if not np.all(np.isfinite(posterior)) or not np.all(np.isfinite(target_prior)):
        raise ValueError("posterior and target_prior must contain only finite values")
    if np.any(posterior < 0.0) or np.any(target_prior < 0.0):
        raise ValueError("probabilities must be non-negative")
    if np.sum(posterior) <= 0.0 or np.sum(target_prior) <= 0.0:
        raise ValueError("probability vectors must have positive total mass")

    q = posterior / np.sum(posterior)
    p = target_prior / np.sum(target_prior)
    if np.any((q > 0.0) & (p == 0.0)):
        return float("inf")
    nonzero = q > 0.0
    return float(np.sum(q[nonzero] * np.log(q[nonzero] / p[nonzero])))


class MotorEFEController:
    def __init__(self, d_act=D_ACT):
        self.d_act = d_act

    def synthesize_motor_action(self, mu_self, m_poly_aixi):
        v_spatial = mu_self[1:1 + self.d_act]
        a_efe = np.tanh(v_spatial + m_poly_aixi[:len(v_spatial)])
        return a_efe

    def target_divergence(self, predicted_state_distribution, target_distribution):
        """Compute categorical target KL; no state predictor is included here."""
        return categorical_target_kl(predicted_state_distribution, target_distribution)
