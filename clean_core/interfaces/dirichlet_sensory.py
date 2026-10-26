import numpy as np
from clean_core.config import D_LORENTZ, D_SPATIAL, FP64, POLE
from clean_core.geometry.lorentz import exp_map

class DirichletSensoryLayer:
    """Project vector observations through a Dirichlet prior-posterior update."""

    def __init__(self, alpha_params=None):
        self.alpha = np.asarray(
            alpha_params if alpha_params is not None else np.ones(D_SPATIAL),
            dtype=FP64,
        )
        if self.alpha.shape != (D_SPATIAL,) or np.any(self.alpha <= 0.0):
            raise ValueError(f"alpha_params must contain {D_SPATIAL} positive values")

    def ground_sensory_vector(self, raw_input):
        values = np.asarray(raw_input, dtype=FP64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("raw_input must be a non-empty one-dimensional vector")
        if not np.all(np.isfinite(values)):
            raise ValueError("raw_input must contain only finite values")
        if values.size < D_SPATIAL:
            values = np.pad(values, (0, D_SPATIAL - values.size))
        else:
            values = values[:D_SPATIAL]

        evidence = np.exp(values - np.max(values))
        evidence /= np.sum(evidence)
        posterior_alpha = self.alpha + evidence
        probs = posterior_alpha / np.sum(posterior_alpha)
        v_tangent = np.zeros(D_LORENTZ, dtype=FP64)
        v_tangent[1:] = probs
        return exp_map(POLE, v_tangent)
