import numpy as np
from clean_core.config import FP64, POLE
from clean_core.geometry.lorentz import exp_map

class DirichletSensoryLayer:
    def __init__(self, alpha_params=None):
        self.alpha = alpha_params if alpha_params is not None else np.ones(15, dtype=FP64)

    def ground_sensory_vector(self, raw_input):
        probs = np.exp(raw_input - np.max(raw_input))
        probs /= np.sum(probs)
        v_tangent = np.array([0.0] + list(probs[:15]), dtype=FP64)
        return exp_map(POLE, v_tangent)
