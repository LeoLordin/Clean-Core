import numpy as np
from clean_core.config import M_CROSSBAR, UINT8, FP32
from clean_core.core.inference import precision_weighted_spike_mask

class NeuromorphicCrossbarEncoder:
    def __init__(self, m_size=M_CROSSBAR):
        self.m_size = m_size
        self.G_conductance = np.random.uniform(0.1, 1.0, (m_size, m_size)).astype(FP32)

    def encode_spikes_ste(
        self,
        v_tangent,
        prediction_error=None,
        precision=None,
        threshold=0.0,
        beta=5.0,
    ):
        v_tangent = np.asarray(v_tangent, dtype=FP32)
        if v_tangent.ndim != 1 or v_tangent.size < self.m_size:
            raise ValueError(f"v_tangent must be a vector of length >= {self.m_size}")
        if (prediction_error is None) != (precision is None):
            raise ValueError("prediction_error and precision must be provided together")
        activation_mask = 1.0
        if prediction_error is not None:
            activation_mask = precision_weighted_spike_mask(
                prediction_error,
                precision,
                threshold,
                beta,
            )
        logits = np.clip(v_tangent[:self.m_size], -60.0, 60.0)
        probs = activation_mask / (1.0 + np.exp(-logits))
        spikes = (probs > 0.5).astype(UINT8)
        current = self.G_conductance[:len(spikes), :len(spikes)] @ spikes.astype(FP32)
        return spikes, current
