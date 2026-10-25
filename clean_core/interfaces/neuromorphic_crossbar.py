import numpy as np
from clean_core.config import M_CROSSBAR, UINT8, FP32

class NeuromorphicCrossbarEncoder:
    def __init__(self, m_size=M_CROSSBAR):
        self.m_size = m_size
        self.G_conductance = np.random.uniform(0.1, 1.0, (m_size, m_size)).astype(FP32)

    def encode_spikes_ste(self, v_tangent):
        probs = 1.0 / (1.0 + np.exp(-v_tangent[:self.m_size]))
        spikes = (probs > 0.5).astype(UINT8)
        current = self.G_conductance[:len(spikes), :len(spikes)] @ spikes.astype(FP32)
        return spikes, current
