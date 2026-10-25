import numpy as np
from clean_core.config import D_LORENTZ, D_SPATIAL, H_DIM, R_RANK, M_CROSSBAR, FP64, FP32, COMPLEX_MEM

class NeuronMicrostate:
    def __init__(self, neuron_id=0):
        self.id = neuron_id
        self.mu = np.array([1.0] + [0.0] * D_SPATIAL, dtype=FP64)
        self.p = np.zeros(D_LORENTZ, dtype=FP64)
        self.v = np.zeros(D_LORENTZ, dtype=FP64)
        self.v_hat = np.zeros(D_SPATIAL, dtype=FP64)
        self.w = np.zeros(D_SPATIAL, dtype=FP64)
        self.Omega = 1.0
        self.grad_ln_Omega = np.zeros(D_SPATIAL, dtype=FP64)
        self.W1 = np.random.randn(H_DIM, D_SPATIAL).astype(FP32) * 0.01
        self.W2 = np.random.randn(D_SPATIAL, H_DIM).astype(FP32) * 0.01
        self.eps_tilde = np.zeros(D_SPATIAL, dtype=FP32)
        self.Pi_precision = np.eye(D_SPATIAL, dtype=FP32)
        self.d_diag = np.ones(D_SPATIAL, dtype=FP32)
        self.V_basis = np.zeros((D_SPATIAL, R_RANK), dtype=FP32)
        self.psi = np.zeros(H_DIM, dtype=COMPLEX_MEM)
        self.lambda_spike = np.zeros(M_CROSSBAR, dtype=FP32)
        self.padding = np.zeros(8, dtype=np.uint8)

    def get_byte_size(self):
        return (self.mu.nbytes + self.p.nbytes + self.v.nbytes + self.v_hat.nbytes +
                self.w.nbytes + 8 + self.grad_ln_Omega.nbytes +
                self.W1.nbytes + self.W2.nbytes + self.eps_tilde.nbytes +
                self.Pi_precision.nbytes + self.d_diag.nbytes + self.V_basis.nbytes +
                self.psi.nbytes + self.lambda_spike.nbytes + self.padding.nbytes)
