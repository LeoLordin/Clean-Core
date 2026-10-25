import numpy as np

D_LORENTZ = 16
D_SPATIAL = 15
R_RANK = 4
H_DIM = 32
D_ACT = 8
D_SCM = 16
M_CROSSBAR = 64
D_HIST = 128
K_MAX = 1024

FP64 = np.float64
FP32 = np.float32
COMPLEX_MEM = np.complex64
UINT8 = np.uint8

ETA_L = np.array([-1.0] + [1.0] * D_SPATIAL, dtype=FP64)
POLE = np.array([1.0] + [0.0] * D_SPATIAL, dtype=FP64)

E_BASIS = np.zeros((D_SPATIAL, D_LORENTZ), dtype=FP64)
for k in range(D_SPATIAL):
    E_BASIS[k, k + 1] = 1.0
