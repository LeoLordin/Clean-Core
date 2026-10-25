import numpy as np
from clean_core.config import D_ACT, FP64

class MotorEFEController:
    def __init__(self, d_act=D_ACT):
        self.d_act = d_act

    def synthesize_motor_action(self, mu_self, m_poly_aixi):
        v_spatial = mu_self[1:1 + self.d_act]
        a_efe = np.tanh(v_spatial + m_poly_aixi[:len(v_spatial)])
        return a_efe
