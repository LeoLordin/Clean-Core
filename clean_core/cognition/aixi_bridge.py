import numpy as np
from clean_core.config import D_ACT, FP64

class SolomonoffAIXIBridge:
    def __init__(self, num_actions=D_ACT):
        self.num_actions = num_actions
        self.k_step = 0

    def solve_polyhedral_aixi(self, obs_vector, r_max=1.0, gamma=0.95):
        self.k_step += 1
        halted_kraft_sum = 0.999 + 0.001 * np.tanh(self.k_step * 0.1)
        omega_unresolved = max(1e-6, 1.0 - halted_kraft_sum)
        q_min = np.zeros(self.num_actions, dtype=FP64)
        for a in range(self.num_actions):
            q_min[a] = np.tanh(obs_vector[a % len(obs_vector)]) * 0.4 + 0.5
        a_star = int(np.argmax(q_min))
        q_min[a_star] += 0.5
        q_max = q_min.copy()
        for a in range(self.num_actions):
            if a != a_star:
                q_max[a] = q_min[a] + omega_unresolved * (r_max / (1.0 - gamma))
            else:
                q_max[a] = q_min[a]
        other_max_val = np.max([q_max[a] for a in range(self.num_actions) if a != a_star])
        eps_safe = float(q_min[a_star] - other_max_val)
        denom = max(1e-4, abs(eps_safe))
        m_poly = np.zeros(self.num_actions, dtype=FP64)
        m_poly[a_star] = max(0.1, eps_safe) / denom
        return a_star, eps_safe, m_poly
