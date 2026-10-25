import numpy as np
from clean_core.config import FP64, POLE

class SleepReplayAntiLivelock:
    def __init__(self, capacity=128):
        self.capacity = capacity
        self.ring_buffer = []
        self.k_check = 5

    def push_checkpoint(self, state_dict):
        if len(self.ring_buffer) >= self.capacity:
            self.ring_buffer.pop(0)
        self.ring_buffer.append(state_dict)

    def evaluate_impasse_and_rollback(self, fe_history, threshold=2.0):
        if len(fe_history) < self.k_check:
            return False, None
        recent_fe = fe_history[-self.k_check:]
        d_fe = np.diff(recent_fe)
        if np.mean(d_fe) > threshold and len(self.ring_buffer) > 0:
            return True, self.ring_buffer[0]
        return False, None
