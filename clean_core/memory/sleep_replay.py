import copy
from collections import deque

import numpy as np

class SleepReplayAntiLivelock:
    def __init__(self, capacity=10_000):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.ring_buffer = deque(maxlen=capacity)
        self.k_check = 5

    def push_checkpoint(self, state_dict):
        self.ring_buffer.append(copy.deepcopy(state_dict))

    def evaluate_impasse_and_rollback(self, fe_history, threshold=2.0):
        if len(fe_history) < self.k_check:
            return False, None
        recent_fe = np.asarray(fe_history[-self.k_check:], dtype=np.float64)
        if not np.all(np.isfinite(recent_fe)):
            raise ValueError("fe_history must contain only finite values")
        d_fe = np.diff(recent_fe)
        if np.mean(d_fe) > threshold and len(self.ring_buffer) > 0:
            return True, copy.deepcopy(self.ring_buffer[-1])
        return False, None
