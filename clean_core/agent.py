import pickle
import numpy as np
from clean_core.config import POLE, D_SPATIAL, FP64
from clean_core.geometry.lorentz import streaming_frechet_barycenter
from clean_core.core.neuron import NeuronMicrostate
from clean_core.core.inference import system1_fast_inference_step
from clean_core.cognition.aixi_bridge import BoundedActionHeuristic
from clean_core.cognition.goedel_reflection import GoedelEgoReflectionEngine
from clean_core.cognition.causal_scm import NOTEARSCausalSCM
from clean_core.memory.harish_chandra import HarishChandraHolographicMemory
from clean_core.interfaces.dirichlet_sensory import DirichletSensoryLayer
from clean_core.interfaces.motor_efe import MotorEFEController

class CleanCoreProductionAgent:
    def __init__(self, num_neurons=5):
        self.neurons = [NeuronMicrostate(i) for i in range(num_neurons)]
        self.sensory = DirichletSensoryLayer()
        self.action_heuristic = BoundedActionHeuristic()
        self.goedel = GoedelEgoReflectionEngine()
        self.scm = NOTEARSCausalSCM()
        self.memory = HarishChandraHolographicMemory()
        self.motor = MotorEFEController()
        self.fe_history = []
        self.barycenter_L1 = POLE.copy()

    def train_step(self, sensory_dict, target_motor_action=None):
        raw_v = np.array(list(sensory_dict.values()), dtype=FP64)
        if raw_v.size == 0 or not np.all(np.isfinite(raw_v)):
            raise ValueError("sensory_dict must contain finite numeric observations")
        if len(raw_v) < 15:
            raw_v = np.pad(raw_v, (0, 15 - len(raw_v)))
        mu_sensory = self.sensory.ground_sensory_vector(raw_v)
        self.barycenter_L1 = streaming_frechet_barycenter(
            self.barycenter_L1, [neuron.mu for neuron in self.neurons]
        )
        barycenter_L1 = self.barycenter_L1
        eps_list = []
        for n in self.neurons:
            _, eps = system1_fast_inference_step(n, mu_sensory, barycenter_L1)
            eps_list.append(eps)
        selected_action, score_margin, m_poly = self.action_heuristic.select_action(raw_v)
        mu_self, T_intro = self.goedel.step_ego_reflection(eps_list, barycenter_L1)
        a_motor = self.motor.synthesize_motor_action(mu_self, m_poly)
        state_error = float(np.mean([np.sum(e**2) for e in eps_list]))
        self.fe_history.append(state_error)
        return {
            "selected_action": selected_action,
            "score_margin": score_margin,
            "motor_action": a_motor,
            "state_error": state_error,
        }

    def speak_or_interact(self, text_prompt):
        char_codes = [ord(c) for c in text_prompt[:15]]
        if len(char_codes) < 15:
            char_codes += [0] * (15 - len(char_codes))
        info = self.train_step({f"char_{i}": val for i, val in enumerate(char_codes)})
        d_pole = float(np.linalg.norm(self.neurons[0].mu[1:]))
        return f"[Clean Core 5.0]: Acknowledged (Stable attractor, d_pole={d_pole:.4f})"

    def save_checkpoint(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_checkpoint(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
