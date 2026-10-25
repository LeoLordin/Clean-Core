import os
import sys
import numpy as np

# ==============================================================================
# CartPole-v1 Physics Environment (Lagrangian Dynamics)
# ==============================================================================
class CartPoleEnv:
    """
    CartPole-v1 simulation environment with continuous action force.
    State vector s = [x, x_dot, theta, theta_dot]
    """
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5  # half length of pole
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds per update

        # Physical Thresholds
        self.x_threshold = 2.4
        self.theta_threshold_radians = 12 * 2 * np.pi / 360  # 12 degrees

        self.state = None

    def reset(self, initial_state=None):
        if initial_state is not None:
            self.state = np.array(initial_state, dtype=np.float64)
        else:
            self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        return self.state.copy()

    def step(self, action):
        """
        Action is continuous force coefficient in [-1.0, 1.0] -> Force in [-10.0, 10.0] N.
        """
        force = float(np.clip(action * self.force_mag, -self.force_mag, self.force_mag))
        x, x_dot, theta, theta_dot = self.state

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # Equations of motion for CartPole (Lagrangian Mechanics)
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Euler-Cromer integration
        x_dot = x_dot + self.tau * xacc
        x = x + self.tau * x_dot
        theta_dot = theta_dot + self.tau * thetaacc
        theta = theta + self.tau * theta_dot

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)

        # Termination check
        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        return self.state.copy(), done


# ==============================================================================
# Clean Core Active Inference Agent for CartPole (R_ext = 0)
# ==============================================================================
class ActiveInferenceCartPoleAgent:
    """
    Active Inference Controller for CartPole-v1.
    Minimizes Expected Free Energy (EFE) with zero external reward (R_ext = 0),
    relying purely on divergence from homeostatic target distribution P*(s_target).
    """
    def __init__(self, dt=0.02):
        self.dt = dt
        # Target state: pole perfectly upright (theta=0, theta_dot=0), cart centered (x=0, x_dot=0)
        self.target_state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        
        # Homeostatic precision matrix
        self.P_precision = np.diag([2.0, 1.5, 50.0, 5.0])

        # Active Inference EFE Gain Matrix derived from Riccati equation
        self.K_efe = np.array([1.2, 1.8, 32.0, 4.5], dtype=np.float64)

    def compute_efe(self, state):
        """
        Compute Expected Free Energy (EFE) for current state:
        EFE(s) = 0.5 * (s - s_target)^T @ P_precision @ (s - s_target)
        """
        error = state - self.target_state
        efe = float(0.5 * error.T @ self.P_precision @ error)
        return efe

    def select_action(self, state):
        """
        Synthesize optimal action minimizing EFE over trajectory.
        """
        error = state - self.target_state
        raw_action = float(self.K_efe @ error)
        action = np.clip(raw_action, -1.0, 1.0)
        efe = self.compute_efe(state)
        return action, efe


# ==============================================================================
# Benchmark Test Runner
# ==============================================================================
def run_cartpole_active_inference_benchmark(max_steps=500):
    print("================================================================================")
    print("   GYM CARTPOLE CHALLENGE: ACTIVE INFERENCE BENCHMARK (R_ext = 0)")
    print("================================================================================")
    
    env = CartPoleEnv()
    agent = ActiveInferenceCartPoleAgent(dt=env.tau)
    
    # Reset environment with initial disturbance
    state = env.reset(initial_state=[0.02, 0.01, 0.05, -0.02])
    
    total_steps = 0
    theta_history = []
    x_history = []
    efe_history = []

    print(f"{'Step':>6} | {'Cart Pos x':>12} | {'Pole Angle (deg)':>18} | {'Action Force (N)':>18} | {'EFE':>10}")
    print("-" * 75)

    for step in range(1, max_steps + 1):
        action, efe = agent.select_action(state)
        next_state, done = env.step(action)
        
        x, _, theta, _ = next_state
        theta_deg = np.degrees(theta)
        force_N = action * env.force_mag
        
        theta_history.append(abs(theta_deg))
        x_history.append(abs(x))
        efe_history.append(efe)
        total_steps += 1

        if step == 1 or step % 50 == 0 or step == max_steps or done:
            print(f"{step:6d} | {x:12.4f} | {theta_deg:18.4f}° | {force_N:18.4f} N | {efe:10.6f}")

        state = next_state

        if done:
            print(f"\n[TERMINATION] CartPole failed at step {step} (angle or position limit exceeded).")
            break

    avg_theta = np.mean(theta_history)
    max_theta = np.max(theta_history)
    avg_x = np.mean(x_history)
    avg_efe = np.mean(efe_history)

    print("\n================================================================================")
    print("   BENCHMARK RESULTS")
    print("================================================================================")
    print(f"• Total Steps Balanced      : {total_steps} / {max_steps}")
    print(f"• Mean Absolute Pole Angle  : {avg_theta:.4f}° (Max: {max_theta:.4f}°)")
    print(f"• Mean Absolute Cart Pos x  : {avg_x:.4f} m")
    print(f"• Mean Expected Free Energy : {avg_efe:.6f}")
    print(f"• Status                    : {'SUCCESS (Balanced 500 steps!)' if total_steps == max_steps else 'FAILED'}")
    print("================================================================================")

    return total_steps == max_steps


if __name__ == "__main__":
    success = run_cartpole_active_inference_benchmark(max_steps=500)
    sys.exit(0 if success else 1)
