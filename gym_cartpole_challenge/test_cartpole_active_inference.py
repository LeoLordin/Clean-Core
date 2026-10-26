import sys
import numpy as np

# ==============================================================================
# CartPole-like Physics Environment (Lagrangian Dynamics)
# ==============================================================================
class CartPoleEnv:
    """
    Standalone CartPole-like simulation environment with continuous action force.
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
# Hand-tuned state-feedback controller demo
# ==============================================================================
class CartPoleStateFeedbackController:
    """
    Fixed linear state-feedback controller with a quadratic state-error score.

    This standalone demo is not a full Active Inference implementation; the
    fixed gain does not optimize the score returned by compute_state_score.
    """
    def __init__(self, dt=0.02):
        self.dt = dt
        # Target state: pole perfectly upright (theta=0, theta_dot=0), cart centered (x=0, x_dot=0)
        self.target_state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        
        # Homeostatic precision matrix
        self.P_precision = np.diag([2.0, 1.5, 50.0, 5.0])

        self.K_feedback = np.array([1.2, 1.8, 32.0, 4.5], dtype=np.float64)

    def compute_state_score(self, state):
        """
        Compute a quadratic state-error score:
        score(s) = 0.5 * (s - s_target)^T @ P_precision @ (s - s_target)
        """
        error = state - self.target_state
        return float(0.5 * error.T @ self.P_precision @ error)

    def compute_efe(self, state):
        """Backward-compatible name for the quadratic state-error score."""
        return self.compute_state_score(state)

    def select_action(self, state):
        """
        Apply fixed linear state feedback and return its state-error score.
        """
        error = state - self.target_state
        raw_action = float(self.K_feedback @ error)
        action = np.clip(raw_action, -1.0, 1.0)
        state_score = self.compute_state_score(state)
        return action, state_score


# ==============================================================================
# Benchmark Test Runner
# ==============================================================================
def run_cartpole_demo(max_steps=500):
    print("================================================================================")
    print("   CARTPOLE-LIKE DEMO: HAND-TUNED LINEAR STATE FEEDBACK")
    print("================================================================================")
    
    env = CartPoleEnv()
    agent = CartPoleStateFeedbackController(dt=env.tau)
    
    # Reset environment with initial disturbance
    state = env.reset(initial_state=[0.02, 0.01, 0.05, -0.02])
    
    total_steps = 0
    theta_history = []
    x_history = []
    state_score_history = []

    print(f"{'Step':>6} | {'Cart Pos x':>12} | {'Pole Angle (deg)':>18} | {'Action Force (N)':>18} | {'State score':>12}")
    print("-" * 75)

    for step in range(1, max_steps + 1):
        action, state_score = agent.select_action(state)
        next_state, done = env.step(action)
        
        x, _, theta, _ = next_state
        theta_deg = np.degrees(theta)
        force_N = action * env.force_mag
        
        theta_history.append(abs(theta_deg))
        x_history.append(abs(x))
        state_score_history.append(state_score)
        total_steps += 1

        if step == 1 or step % 50 == 0 or step == max_steps or done:
            print(f"{step:6d} | {x:12.4f} | {theta_deg:18.4f}° | {force_N:18.4f} N | {state_score:12.6f}")

        state = next_state

        if done:
            print(f"\n[TERMINATION] CartPole failed at step {step} (angle or position limit exceeded).")
            break

    avg_theta = np.mean(theta_history)
    max_theta = np.max(theta_history)
    avg_x = np.mean(x_history)
    avg_state_score = np.mean(state_score_history)

    print("\n================================================================================")
    print("   BENCHMARK RESULTS")
    print("================================================================================")
    print(f"• Total Steps Balanced      : {total_steps} / {max_steps}")
    print(f"• Mean Absolute Pole Angle  : {avg_theta:.4f}° (Max: {max_theta:.4f}°)")
    print(f"• Mean Absolute Cart Pos x  : {avg_x:.4f} m")
    print(f"• Mean Quadratic State Score: {avg_state_score:.6f}")
    print(f"• Demo status               : {'completed' if total_steps == max_steps else 'terminated early'}")
    print("================================================================================")

    return total_steps == max_steps


ActiveInferenceCartPoleAgent = CartPoleStateFeedbackController
run_cartpole_active_inference_benchmark = run_cartpole_demo


if __name__ == "__main__":
    success = run_cartpole_demo(max_steps=500)
    sys.exit(0 if success else 1)
