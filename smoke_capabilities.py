import time

import numpy as np

import clean_core as cc


def run_smoke_check():
    np.random.seed(42)

    point = cc.POLE.copy()
    started = time.perf_counter()
    for _ in range(10_000):
        tangent = np.zeros(cc.D_LORENTZ, dtype=cc.FP64)
        tangent[1:] = np.random.normal(0.0, 0.01, size=cc.D_SPATIAL)
        tangent += cc.minkowski_dot(point, tangent) * point
        point = cc.exp_map(point, tangent)
    geometry_elapsed = time.perf_counter() - started
    geometry_drift = abs(cc.minkowski_norm_sq(point) + 1.0)
    assert geometry_drift < 1e-12

    heuristic = cc.BoundedActionHeuristic(num_actions=cc.D_ACT)
    action, score_margin, motor_pressure = heuristic.select_action(np.random.randn(cc.D_SPATIAL))
    assert 0 <= action < cc.D_ACT
    assert np.isfinite(score_margin)
    assert motor_pressure.shape == (cc.D_ACT,)

    agent = cc.CleanCoreProductionAgent(num_neurons=5)
    started = time.perf_counter()
    for step in range(1_000):
        result = agent.train_step({"s0": float(step % 10), "s1": 0.5})
    agent_elapsed = time.perf_counter() - started
    assert all(abs(cc.minkowski_norm_sq(neuron.mu) + 1.0) < 1e-12 for neuron in agent.neurons)
    assert np.all(np.isfinite(result["motor_action"]))

    print("Clean Core smoke check: PASS")
    print(f"Hyperboloid drift after 10,000 steps: {geometry_drift:.3e}")
    print(f"Bounded action heuristic: action={action}, score_gap={score_margin:.4f}")
    print(f"Agent 1,000-step loop: {agent_elapsed:.3f} s ({1_000 / agent_elapsed:.1f} steps/s)")
    print(f"Geometry loop: {geometry_elapsed:.3f} s")


if __name__ == "__main__":
    run_smoke_check()