import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import clean_core as cc

print("================================================================================")
print("   CLEAN CORE 5.0 — PRODUCTION CAPABILITIES BENCHMARK (clean_core_package)")
print("================================================================================")

# 1. Lorentz & Geodesics (10,000 Stochastic Manifold Steps)
mu = cc.POLE.copy()
np.random.seed(42)
t0 = time.time()
for _ in range(10000):
    # Случайный касательный вектор шума
    v_raw = np.random.normal(0.0, 0.01, size=15)
    v_tangent = np.array([0.0] + list(v_raw), dtype=cc.FP64)
    # Ортогонализация к текущей точке mu
    v_tangent = v_tangent + cc.minkowski_dot(mu, v_tangent) * mu
    mu = cc.exp_map(mu, v_tangent)
t1 = time.time()
drift = abs(cc.minkowski_norm_sq(mu) - (-1.0))
print(f"[MODULE 1/3] Geometry H^16: Drift <mu, mu>_L = {drift:.2e} | Time = {(t1-t0)*1000:.2f} ms")

# 2. AIXI
aixi = cc.SolomonoffAIXIBridge(num_actions=8)
obs = np.random.randn(15)
a_star, eps_safe, m_poly = aixi.solve_polyhedral_aixi(obs)
print(f"[MODULE 2/3] AIXI Polyhedral Decider: a*={a_star}, eps_safe={eps_safe:.4f}")

# 3. Agent 1000 steps
agent = cc.CleanCoreProductionAgent(num_neurons=5)
t0 = time.time()
for i in range(1000):
    agent.train_step({"s0": float(i % 10), "s1": 0.5})
t1 = time.time()
print(f"[MODULE 3/3] Agent 1000 steps: {(t1-t0):.2f} sec ({(1000/(t1-t0)):.1f} TPS)")
print("================================================================================")
