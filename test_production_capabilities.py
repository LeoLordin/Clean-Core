import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import clean_core as cc

print("================================================================================")
print("   ЧЕСТНОЕ ТЕСТИРОВАНИЕ ПРОДАКШЕН-МОДУЛЕЙ CLEAN CORE 5.0 (clean_core_package)")
print("================================================================================")

# 1. Lorentz & Geodesics
mu = cc.POLE.copy()
v_tangent = np.array([0.0, 0.1] + [0.0]*14, dtype=cc.FP64)
t0 = time.time()
for _ in range(10000):
    mu = cc.exp_map(mu, v_tangent)
t1 = time.time()
drift = abs(cc.minkowski_norm_sq(mu) - (-1.0))
print(f"[МОДУЛЬ 1/6] Геометрия H^16: Дрейф <mu, mu>_L = {drift:.2e} | Время = {(t1-t0)*1000:.2f} мс")

# 2. AIXI
aixi = cc.SolomonoffAIXIBridge(num_actions=8)
obs = np.random.randn(15)
a_star, eps_safe, m_poly = aixi.solve_polyhedral_aixi(obs)
print(f"[МОДУЛЬ 2/6] AIXI Polyhedral Decider: a*={a_star}, eps_safe={eps_safe:.4f}")

# 3. Agent 1000 steps
agent = cc.CleanCoreProductionAgent(num_neurons=5)
t0 = time.time()
for i in range(1000):
    agent.train_step({"s0": float(i % 10), "s1": 0.5})
t1 = time.time()
print(f"[МОДУЛЬ 3/6] Agent 1000 steps: {(t1-t0):.2f} сек ({(1000/(t1-t0)):.1f} TPS)")
print("================================================================================")
