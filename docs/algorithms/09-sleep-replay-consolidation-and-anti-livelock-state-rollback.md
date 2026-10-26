[Back to theory contents](../THEORY.md)

### Algorithm 9: Sleep Replay Consolidation & Anti-Livelock State Rollback
* **Inputs:** Episodic memory buffer $`\mathcal{M}_{\mathrm{fast}}`$, free energy derivative $`\dot{\bar{\mathcal{J}}}(t)`$, VRAM ring buffer.
* **Outputs:** Re-consolidated synaptic tensors or restored checkpoint state.
* **Complexity:** $`O(K_{\mathrm{clusters}} \cdot |\mathcal{B}_{\mathrm{replay}}|)`$ during sleep, $`O(\mathrm{Size}(\text{Snapshot}))`$ during rollback.
* **Execution Steps:**
  1. **Sleep Phase (Sleep Replay):**
     * Uniformly sample episodic memory clusters using Vitter's reservoir algorithm.
     * Execute state reconstruction through projection perceptrons.
     * Perform natural gradient descent with step-size $`-\eta_{\mathrm{replay}}`$.
  2. **Wake Phase (Anti-Livelock Watchdog):**
     * Evaluate running free energy derivative $`\dot{\bar{\mathcal{J}}}(t)`$.
     * If $`\dot{\bar{\mathcal{J}}} > \theta_{\mathrm{impasse}}`$ and cooldown timer $`t - t_{\mathrm{last\_reset}} \ge \tau_{\mathrm{cooldown}}`$:
       * Restore full neuron microstates and K-FAC factors from checkpoint $`K_{\mathrm{check}} = 5`$ in VRAM ring buffer.
       * Re-ground Dirichlet sensory layer with current environmental stream.
       * Anneal Langevin temperature: $`\mathbf{T}_i \leftarrow T_{\max} \cdot (0.8)^N`$.
       * Double cooldown timer: $`\tau_{\mathrm{cooldown}} \leftarrow \min(\tau_{\max}, 2.0 \cdot \tau_{\mathrm{cooldown}})`$.
       * Increment failure counter $`N_{\mathrm{consecutive\_resets}}`$.
     * If $`\dot{\bar{\mathcal{J}}} \le 0`$, reset failure counter to zero and restore base cooldown $`\tau_{\mathrm{cooldown}, 0}`$.

---
