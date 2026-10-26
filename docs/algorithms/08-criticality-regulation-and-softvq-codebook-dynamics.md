[Back to theory contents](../THEORY.md)

### Algorithm 8: Criticality Regulation & SoftVQ Codebook Dynamics
* **Inputs:** Prediction error distribution $`\tilde{\boldsymbol{\varepsilon}}_i`$ across all network neurons.
* **Outputs:** Dynamic damping coefficient $`\alpha_{\mathrm{damp}}(t+\Delta t)`$, updated codebook $`\mathcal{C}`$.
* **Complexity:** $`O(|V| + |\mathcal{C}| \cdot (d-1))`$.
* **Execution Steps:**
  1. Identify avalanche vertices $`\mathcal{S}_{\mathrm{tail}}`$ where $`\|\tilde{\boldsymbol{\varepsilon}}_i\|_{\tilde{\mathbf{\Pi}}} \ge \varepsilon_{\min}`$.
  2. If $`|\mathcal{S}_{\mathrm{tail}}| \ge N_{\min}`$, smooth log-tail mean:  

```math
\overline{L}_{\mathrm{tail}}(t) = (1 - \gamma_{\mathrm{soc}}) \overline{L}_{\mathrm{tail}}(t-\Delta t) + \gamma_{\mathrm{soc}} \mathrm{mean}_{i \in \mathcal{S}_{\mathrm{tail}}} \ln(\|\tilde{\boldsymbol{\varepsilon}}_i\| / \varepsilon_{\min})
```

  3. Compute instantaneous Pareto exponent: $`\hat{\alpha}_{\mathrm{measured}} = 1 + 1/\overline{L}_{\mathrm{tail}}`$.
  4. Calculate damping adjustment: $`\Delta \alpha = \mathrm{clip}(\Delta t \cdot 1.2 \eta_\alpha (1.5 - \hat{\alpha}_{\mathrm{measured}}), -\Delta \alpha_{\max}, +\Delta \alpha_{\max})`$.
  5. Update damping with lower bound: $`\alpha_{\mathrm{damp}} \leftarrow \max(\alpha_{\min}, \alpha_{\mathrm{damp}} + \Delta \alpha)`$.
  6. Apply exponential decay to token usage: $`u_k \leftarrow \gamma_{\mathrm{usage}} u_k`$.
  7. Prune tokens below $`\theta_{\mathrm{prune}} = 0.05`$ outside $`\mathcal{K}_{\mathrm{protected}}`$, charging Landauer penalty $`\mathcal{L}_{\mathrm{Landauer}}`$.
  8. If prediction error exceeds $`\theta_{\mathrm{spawn}}`$, spawn a new concept centroid projected from the pole.

---
