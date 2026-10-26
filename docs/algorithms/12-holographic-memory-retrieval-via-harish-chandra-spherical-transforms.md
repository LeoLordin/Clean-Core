[Back to theory contents](../THEORY.md)

### Algorithm 12: Holographic Memory Retrieval via Harish-Chandra Spherical Transforms
* **Inputs:** Query coordinate $`\boldsymbol{\mu}_i^{(l)}`$, memory buffer $`\mathcal{M}_{\mathrm{fast}}`$, spectral amplitudes $`\tilde{A}_m(\lambda, \mathbf{b})`$.
* **Outputs:** Associative context recall vector $`\mathbf{H}_{\mathrm{recall}}(\boldsymbol{\mu}_i^{(l)})`$.
* **Complexity:** $`O(M_{\mathrm{capacity}} \cdot (d-1))`$.
* **Execution Steps:**
  1. Evaluate Gelfand-Graev spherical kernels $`\phi_{\lambda, \mathbf{b}}(\boldsymbol{\mu}_i^{(l)})`$ across sampled frequency-angle pairs $`(\lambda, \mathbf{b})`$.
  2. Compute geodesic phase multipliers: $`\exp(-i \lambda d_{\mathcal{H}^{d-1}}(\boldsymbol{\mu}_i^{(l)}, \boldsymbol{\mu}_m) / \hbar_{\mathrm{cog}})`$.
  3. Integrate spherical convolution with amplitudes $`\tilde{A}_m(\lambda, \mathbf{b})`$ under Plancherel density $`|c(\lambda)|^{-2}`$.
  4. Return associative vector $`\mathbf{H}_{\mathrm{recall}}`$ to the inference loop.

---
