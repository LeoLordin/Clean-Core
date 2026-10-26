[Back to theory contents](../THEORY.md)

### Algorithm 10: System-1 Semantic Query Relaxation & Holographic Retrieval
* **Inputs:** Query probe $`\mathbf{x}_{\mathrm{query}}`$, holographic phase memory $`\mathbf{H}_{\mathrm{hologram}}`$.
* **Outputs:** Semantic target action response $`\mathbf{a}_{\mathrm{target}}`$.
* **Complexity:** $`O(K_{\mathrm{relax}} \cdot |E| + M_{\mathrm{capacity}} \cdot (d-1))`$.
* **Execution Steps:**
  1. Pin $`\mathbf{x}_{\mathrm{query}}`$ to Dirichlet sensory layer $`\boldsymbol{\mu}_0^{(0)}`$.
  2. Lock synaptic adaptation: $`\Delta \mathbf{W} \equiv \mathbf{0}, \Delta E \equiv 0`$.
  3. Execute zero-inertia System-1 relaxation: perform $`K_{\mathrm{relax}} = 4`$ consecutive geodesic steps driven by bottom-up forces $`\tilde{\mathbf{v}}_{\mathrm{bottom-up}}`$.
  4. Reach stationary state at top layer $`\boldsymbol{\mu}^{(L)}`$.
  5. Compute Harish-Chandra spherical convolution: evaluate phase interference between $`\boldsymbol{\mu}^{(L)}`$ and spectral modes $`\tilde{A}_m`$ weighted by Plancherel measure $`|c(\lambda)|^{-2}`$.
  6. Synthesize associative recall vector $`\mathbf{H}_{\mathrm{recall}}`$.
  7. Project stationary state and associative context through motor decoder to yield $`\mathbf{a}_{\mathrm{target}}`$.

---
