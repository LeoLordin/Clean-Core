[Back to theory contents](../THEORY.md)

### Algorithm 4: Fast Sweeping Eikonal Wavefront Solver
* **Inputs:** Adjacency matrix $`E_{ij}`$, vertex coordinates $`\boldsymbol{\mu}_i`$, refractive index $`n_i = 1/\Omega_i`$, target coordinate $`\boldsymbol{\mu}_{\mathrm{target}}`$.
* **Outputs:** Discrete scalar action field $`S_i`$ for all vertices $`i \in V`$.
* **Complexity:** $`O(K_{\mathrm{sweep}} \cdot |E|)`$.
* **Execution Steps:**
  1. Initialize action field: $`S(\boldsymbol{\mu}_{\mathrm{target}}) \leftarrow 0.0`$, all other vertices $`S_i \leftarrow +\infty`$.
  2. Iterate across graph topological sweeps:
     * Extract neighbor values $`\mathcal{N}(i)`$ and chordal distances $`d_k = d_{\mathcal{H}^{d-1}}(\boldsymbol{\mu}_i, \boldsymbol{\mu}_k)`$.
     * Sort neighbor values in ascending order: $`S_{(1)} \le S_{(2)} \le \dots \le S_{(m)}`$.
     * Determine active neighbor count $`m`$ satisfying Godunov causality $`S_i > S_{(m)}`$.
     * Compute candidate: $`S_i^{\mathrm{new}} = \frac{B_m + \sqrt{B_m^2 - A_m(C_m - n_i^2)}}{A_m}`$ incorporating viscosity $`\varepsilon_{\mathrm{visc}} (\Delta_{\mathcal{H}^{d-1}} S)_i`$.
     * Update monotonically: $`S_i \leftarrow \min(S_i, S_i^{\mathrm{new}})`$.
  3. Check convergence criterion: $`\max_i |S_i^{(k+1)} - S_i^{(k)}| < 10^{-6}`$.
  4. Compute discrete Riemannian gradient $`\nabla_{\mathcal{H}^{d-1}} S_i`$ via upwind finite differences.

---
