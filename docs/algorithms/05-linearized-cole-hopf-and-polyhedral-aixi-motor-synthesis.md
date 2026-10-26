[Back to theory contents](../THEORY.md)

### Algorithm 5: Linearized Cole-Hopf & Polyhedral AIXI Motor Synthesis
> **Implementation status:** Partial prototype. It includes graph diffusion, finite-model bounds, and a bounded motor update, but not the continuous PDE or complete expected-free-energy gradient.
* **Inputs:** Top-layer coordinate $`\boldsymbol{\mu}^{(L)}`$, goal potentials $`\psi_k`$, optimal discrete action $`a^{*}`$, social vector $`\mathbf{a}_{\mathrm{ToM}}^{*}`$.
* **Outputs:** Continuous motor control vector $`\mathbf{a}(t+\Delta t) \in [-1, 1]^{d_{\mathrm{act}}}`$.
* **Complexity:** $`O(d_{\mathrm{act}} \cdot (d-1))`$.
* **Execution Steps:**
  1. Superimpose goal wavefunctions: $`\psi_{\mathrm{total}} = \sum_k w_k \psi_k`$.
  2. Compute optical geodesic gradient: $`\mathbf{w}_{\mathrm{geodesic}}^{(L)} = \lambda \sigma^2 E(\boldsymbol{\mu}^{(L)})^\top \boldsymbol{\eta}_{\mathcal{L}} \frac{\nabla \psi_{\mathrm{total}}}{\psi_{\mathrm{total}}}`$.
  3. Calculate polyhedral AIXI safety margin $`\varepsilon_{\mathrm{safe}}`$ and pressure vector $`\mathbf{m}_{\mathrm{poly}}(a^{*})`$.
  4. Synthesize target motor action:  

```math
\mathbf{a}_{\mathrm{target}} = \tanh\left( \mathbf{W}_{\mathrm{motor}} \mathbf{w}_{\mathrm{geodesic}}^{(L)} + \mathbf{W}_{\mathrm{poly}} \mathbf{m}_{\mathrm{poly}}(a^{*}) + \mathbf{W}_{\mathrm{social}} \mathbf{a}_{\mathrm{ToM}}^{*} + \mathbf{b}_{\mathrm{motor}} \right)
```

  5. Evaluate motor EFE gradient: KL divergence to target priors plus causal intervention $`\mathcal{I}_{\mathrm{Pearl}}^{\mathrm{causal}}`$ via projector $`\mathbf{P}_{\mathbf{a}^\perp}`$.
  6. Integrate velocity: $`\frac{\mathrm{d}\mathbf{a}}{\mathrm{d}t} = -\eta_a \nabla_{\mathbf{a}} \mathrm{EFE}`$.
  7. Apply Euler integration and continuous clamping: $`\mathbf{a}(t+\Delta t) = \mathrm{Softclip}_{[-1, 1]}(\mathbf{a}(t) + \Delta t \frac{\mathrm{d}\mathbf{a}}{\mathrm{d}t})`$.

---
