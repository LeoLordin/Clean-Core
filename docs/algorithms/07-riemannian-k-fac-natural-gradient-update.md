[Back to theory contents](../THEORY.md)

### Algorithm 7: Riemannian K-FAC Natural Gradient Update ($`\epsilon_1 \sim 1\text{ s}`$)
* **Inputs:** Error sensitivity vectors $`\tilde{\boldsymbol{\varepsilon}}_i`$, perceptron activation matrices, Augmented Lagrangian gradients.
* **Outputs:** Updated synaptic tensors $`\mathbf{W}_i^{(1, l)}, \mathbf{W}_i^{(2, l)}`$.
* **Complexity:** $`O(|V| \cdot (h_{\mathrm{dim}}^3 + (d-1)^3))`$.
* **Execution Steps:**
  1. Parallel transport local gradient $`\mathbf{g}_i^{(2, \mathrm{local})}`$ to layer barycenter: $`\mathbf{g}_i^{(2, l)} = \mathbf{P}_{\boldsymbol{\mu}_i \to \overline{\boldsymbol{\mu}}}(\mathbf{g}_i^{(2, \mathrm{local})})`$.
  2. Backpropagate sensitivity: $`\mathbf{g}_i^{(1, l)} = ((\mathbf{W}_i^{(2)})^\top \mathbf{g}_i^{(2)}) \odot \sigma'`$.
  3. Update running Kronecker factors $`\tilde{\mathbf{A}}^{(1)}, \tilde{\mathbf{B}}^{(1)}, \tilde{\mathbf{A}}^{(2)}, \tilde{\mathbf{B}}^{(2)}`$ with momentum $`\eta_k`$.
  4. Adiabatic decimation check: step executes strictly every $`K_{\mathrm{infer}} = 4`$ inference ticks.
  5. Invert Tikhonov-damped factors: $`(\tilde{\mathbf{A}} + \delta_{\mathrm{kfac}}\mathbf{I})^{-1}`$ and $`(\tilde{\mathbf{B}} + \delta_{\mathrm{kfac}}\mathbf{I})^{-1}`$.
  6. Synthesize natural gradient matrix tensor: $`\Delta \mathbf{W} = \tilde{\mathbf{B}}^{-1} \nabla \mathbf{W} \tilde{\mathbf{A}}^{-1}`$.
  7. Clamp Frobenius norm: $`\|\Delta \mathbf{W}\|_F \le M_{\mathrm{step}} = 1.0`$.
  8. Update synaptic weights: $`\mathbf{W} \leftarrow \mathbf{W} - \frac{\Delta t}{\epsilon_1} \Delta \mathbf{W}_{\mathrm{clipped}}`$.

---
