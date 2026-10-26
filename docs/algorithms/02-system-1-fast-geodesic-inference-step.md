[Back to theory contents](../THEORY.md)

### Algorithm 2: System-1 Fast Geodesic Inference Step ($`\epsilon_0 \sim 1\text{ ms}`$)
* **Inputs:** Current state $`\boldsymbol{\mu}_i^{(l)}(t)`$, context $`\tilde{\mathbf{a}}_i^{(l)}(t)`$, layer barycenter $`\overline{\boldsymbol{\mu}}^{(l)}(t)`$.
* **Outputs:** Updated coordinates $`\boldsymbol{\mu}_i^{(l)}(t+\Delta t)`$, velocity $`\mathbf{w}_i^{(l)}(t)`$, variational precision $`\pi_i^{(l)}(t+\Delta t)`$.
* **Complexity:** $`O(|V| \cdot (d-1) \cdot (h_{\mathrm{dim}} + r^2))`$.
* **Execution Steps:**
  1. Stream layer barycenter $`\overline{\boldsymbol{\mu}}^{(l)}`$ via Fréchet exponential integration.
  2. Construct analytical Lorentz boost tetrad frame $`E(\boldsymbol{\mu}_i^{(l)})`$.
  3. SoftVQ context quantization: evaluate distances $`d_{\mathcal{H}^d}^2`$ to centroids $`\mathbf{c}_k`$, normalize softmax weights, assemble $`\tilde{\mathbf{a}}_i^{(l)}`$.
  4. Forward pass through two-layer perceptron: hidden state $`\mathbf{h} = \sigma(\mathbf{W}^{(1)}\tilde{\mathbf{a}} + \mathbf{b}^{(1)})`$, output $`\mathbf{u}_2 = \mathbf{W}^{(2)}\mathbf{h} + \mathbf{b}^{(2)}`$.
  5. Enforce Euclidean retraction: $`\mathbf{u}_{2, \mathrm{safe}} = \mathbf{u}_2 \cdot \min(1.0, 3.0 / \|\mathbf{u}_2\|_2)`$.
  6. Compute target prediction: $`\boldsymbol{\mu}_{\mathrm{pred}} = \mathrm{Exp}_{\overline{\boldsymbol{\mu}}}(E(\overline{\boldsymbol{\mu}})\mathbf{u}_{2, \mathrm{safe}})`$.
  7. Compute frame-projected prediction error: $`\boldsymbol{\varepsilon}_i = \mathrm{Log}_{\boldsymbol{\mu}_i}(\boldsymbol{\mu}_{\mathrm{pred}})`$, $`\tilde{\boldsymbol{\varepsilon}}_i = E(\boldsymbol{\mu}_i)^\top \boldsymbol{\eta}_{\mathcal{L}} \boldsymbol{\varepsilon}_i`$.
  8. Oja-QR subspace update: project $`\mathbf{y} = \tilde{\mathbf{V}}^\top \tilde{\boldsymbol{\varepsilon}}`$, update singular values $`\mathbf{s}`$, QR factorize $`\tilde{\mathbf{V}}`$, compute residual $`\tilde{\boldsymbol{\varepsilon}}_{\mathrm{residual}}`$, smooth $`\tilde{\mathbf{d}}`$.
  9. Woodbury inversion: compute precision matrix $`\tilde{\mathbf{\Pi}}_i`$.
  10. Calculate Maupertuis conformal drift $`\tilde{\mathbf{v}}_{\mathrm{conformal}, i}^{(l)}`$ and momentum term $`E(\boldsymbol{\mu}_i)^\top \boldsymbol{\eta}_{\mathcal{L}} \frac{\nabla \Omega}{\Omega^3} \langle \mathbf{p}_i, \mathbf{p}_i \rangle_{\mathcal{L}}`$.
  11. Assemble total forces $`\tilde{\mathbf{v}}_{\mathrm{forces}}`$, apply semi-implicit $`A`$-stable solver, add Langevin noise $`\sigma_{\mathrm{discrete}} \boldsymbol{\xi}`$.
  12. Clamp Euclidean velocity norm to $`v_{\max} = 5.0`$.
  13. Execute exponential step $`\mathrm{Exp}_{\boldsymbol{\mu}_i}(E(\boldsymbol{\mu}_i)\mathbf{w}_i)`$ and project coordinates onto hyperboloid.
  14. Update adaptive scalar precision $`\pi_i(t+\Delta t)`$.

---
