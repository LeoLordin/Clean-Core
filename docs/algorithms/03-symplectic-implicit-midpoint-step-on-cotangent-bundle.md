[Back to theory contents](../THEORY.md)

### Algorithm 3: Symplectic Implicit Midpoint Step on Cotangent Bundle $`T^{*}\mathcal{H}^{15}`$
* **Inputs:** Phase state $`(\boldsymbol{\mu}_{i, t}^{(l)}, \mathbf{p}_{i, t}^{(l)})`$, free energy gradient $`\nabla_{\mathcal{H}^{d-1}} \mathcal{F}_{\mathrm{pragmatic}}`$.
* **Outputs:** Updated symplectic phase coordinates $`(\boldsymbol{\mu}_{i, t+\Delta t}^{(l)}, \mathbf{p}_{i, t+\Delta t}^{(l)})`$.
* **Complexity:** $`O(d)`$ elementary operations per neuron.
* **Execution Steps:**
  1. Compute midpoint coordinates and momenta:

```math
\boldsymbol{\mu}_{i, t+\frac{\Delta t}{2}}^{(l)} = \frac{\boldsymbol{\mu}_{i, t}^{(l)} + \boldsymbol{\mu}_{i, t+\Delta t}^{(l)}}{2}, \quad \mathbf{p}_{i, t+\frac{\Delta t}{2}}^{(l)} = \frac{\mathbf{p}_{i, t}^{(l)} + \mathbf{p}_{i, t+\Delta t}^{(l)}}{2}
```

  2. Compute physical force at midpoint: $`\mathbf{g}_{t+1/2} = \nabla_{\mathcal{H}^{d-1}} \mathcal{F}_{\mathrm{pragmatic}}(\boldsymbol{\mu}_{i, t+\frac{\Delta t}{2}}^{(l)})`$.
  3. Implicit momentum kick:

```math
\mathbf{p}_{i, t+\Delta t}^{(l)} = \mathbf{p}_{i, t}^{(l)} - \Delta t \left( \mathbf{g}_{t+1/2} + \lambda_{\mathrm{mid}} \boldsymbol{\eta}_{\mathcal{L}} \boldsymbol{\mu}_{i, t+\frac{\Delta t}{2}}^{(l)} \right)
```

  4. Implicit coordinate step:

```math
\boldsymbol{\mu}_{i, t+\Delta t}^{(l)} = \boldsymbol{\mu}_{i, t}^{(l)} + \Delta t \boldsymbol{\eta}_{\mathcal{L}}^{-1} \mathbf{p}_{i, t+\frac{\Delta t}{2}}^{(l)}
```

  5. Solve scalar constraint equation for $`\lambda_{\mathrm{mid}}`$ to satisfy $`\langle \boldsymbol{\mu}_{i, t+\Delta t}^{(l)}, \boldsymbol{\mu}_{i, t+\Delta t}^{(l)} \rangle_{\mathcal{L}} = -1.0`$.
  6. Execute momentum tangent projection: $`\mu_{\mathrm{mid}} = \langle \boldsymbol{\mu}_{i, t+\Delta t}^{(l)}, \boldsymbol{\eta}_{\mathcal{L}}^{-1} \mathbf{p}_{i, t+\Delta t}^{(l)} \rangle_{\mathcal{L}}`$, then update $`\mathbf{p}_{i, t+\Delta t}^{(l)} \leftarrow \mathbf{p}_{i, t+\Delta t}^{(l)} - \mu_{\mathrm{mid}} \boldsymbol{\eta}_{\mathcal{L}} \boldsymbol{\mu}_{i, t+\Delta t}^{(l)}`$.
  7. Verify canonical symplectic 2-form invariance $`\omega_{t+\Delta t} = \omega_t`$.

---
