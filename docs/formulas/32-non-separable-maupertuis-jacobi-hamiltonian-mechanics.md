[Back to theory contents](../THEORY.md)

### Section 32: Non-Separable Maupertuis-Jacobi Hamiltonian Mechanics
Cotangent bundle Hamiltonian $`T^{*}\mathcal{H}^{d-1}`$:

```math
H_{\mathrm{neuron}}(\boldsymbol{\mu}, \mathbf{p}) = \frac{1}{2 \Omega^2(\boldsymbol{\mu})} \langle \mathbf{p}, \mathbf{p} \rangle_{\mathcal{L}} + \mathcal{F}_{\mathrm{pragmatic}}^{\mathrm{Sylvester}}(\boldsymbol{\mu}) + \frac{\kappa_v}{2 \Omega^2(\boldsymbol{\mu})} \|\mathbf{p}\|_{\mathcal{L}}^2
```

Canonical equations of motion:

```math
\frac{\mathrm{d}\boldsymbol{\mu}}{\mathrm{d}t} = \frac{\boldsymbol{\eta}_{\mathcal{L}}^{-1} \mathbf{p}}{\Omega^2(\boldsymbol{\mu})}, \qquad \frac{\mathrm{d}\mathbf{p}}{\mathrm{d}t} = -\nabla_{\mathcal{H}^{d-1}} \mathcal{F}_{\mathrm{pragmatic}}^{\mathrm{Sylvester}}(\boldsymbol{\mu}) + \frac{\nabla_{\mathcal{H}^{d-1}} \Omega(\boldsymbol{\mu})}{\Omega^3(\boldsymbol{\mu})} \langle \mathbf{p}, \mathbf{p} \rangle_{\mathcal{L}} + \lambda_{\mathrm{mid}} \boldsymbol{\eta}_{\mathcal{L}} \boldsymbol{\mu}
```

Symplectic midpoint step ($`\boldsymbol{\mu}_{1/2} = \frac{\boldsymbol{\mu}_t + \boldsymbol{\mu}_{t+\Delta t}}{2}`$, $`\mathbf{p}_{1/2} = \frac{\mathbf{p}_t + \mathbf{p}_{t+\Delta t}}{2}`$):

```math
\frac{\boldsymbol{\mu}_{t+\Delta t} - \boldsymbol{\mu}_t}{\Delta t} = \frac{\boldsymbol{\eta}_{\mathcal{L}}^{-1} \mathbf{p}_{1/2}}{\Omega^2(\boldsymbol{\mu}_{1/2})}
```

```math
\frac{\mathbf{p}_{t+\Delta t} - \mathbf{p}_t}{\Delta t} = -\nabla_{\mathcal{H}^{d-1}} \mathcal{F}(\boldsymbol{\mu}_{1/2}) + \frac{\nabla \Omega(\boldsymbol{\mu}_{1/2})}{\Omega^3(\boldsymbol{\mu}_{1/2})} \langle \mathbf{p}_{1/2}, \mathbf{p}_{1/2} \rangle_{\mathcal{L}} + \lambda_{\mathrm{mid}} \boldsymbol{\eta}_{\mathcal{L}} \boldsymbol{\mu}_{1/2}
```

---
