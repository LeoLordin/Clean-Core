[Back to theory contents](../THEORY.md)

### Section 4: Linearized Cole-Hopf Active Inference (Log-Diffusion HJB)
Nonlinear Hamilton-Jacobi-Bellman stochastic dynamic programming equation:

```math
\frac{\partial V(\boldsymbol{\mu})}{\partial t} + \min_{\mathbf{w}} \left[ \langle \nabla_{\mathcal{H}^{d-1}} V(\boldsymbol{\mu}), \mathbf{w} \rangle_{\mathcal{L}} + \frac{1}{2 \lambda} \|\mathbf{w}\|_2^2 \right] + \mathcal{F}_{\mathrm{pragmatic}}(\boldsymbol{\mu}) + \frac{\lambda \sigma^2}{2} \Delta_{\mathcal{H}^{d-1}} V(\boldsymbol{\mu}) = 0
```

Logarithmic Cole-Hopf transformation:

```math
\psi(\boldsymbol{\mu}, t) = \exp\left( -\frac{V(\boldsymbol{\mu}, t)}{\lambda \sigma^2} \right) \Longleftrightarrow V(\boldsymbol{\mu}, t) = -\lambda \sigma^2 \ln \psi(\boldsymbol{\mu}, t)
```

Linear parabolic Schrödinger-Beltrami value diffusion:

```math
\frac{\partial \psi(\boldsymbol{\mu}, t)}{\partial t} = \frac{\lambda \sigma^2}{2} \Delta_{\mathcal{H}^{d-1}} \psi(\boldsymbol{\mu}, t) - \frac{\mathcal{F}_{\mathrm{pragmatic}}(\boldsymbol{\mu})}{\lambda \sigma^2} \psi(\boldsymbol{\mu}, t)
```

where the radial-spherical Laplace-Beltrami operator on $`\mathcal{H}^{d-1}`$ evaluates to:

```math
\Delta_{\mathcal{H}^{d-1}} \psi = \frac{1}{\sinh^{d-1}(r)} \frac{\partial}{\partial r} \left( \sinh^{d-1}(r) \frac{\partial \psi}{\partial r} \right) + \frac{1}{\sinh^2(r)} \Delta_{\mathbb{S}^{d-2}} \psi
```

Linear goal superposition and analytical geodesic velocity optimum:

```math
\psi_{\mathrm{total}}(\boldsymbol{\mu}) = \sum_{k=1}^K w_k \psi_k(\boldsymbol{\mu})
```

```math
\mathbf{w}_{\mathrm{optimal}}(\boldsymbol{\mu}) = \lambda \sigma^2 \cdot E(\boldsymbol{\mu})^\top \boldsymbol{\eta}_{\mathcal{L}} \frac{\nabla_{\mathcal{H}^{d-1}} \psi_{\mathrm{total}}(\boldsymbol{\mu})}{\psi_{\mathrm{total}}(\boldsymbol{\mu})} = \lambda \sigma^2 \cdot E(\boldsymbol{\mu})^\top \boldsymbol{\eta}_{\mathcal{L}} \nabla_{\mathcal{H}^{d-1}} \ln \psi_{\mathrm{total}}(\boldsymbol{\mu})
```

---
