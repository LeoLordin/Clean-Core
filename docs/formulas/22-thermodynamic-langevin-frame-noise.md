[Back to theory contents](../THEORY.md)

### Section 22: Thermodynamic Langevin Frame Noise

```math
\mathbf{T}_i\left( \tilde{\boldsymbol{\varepsilon}}_i^{(l)} \right) = T_{\max} \cdot \tanh\left( \frac{\alpha_{\mathrm{temp}}}{T_{\max}} \max\left(0.0, \left( \tilde{\boldsymbol{\varepsilon}}_i^{(l)} \right)^\top \tilde{\mathbf{\Pi}}_i^{(l)} \left( \tilde{\boldsymbol{\varepsilon}}_i^{(l)} \right) \right) \right), \quad T_{\max} = 1.0
```

```math
\boldsymbol{\zeta}_{i, t}^{\mathrm{tangent}} = E\left(\boldsymbol{\mu}_i^{(l)}(t)\right) \left[ \sigma_{\mathrm{discrete}, i}^{(l)}(t) \boldsymbol{\xi}_{i, t} \right], \quad \boldsymbol{\xi}_{i, t} \sim \mathcal{N}\left(\mathbf{0}, \mathbf{I}_{d-1}\right)
```

---
