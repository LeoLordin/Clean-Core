[Back to theory contents](../THEORY.md)

### Section 31: Conformal Connection & Christoffel Symbols
The conformal deformation tensor:

```math
\tilde{g}_{AB}(\boldsymbol{\mu}) = \Omega^2(\boldsymbol{\mu}) (\boldsymbol{\eta}_{\mathcal{L}})_{AB}, \quad \Omega(\boldsymbol{\mu}) = \max\left( \delta_{\mathrm{eps}}, \mathcal{F}_{\mathrm{threshold}} - \mathcal{F}_{\mathrm{pragmatic}}^{\mathrm{Sylvester}}(\boldsymbol{\mu}) \right)
```

Christoffel symbols of the second kind:

```math
\tilde{\Gamma}_{BC}^A = \Gamma_{BC}^A + \delta_B^A \partial_C \ln \Omega(\boldsymbol{\mu}) + \delta_C^A \partial_B \ln \Omega(\boldsymbol{\mu}) - (\boldsymbol{\eta}_{\mathcal{L}})_{BC} (\boldsymbol{\eta}_{\mathcal{L}})^{AK} \partial_K \ln \Omega(\boldsymbol{\mu})
```

Conformal fictitious force:

```math
\tilde{\mathbf{v}}_{\mathrm{conformal}, i}^{(l)}(t) = -\|\mathbf{w}_i^{(l)}(t)\|_2^2 \cdot E\left(\boldsymbol{\mu}_i^{(l)}\right)^\top \boldsymbol{\eta}_{\mathcal{L}} \nabla_{\mathcal{H}^d} \ln \Omega\left(\boldsymbol{\mu}_i^{(l)}\right) + 2 \left( \left(\mathbf{w}_i^{(l)}(t)\right)^\top E\left(\boldsymbol{\mu}_i^{(l)}\right)^\top \boldsymbol{\eta}_{\mathcal{L}} \nabla_{\mathcal{H}^d} \ln \Omega\left(\boldsymbol{\mu}_i^{(l)}\right) \right) \mathbf{w}_i^{(l)}(t)
```

---
