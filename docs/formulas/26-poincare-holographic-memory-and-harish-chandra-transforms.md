[Back to theory contents](../THEORY.md)

### Section 26: Poincaré Holographic Memory & Harish-Chandra Transforms
Lorentz group spherical basis ($`\mathrm{SO}^+(1, d-1)`$):

```math
\phi_{\lambda, \mathbf{b}}(\boldsymbol{\mu}) = \left( \langle \boldsymbol{\mu}, \boldsymbol{\xi}(\mathbf{b}) \rangle_{\mathcal{L}} \right)^{-\frac{d-2}{2} + i \lambda}, \quad \lambda \in \mathbb{R}^+, \quad \mathbf{b} \in \mathbb{S}^{d-2}, \quad \boldsymbol{\xi}(\mathbf{b}) = (1, \mathbf{b})^\top
```

Direct Harish-Chandra episode encoding:

```math
\tilde{A}_m(\lambda, \mathbf{b}) = \int_{\mathcal{H}^{d-1}} \mathrm{Log}_{\mathbf{o}}^{\mathcal{H}^{d-1}}\left(\boldsymbol{\mu}_m\right) \cdot \phi_{-\lambda, \mathbf{b}}(\boldsymbol{\mu}) \, \mathrm{d}\mathrm{Vol}_{\mathcal{H}^{d-1}}(\boldsymbol{\mu})
```

Interference retrieval via Plancherel measure $`|c(\lambda)|^{-2}`$:

```math
\mathbf{H}_{\mathrm{recall}}\left(\boldsymbol{\mu}_i^{(l)}\right) = \int_{0}^\infty \mathrm{d}\lambda \, |c(\lambda)|^{-2} \int_{\mathbb{S}^{d-2}} \mathrm{d}\mathbf{b} \left[ \sum_{m \in \mathcal{M}_{\mathrm{fast}}} \tilde{A}_m(\lambda, \mathbf{b}) \exp\left( -i \frac{\lambda \cdot d_{\mathcal{H}^{d-1}}\left( \boldsymbol{\mu}_i^{(l)}, \boldsymbol{\mu}_m \right)}{\hbar_{\mathrm{cog}}} \right) \right] \phi_{\lambda, \mathbf{b}}\left(\boldsymbol{\mu}_i^{(l)}\right)
```

```math
|c(\lambda)|^{-2} = \frac{1}{2\pi} \cdot \left| \frac{\Gamma\left( \frac{d-1}{2} \right) \Gamma(i \lambda)}{\Gamma\left( \frac{d-1}{2} + i \lambda \right)} \right|^{-2}
```

---
