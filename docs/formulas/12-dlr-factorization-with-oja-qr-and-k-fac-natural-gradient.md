[Back to theory contents](../THEORY.md)

### Section 12: DLR Factorization with Oja-QR & K-FAC Natural Gradient
Streaming Oja subspace tracking:

```math
\mathbf{y}_i^{(l)}(t) = \left( \tilde{\mathbf{V}}_i^{(l)}(t) \right)^\top \tilde{\boldsymbol{\varepsilon}}_i^{(l)}(t) \in \mathbb{R}^{r}
```

```math
\mathbf{s}_i^{(l)}(t + \Delta t) = (1 - \eta_m) \mathbf{s}_i^{(l)}(t) + \eta_m \left( \mathbf{y}_i^{(l)}(t) \odot \mathbf{y}_i^{(l)}(t) + \delta_{\min} \mathbf{1} \right) \in \mathbb{R}^r
```

```math
\tilde{\mathbf{V}}_{\mathrm{raw}, i}^{(l)} = \tilde{\mathbf{V}}_i^{(l)}(t) + \eta_m \left[ \tilde{\boldsymbol{\varepsilon}}_i^{(l)}(t) \left( \mathbf{y}_i^{(l)}(t) \right)^\top - \tilde{\mathbf{V}}_i^{(l)}(t) \mathbf{y}_i^{(l)}(t) \left( \mathbf{y}_i^{(l)}(t) \right)^\top \right]
```

```math
\tilde{\mathbf{V}}_i^{(l)}(t + \Delta t) = \mathrm{QR\_Q}\left( \tilde{\mathbf{V}}_{\mathrm{raw}, i}^{(l)} \right) \in \mathbb{R}^{(d-1) \times r}
```

Residual variance tracking:

```math
\tilde{\boldsymbol{\varepsilon}}_{\mathrm{residual}, i}^{(l)}(t) = \tilde{\boldsymbol{\varepsilon}}_i^{(l)}(t) - \tilde{\mathbf{V}}_i^{(l)}(t + \Delta t) \left( \left( \tilde{\mathbf{V}}_i^{(l)}(t + \Delta t) \right)^\top \tilde{\boldsymbol{\varepsilon}}_i^{(l)}(t) \right) \in \mathbb{R}^{d-1}
```

```math
\tilde{\mathbf{d}}_i^{(l)}(t + \Delta t) = (1 - \eta_m) \tilde{\mathbf{d}}_i^{(l)}(t) + \eta_m \left( \tilde{\boldsymbol{\varepsilon}}_{\mathrm{residual}, i}^{(l)}(t) \odot \tilde{\boldsymbol{\varepsilon}}_{\mathrm{residual}, i}^{(l)}(t) + \delta_{\min} \mathbf{1} \right)
```

Sensitivity vectors under isometric Lorentz parallel transport:

```math
\mathbf{g}_i^{(2, \mathrm{local})} = +\pi_i^{(l)}(t) \cdot \tilde{\mathbf{\Pi}}_i^{(l)}(t) \tilde{\boldsymbol{\varepsilon}}_i^{(l)}(t) \in \mathbb{R}^{d-1}
```

```math
\mathbf{g}_i^{(2, l)} = E\left(\overline{\boldsymbol{\mu}}^{(l)}(t)\right)^\top \boldsymbol{\eta}_{\mathcal{L}} \mathbf{P}_{\boldsymbol{\mu}_i^{(l)}(t) \to \overline{\boldsymbol{\mu}}^{(l)}(t)}\left( E\left(\boldsymbol{\mu}_i^{(l)}(t)\right) \mathbf{g}_i^{(2, \mathrm{local})} \right) \in \mathbb{R}^{d-1}
```

```math
\mathbf{g}_i^{(1, l)} = \left( \left(\mathbf{W}_i^{(2, l)}\right)^\top \mathbf{g}_i^{(2, l)} \right) \odot \sigma'\left( \mathbf{W}_i^{(1, l)} \tilde{\mathbf{a}}_i^{(l)} + \mathbf{b}_i^{(1, l)} \right) \in \mathbb{R}^{h_{\mathrm{dim}}}
```

Kronecker covariance factors with identity initialization $`\tilde{\mathbf{A}}(0) = \mathbf{I}`$, $`\tilde{\mathbf{B}}(0) = \mathbf{I}`$:

```math
\tilde{\mathbf{A}}_i^{(1, l)}(t + \Delta t) = (1 - \eta_k) \tilde{\mathbf{A}}_i^{(1, l)}(t) + \eta_k \left( \frac{\tilde{\mathbf{a}}_i^{(l)} \left( \tilde{\mathbf{a}}_i^{(l)} \right)^\top}{1.0 + \|\tilde{\mathbf{a}}_i^{(l)}\|^2} \right)
```

```math
\tilde{\mathbf{B}}_i^{(1, l)}(t + \Delta t) = (1 - \eta_k) \tilde{\mathbf{B}}_i^{(1, l)}(t) + \eta_k \left( \frac{\mathbf{g}_i^{(1, l)} \left( \mathbf{g}_i^{(1, l)} \right)^\top}{1.0 + \|\mathbf{g}_i^{(1, l)}\|^2} \right)
```

```math
\tilde{\mathbf{A}}_i^{(2, l)}(t + \Delta t) = (1 - \eta_k) \tilde{\mathbf{A}}_i^{(2, l)}(t) + \eta_k \left( \frac{\mathbf{h}_i^{(l)} \left( \mathbf{h}_i^{(l)} \right)^\top}{1.0 + \|\mathbf{h}_i^{(l)}\|^2} \right)
```

```math
\tilde{\mathbf{B}}_i^{(2, l)}(t + \Delta t) = (1 - \eta_k) \tilde{\mathbf{B}}_i^{(2, l)}(t) + \eta_k \left( \frac{\mathbf{g}_i^{(2, l)} \left( \mathbf{g}_i^{(2, l)} \right)^\top}{1.0 + \|\mathbf{g}_i^{(2, l)}\|^2} \right)
```

---
