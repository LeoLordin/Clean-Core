[Back to theory contents](../THEORY.md)

### Section 10: Retraction Perceptron & Tangent-Space SoftVQ
Parent context assembly:

```math
\tilde{\mathbf{a}}_i^{(l)}(t) = \begin{cases} \sum_{j \in \mathrm{parents}(i)} E_{ji}^{(l+1)} \mathrm{SoftVQ}_{\mathrm{tangent}}\left( \boldsymbol{\mu}_j^{(l+1)}(t) \right), & 0 \le l < L \\ E\left(\overline{\boldsymbol{\mu}}^{(L)}(t)\right)^\top \boldsymbol{\eta}_{\mathcal{L}} \mathrm{Log}_{\overline{\boldsymbol{\mu}}^{(L)}(t)}^{\mathcal{H}^{d-1}}\left( \overline{\mathcal{T}}_{\mathrm{introspect}}(t) \right), & l = L \end{cases} \in \mathbb{R}^{d-1}
```

Two-layer prediction perceptron with tangent Euclidean retraction ($`R_{\mathrm{pred\_max}} = 3.0`$):

```math
\mathbf{h}_i^{(l)}(t) = \sigma\left( \mathbf{W}_i^{(1, l)} \tilde{\mathbf{a}}_i^{(l)}(t) + \mathbf{b}_i^{(1, l)} \right) \in \mathbb{R}^{h_{\mathrm{dim}}}
```

```math
\mathbf{u}_{2, i}^{(l)}(t) = \mathbf{W}_i^{(2, l)} \mathbf{h}_i^{(l)}(t) + \mathbf{b}_i^{(2, l)} \in \mathbb{R}^{d-1}
```

```math
\mathbf{u}_{2, \mathrm{safe}, i}^{(l)}(t) = \mathbf{u}_{2, i}^{(l)}(t) \cdot \min\left(1.0, \frac{R_{\mathrm{pred\_max}}}{\|\mathbf{u}_{2, i}^{(l)}(t)\|_2}\right), \quad R_{\mathrm{pred\_max}} = 3.0
```

```math
\mathbf{v}_{\mathrm{pred}, i}^{(l)}(t) = E\left(\overline{\boldsymbol{\mu}}^{(l)}(t)\right) \cdot \mathbf{u}_{2, \mathrm{safe}, i}^{(l)}(t) \in T_{\overline{\boldsymbol{\mu}}^{(l)}}\mathcal{H}^{d-1}, \quad \|\mathbf{v}_{\mathrm{pred}, i}^{(l)}(t)\|_{\mathcal{L}} \equiv \|\mathbf{u}_{2, \mathrm{safe}, i}^{(l)}(t)\|_2 \le 3.0
```

```math
\boldsymbol{\mu}_{\mathrm{pred}, i}^{(l)}(t) = \mathrm{Exp}_{\overline{\boldsymbol{\mu}}^{(l)}(t)}^{\mathcal{H}^{d-1}}\left( \mathbf{v}_{\mathrm{pred}, i}^{(l)}(t) \right) \in \mathcal{H}^{d-1}
```

```math
\boldsymbol{\varepsilon}_i^{(l)}(t) = \mathrm{Log}_{\boldsymbol{\mu}_i^{(l)}(t)}^{\mathcal{H}^{d-1}}\left( \boldsymbol{\mu}_{\mathrm{pred}, i}^{(l)}(t) \right) \in T_{\boldsymbol{\mu}_i^{(l)}}\mathcal{H}^{d-1}
```

SoftVQ tangent quantization with active token preservation:

```math
d_{\mathrm{min\_act}}^2 = \min_{m: \, m_{\mathrm{pruned}, m} = 1} d_{\mathcal{H}^{d-1}}^2\left( \boldsymbol{\mu}_j^{(l+1)}, \mathrm{Exp}_{\mathbf{o}}^{\mathcal{H}^{d-1}}(E(\mathbf{o})\mathbf{c}_m) \right)
```

```math
w_k = \frac{m_{\mathrm{pruned}, k} \cdot \exp\left( \frac{-d_{\mathcal{H}^{d-1}}^2\left( \boldsymbol{\mu}_j^{(l+1)}, \mathrm{Exp}_{\mathbf{o}}(E(\mathbf{o})\mathbf{c}_k) \right) + d_{\mathrm{min\_act}}^2}{T_{\mathrm{vq}}} \right)}{\sum_{k'=0}^{|\mathcal{C}|} m_{\mathrm{pruned}, k'} \exp\left( \frac{-d_{\mathcal{H}^{d-1}}^2\left( \boldsymbol{\mu}_j^{(l+1)}, \mathrm{Exp}_{\mathbf{o}}(E(\mathbf{o})\mathbf{c}_{k'}) \right) + d_{\mathrm{min\_act}}^2}{T_{\mathrm{vq}}} \right) + \delta_{\mathrm{eps}}}
```

```math
\mathrm{SoftVQ}_{\mathrm{tangent}}\left( \boldsymbol{\mu}_j^{(l+1)} \right) = \sum_{k=0}^{|\mathcal{C}|} w_k \mathbf{c}_k \in \mathbb{R}^{d-1}, \quad \mathbf{c}_0 = \mathbf{0}, \quad m_{\mathrm{pruned}, 0} \equiv 1
```

---
