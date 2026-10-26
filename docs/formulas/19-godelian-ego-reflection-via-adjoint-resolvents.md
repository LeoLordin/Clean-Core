[Back to theory contents](../THEORY.md)

### Section 19: Gödelian Ego-Reflection via Adjoint Resolvents
Historical state macro-representation:

```math
\text{feat}_{\mathrm{hist}}(t) = \left[ \mathrm{Concat}_{l=1, i}^L\left( \mathrm{sg}\big[\boldsymbol{\mu}_i^{(l)}(t-\Delta t)\big], \pi_i^{(l)}(t-\Delta t) \right) \parallel \mathbf{a}(t-\Delta t) \right] \in \mathbb{R}^{D_{\mathrm{hist}}}
```

Ego-resolvent fixed point:

```math
v_{\mathrm{self}}^{*}(t) = \left( (1 + \delta_{\mathrm{reg}})\mathbf{I}_{d-1} - \mathbf{W}_{\mathrm{self}, v}(t) \right)^{-1} \cdot \left( \mathbf{W}_{\mathrm{self}, \mathrm{hist}}(t) \cdot \text{feat}_{\mathrm{hist}}(t) + \mathbf{b}_{\mathrm{self}}(t) \right) \in \mathbb{R}^{d-1}
```

```math
\mathbf{v}_{\mathrm{ambient}}^{*}(t) = E\left( \overline{\boldsymbol{\mu}}^{(L)}(t) \right) \cdot v_{\mathrm{self}}^{*}(t) \in T_{\overline{\boldsymbol{\mu}}^{(L)}}\mathcal{H}^{d-1}
```

```math
\boldsymbol{\mu}_{\mathrm{self}}^{*}(t) = \mathrm{Exp}_{\overline{\boldsymbol{\mu}}^{(L)}(t)}^{\mathcal{H}^{d-1}}\left( \mathbf{v}_{\mathrm{ambient}}^{*}(t) \right) \in \mathcal{H}^{d-1}
```

```math
\overline{\mathcal{T}}_{\mathrm{introspect}}(t) = \mathrm{Exp}_{\overline{\mathcal{T}}(t-\Delta t)}^{\mathcal{H}^{d-1}}\left( \gamma_{\mathrm{ema}} \mathrm{Log}_{\overline{\mathcal{T}}(t-\Delta t)}^{\mathcal{H}^{d-1}}\Big( \boldsymbol{\mu}_{\mathrm{self}}^{*}(t) \Big) \right)
```

```math
\mathbf{g}_{\mathrm{self}} = E\left(\overline{\mathcal{T}}_{\mathrm{introspect}}\right)^\top \boldsymbol{\eta}_{\mathcal{L}} \mathrm{Log}_{\overline{\mathcal{T}}_{\mathrm{introspect}}(t)}^{\mathcal{H}^{d-1}}\left( \boldsymbol{\mu}_{\mathrm{self}}^{*}(t) \right) \in \mathbb{R}^{d-1}
```

```math
\tilde{\mathbf{g}} = E\left(\overline{\boldsymbol{\mu}}^{(L)}(t)\right)^\top \boldsymbol{\eta}_{\mathcal{L}} \mathbf{P}_{\overline{\mathcal{T}}_{\mathrm{introspect}}(t) \to \overline{\boldsymbol{\mu}}^{(L)}(t)}\left( E\left(\overline{\mathcal{T}}_{\mathrm{introspect}}\right) \mathbf{g}_{\mathrm{self}} \right) \in \mathbb{R}^{d-1}
```

```math
\mathbf{W}_{\mathrm{self}, v}(t + \Delta t) = \mathrm{Clip}_{\sigma_{\max} \le 1 - \epsilon_{\mathrm{gap}}}\left( \mathbf{W}_{\mathrm{self}, v}(t) - \Delta t \eta_{\mathrm{self}} \left( (1 + \delta_{\mathrm{reg}})\mathbf{I}_{d-1} - \mathbf{W}_{\mathrm{self}, v}(t)^\top \right)^{-1} \tilde{\mathbf{g}} \cdot \left( v_{\mathrm{self}}^{*}(t) \right)^\top \right)
```

```math
\mathbf{W}_{\mathrm{self}, \mathrm{hist}}(t + \Delta t) = \mathbf{W}_{\mathrm{self}, \mathrm{hist}}(t) - \Delta t \eta_{\mathrm{self}} \left( (1 + \delta_{\mathrm{reg}})\mathbf{I}_{d-1} - \mathbf{W}_{\mathrm{self}, v}(t)^\top \right)^{-1} \tilde{\mathbf{g}} \cdot \left( \text{feat}_{\mathrm{hist}}(t) \right)^\top
```

```math
\mathbf{b}_{\mathrm{self}}(t + \Delta t) = \mathbf{b}_{\mathrm{self}}(t) - \Delta t \eta_{\mathrm{self}} \left( (1 + \delta_{\mathrm{reg}})\mathbf{I}_{d-1} - \mathbf{W}_{\mathrm{self}, v}(t)^\top \right)^{-1} \tilde{\mathbf{g}}
```

---
