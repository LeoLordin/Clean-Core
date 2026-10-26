[Back to theory contents](../THEORY.md)

### Section 23: Fast Associative Attention & Memristive Crossbar Currents
Key-value manifold projections:

```math
\mathbf{m}_m^{\mathrm{key}} = \mathrm{Exp}_{\mathbf{o}}^{\mathcal{H}^{d-1}}\left( E(\mathbf{o}) \cdot \mathbf{W}_{\mathrm{key}} \mathbf{h}_m \right), \quad \mathbf{m}_m^{\mathrm{value}} = \mathrm{Exp}_{\mathbf{o}}^{\mathcal{H}^{d-1}}\left( E(\mathbf{o}) \cdot \mathbf{W}_{\mathrm{val}} \mathbf{h}_m \right), \quad \mathbf{h}_m \in \mathbf{H}_{\mathrm{memory}}
```

```math
\text{score}_m = -\frac{d_{\mathcal{H}^{d-1}}^2\left( \boldsymbol{\mu}_i^{(l)}, \mathbf{m}_m^{\mathrm{key}} \right)}{T_{\mathrm{attn}} \sqrt{d-1}}, \quad w_m = \frac{\exp\left( \text{score}_m - \max_k(\text{score}_k) \right)}{\sum_{k \in \mathcal{M}_{\mathrm{fast}}} \exp\left( \text{score}_k - \max_k(\text{score}_k) \right)}
```

```math
\tilde{\mathbf{\Delta}}_{\mathrm{fast}, i}^{\mathrm{attn}}(t) = \eta_{\mathrm{dopamine}} \sum_{m \in \mathcal{M}_{\mathrm{fast}}} w_m E\left(\boldsymbol{\mu}_i^{(l)}\right)^\top \boldsymbol{\eta}_{\mathcal{L}} \mathrm{Log}_{\boldsymbol{\mu}_i^{(l)}}\left( \mathbf{m}_m^{\mathrm{value}} \right) \in \mathbb{R}^{d-1}
```

```math
\tilde{\mathbf{\Delta}}_{\mathrm{fast}, i}^{(l)}(t) = \tilde{\mathbf{\Delta}}_{\mathrm{fast}, i}^{\mathrm{attn}}(t) + E\left(\boldsymbol{\mu}_i^{(l)}\right)^\top \boldsymbol{\eta}_{\mathcal{L}} \mathbf{P}_{\mathbf{o} \to \boldsymbol{\mu}_i^{(l)}(t)}\left( E(\mathbf{o}) \cdot \mathbf{W}_{\mathrm{analog\_crossbar}} \cdot \mathbf{s}_i(t) \right) \in \mathbb{R}^{d-1}
```

---
