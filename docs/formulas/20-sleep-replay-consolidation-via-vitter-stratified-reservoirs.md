[Back to theory contents](../THEORY.md)

### Section 20: Sleep Replay Consolidation via Vitter Stratified Reservoirs

```math
\mathcal{M}_{\mathrm{fast}} = \bigcup_{k=1}^K \mathcal{M}_k, \quad |\mathcal{M}_k| \le \left\lfloor \frac{M_{\mathrm{capacity}}}{K} \right\rfloor
```

```math
\mathbb{E}_{\mathbf{m} \sim \mathcal{M}_{\mathrm{fast}}} [\nabla \mathcal{L}_{\mathrm{recon}}] = \frac{1}{K} \sum_{k=1}^K \mathbb{E}_{\mathbf{m} \sim \mathcal{M}_k} [\nabla \mathcal{L}_{\mathrm{recon}}(\mathbf{m})]
```

```math
\left( \frac{\partial \mathbf{W}_i^{(1, l)}}{\partial t} \right)_{\mathrm{replay}} = -\eta_{\mathrm{replay}} \cdot \frac{1}{K} \sum_{k=1}^K \mathbb{E}_{\mathbf{m} \sim \mathcal{M}_k}\left[ \left( \tilde{\mathbf{B}}_i^{(1, l)} + \delta_{\mathrm{kfac}}\mathbf{I} \right)^{-1} \nabla_{\mathbf{W}_i^{(1, l)}} \mathcal{L}_{\mathrm{recon}}\big( \mathbf{m} \big) \left( \tilde{\mathbf{A}}_i^{(1, l)} + \delta_{\mathrm{kfac}}\mathbf{I} \right)^{-1} \right]
```

```math
\left( \frac{\partial \mathbf{W}_i^{(2, l)}}{\partial t} \right)_{\mathrm{replay}} = -\eta_{\mathrm{replay}} \cdot \frac{1}{K} \sum_{k=1}^K \mathbb{E}_{\mathbf{m} \sim \mathcal{M}_k}\left[ \left( \tilde{\mathbf{B}}_i^{(2, l)} + \delta_{\mathrm{kfac}}\mathbf{I} \right)^{-1} \nabla_{\mathbf{W}_i^{(2, l)}} \mathcal{L}_{\mathrm{recon}}\big( \mathbf{m} \big) \left( \tilde{\mathbf{A}}_i^{(2, l)} + \delta_{\mathrm{kfac}}\mathbf{I} \right)^{-1} \right]
```

---
