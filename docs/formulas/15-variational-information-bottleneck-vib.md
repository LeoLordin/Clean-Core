[Back to theory contents](../THEORY.md)

### Section 15: Variational Information Bottleneck (VIB)

```math
\mathcal{K}_{\mathrm{VIB}}\big( \mathbf{\Xi}(t), \mathbf{a}(t) \big) = \sum_{l=0}^L \sum_{i \in V^{(l)}} \mathbb{D}_{\mathrm{KL}}\left( Q\left( \mathbf{\Theta}_i^{(l)} \;\middle|\; \mathcal{D}_{\mathrm{data}} \right) \;\middle\|\; P_0\left( \mathbf{\Theta}_i^{(l)} \right) \right) - \beta_{\mathrm{info}}(t) \mathcal{I}_{\mathrm{var}}\left( \boldsymbol{\mu}_i^{(l)} ; \mathbf{x}_{\mathrm{world}} \;\middle|\; \mathbf{a}(t) \right)
```

```math
\mathcal{I}_{\mathrm{var}}\left( \boldsymbol{\mu}_i^{(l)} ; \mathbf{x}_{\mathrm{world}} \;\middle|\; \mathbf{a}(t) \right) \ge \mathbb{E}_{\mathcal{B}}\left[ f_{\psi}^{\mathrm{VIB}}\left(\mathbf{x}_{\mathrm{world}}, \boldsymbol{\mu}_i^{(l)}, \mathbf{a}(t)\right) - \ln \sum_{\mathbf{x}' \in \mathcal{B}} \exp\left( f_{\psi}^{\mathrm{VIB}}\left(\mathbf{x}', \boldsymbol{\mu}_i^{(l)}, \mathbf{a}(t)\right) \right) \right]
```

---
