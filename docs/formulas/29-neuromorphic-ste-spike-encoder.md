[Back to theory contents](../THEORY.md)

### Section 29: Neuromorphic STE Spike Encoder

```math
\lambda_{\mathrm{spike}, i}^{(l)}(t) = f_{\max} \cdot \sigma\left( \mathbf{W}_{\mathrm{enc}} \cdot E(\mathbf{o})^\top \boldsymbol{\eta}_{\mathcal{L}} \mathrm{Log}_{\mathbf{o}}^{\mathcal{H}^{d-1}}\left( \boldsymbol{\mu}_i^{(l)}(t) \right) + \mathbf{b}_{\mathrm{enc}} \right) \in \mathbb{R}^{M_{\mathrm{crossbar}}}
```

```math
p_{i, m}(t) = 1.0 - \exp\left( -\lambda_{\mathrm{spike}, i, m}^{(l)}(t) \cdot \Delta t \right) \in [0, 1]
```

```math
\mathbf{s}_i(t) = \mathrm{step}\left( p_i(t) - \boldsymbol{\xi}_{\mathrm{uniform}} \right) - \mathrm{sg}\left[ p_i(t) \right] + p_i(t) \in \{0, 1\}^{M_{\mathrm{crossbar}}}, \quad \boldsymbol{\xi}_{\mathrm{uniform}} \sim \mathcal{U}(0, 1)^{M_{\mathrm{crossbar}}}
```

---
