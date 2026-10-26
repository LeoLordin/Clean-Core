[Back to theory contents](../THEORY.md)

### Section 21: SoftVQ Semantic Codebook Evolution & Pruning
Dynamic usage tracking and thresholded pruning:

```math
u_k(t+\Delta t) = \gamma_{\mathrm{usage}} \cdot u_k(t) + \sum_{l=0}^L \sum_{j \in V^{(l)}} \mathbb{I}\left( k = \arg\min_{m: \, m_{\mathrm{pruned}, m} = 1} d_{\mathcal{H}^d}\left(\boldsymbol{\mu}_j^{(l)}(t), \mathrm{Exp}_{\mathbf{o}}^{\mathcal{H}^d}(E(\mathbf{o})\mathbf{c}_m)\right) \right)
```

```math
m_{\mathrm{pruned}, k} = \begin{cases} 1, & k \in \mathcal{K}_{\mathrm{protected}} \lor k = 0 \\ 0, & u_k(t+\Delta t) < \theta_{\mathrm{prune}} = 0.05 \land k \notin \mathcal{K}_{\mathrm{protected}} \\ 1, & u_k(t+\Delta t) \ge 0.05 \end{cases}
```

Spontaneous concept spawning under epistemic prediction divergence:

```math
\text{If } \left( \left( \tilde{\boldsymbol{\varepsilon}}_i^{(l)}(t) \right)^\top \tilde{\mathbf{\Pi}}_i^{(l)}(t) \left( \tilde{\boldsymbol{\varepsilon}}_i^{(l)}(t) \right) > \theta_{\mathrm{spawn}} \right) \land \left( |\mathcal{C}_{\mathrm{active}}| < K_{\max} \right) \land \left( \min_{m} d_{\mathcal{H}^d}\left(\boldsymbol{\mu}_i^{(l)}, \mathrm{Exp}_{\mathbf{o}}^{\mathcal{H}^d}(E(\mathbf{o})\mathbf{c}_m)\right) > d_{\min} \right):
```

```math
\mathcal{C} \leftarrow \mathcal{C} \cup \left\{ E(\mathbf{o})^\top \boldsymbol{\eta}_{\mathcal{L}} \mathrm{Log}_{\mathbf{o}}^{\mathcal{H}^d}\left( \boldsymbol{\mu}_i^{(l)}(t) \right) + \boldsymbol{\eta}_{\mathrm{noise}} \right\}, \quad u_{\mathrm{new}} \leftarrow 1.0, \quad m_{\mathrm{pruned}, \mathrm{new}} \leftarrow 1
```

---
