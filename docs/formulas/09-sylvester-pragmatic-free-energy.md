[Back to theory contents](../THEORY.md)

### Section 9: Sylvester Pragmatic Free Energy

```math
\begin{aligned}
\mathcal{F}_{\mathrm{pragmatic}}^{\mathrm{Sylvester}}\big( \mathbf{\Xi}(t) \big) &= \sum_{l=0}^L \sum_{i \in V^{(l)}} \Bigg[ \frac{\pi_i^{(l)}(t)}{2} \left( \tilde{\boldsymbol{\varepsilon}}_i^{(l)}(t) \right)^\top \tilde{\mathbf{\Pi}}_i^{(l)}(t) \left( \tilde{\boldsymbol{\varepsilon}}_i^{(l)}(t) \right) - \frac{d-1}{2} \ln \pi_i^{(l)}(t) \\
&\quad + \frac{1}{2} \left( \sum_{k=1}^{d-1} \ln \tilde{d}_{i, k}^{(l)} + \sum_{p=1}^r \ln s_{i, p}^{(l)} + \ln \det\left( \mathrm{diag}\left(\mathbf{s}_i^{(l)}\right)^{-1} + \left( \tilde{\mathbf{V}}_i^{(l)} \right)^\top \mathrm{diag}\left(\tilde{\mathbf{d}}_i^{(l)}\right)^{-1} \tilde{\mathbf{V}}_i^{(l)} \right) \right) + \frac{1}{2 M_{\mathrm{trace}}} \sum_{m=1}^{M_{\mathrm{trace}}} \mathbf{u}_m^\top \left( \tilde{\mathbf{\Pi}}_i^{(l)} - \mathbf{I}_{d-1} \right) \mathbf{u}_m \Bigg]
\end{aligned}
```

```math
\mathbf{u}_m \sim \{-1, +1\}^{d-1}
```

---
