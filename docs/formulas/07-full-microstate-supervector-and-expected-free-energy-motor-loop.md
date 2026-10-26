[Back to theory contents](../THEORY.md)

### Section 7: Full Microstate Supervector & Expected Free Energy Motor Loop
Full microstate configuration vector:

```math
\mathbf{\Xi}_i^{(l)}(t) = \begin{pmatrix} \boldsymbol{\mu}_i^{(l)}(t) \\ \mathbf{p}_i^{(l)}(t) \\ \mathbf{\Theta}_i^{(l)}(t) \\ \pi_i^{(l)}(t) \\ E_{ji}^{(l+1)}(t) \\ \mathbf{A}_{\mathrm{SCM}}(t) \end{pmatrix} \in \mathcal{M}_i^{(l)}, \quad \mathbf{\Theta}_i^{(l)}(t) = \left\{ \mathbf{W}_i^{(1, l)}, \mathbf{W}_i^{(2, l)}, \mathbf{b}_i^{(1, l)}, \mathbf{b}_i^{(2, l)} \right\}
```

Motor actuation update via Expected Free Energy (EFE) gradient descent:

```math
\frac{\mathrm{d} \mathbf{a}}{\mathrm{d} t} = -\eta_a \nabla_{\mathbf{a}} \left[ \mathbb{D}_{\mathrm{KL}}\left( Q(s_{t+\Delta t} \mid \mathbf{a}(t)) \parallel P^{*}(s_{\mathrm{target}}) \right) - \lambda_{\mathrm{Pearl}}(t) \mathcal{I}_{\mathrm{Pearl}}^{\mathrm{causal}}\big( \mathbf{do}(\mathbf{a}(t)) \big) + \frac{\lambda_{\mathrm{policy}}}{2} \|\mathbf{a}(t) - \mathbf{a}_{\mathrm{target}}(t)\|^2 \right]
```

```math
\mathbf{a}(t + \Delta t) = \mathrm{Softclip}_{[-1, 1]}\left( \mathbf{a}(t) + \Delta t \cdot \frac{\mathrm{d} \mathbf{a}}{\mathrm{d} t} \right) \in [-1, 1]^{d_{\mathrm{act}}}
```

```math
\mathbf{a}_{\mathrm{target}}(t) = \tanh\left( \mathbf{W}_{\mathrm{motor}} \cdot \mathbf{w}_{\mathrm{geodesic}}^{(L)}(t) + \mathbf{W}_{\mathrm{poly}} \cdot \mathbf{m}_{\mathrm{poly}}(a^{*}) + \mathbf{W}_{\mathrm{social}} \cdot \mathbf{a}_{\mathrm{ToM}}^{*}(t) + \mathbf{b}_{\mathrm{motor}} \right)
```

---
