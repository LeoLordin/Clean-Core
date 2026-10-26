[Back to theory contents](../THEORY.md)

### Section 16: Differentiable Causal Do-Calculus & Schur-Capped NOTEARS
Structural causal forward model:

```math
s_{t+\Delta t} = f_\phi\left( \left( (1 + \delta_{\mathrm{scm}})\mathbf{I}_{d_{\mathrm{scm}}} - \mathbf{A}_{\mathrm{SCM}}^\top \right)^{-1} g(s_t, \mathbf{a}(t)) \right) + \boldsymbol{\epsilon}_{\mathrm{scm}}
```

Causal counterfactual mutual information:

```math
\mathcal{I}_{\mathrm{Pearl}}^{\mathrm{causal}}\big( \mathbf{do}(\mathbf{a}(t)) \big) = \mathbb{E}_{\mathcal{B}}\left[ f_\phi\big(s_{t+\Delta t}, s_t, \mathbf{do}(\mathbf{a}(t))\big) - \ln \sum_{s' \in \mathcal{B}} \exp\Big( f_\phi\big(s', s_t, \mathbf{a}_{\mathrm{counterfactual}}(t)\big) \Big) \right]
```

```math
\mathbf{P}_{\mathbf{a}^\perp}(t) = \mathbf{I}_{d_{\mathrm{act}}} - \frac{\mathbf{a}(t) \mathbf{a}(t)^\top}{\|\mathbf{a}(t)\|^2 + \delta_{\mathrm{proj}}^2}, \quad \delta_{\mathrm{proj}} = 10^{-5}
```

```math
\mathbf{a}_{\mathrm{counterfactual}}(t) = \mathrm{Softclip}_{[-1, 1]}\left( \mathbf{a}(t) + \sigma_{\mathrm{cf}} \mathbf{P}_{\mathbf{a}^\perp}(t) \boldsymbol{\zeta} \right), \quad \boldsymbol{\zeta} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{d_{\mathrm{act}}})
```

Log-determinant acyclicity constraint:

```math
h_{\mathrm{log}}(\mathbf{A}_{\mathrm{SCM}}) = \ln \left( \frac{1}{d_{\mathrm{scm}}} \mathrm{Tr}\left( \mathrm{expm}(\mathbf{A}_{\mathrm{SCM}} \circ \mathbf{A}_{\mathrm{SCM}}) \right) \right) = 0
```

```math
\nabla_{\mathbf{A}} h_{\mathrm{log}}(\mathbf{A}) = \frac{2}{\mathrm{Tr}(\mathrm{expm}(\mathbf{A}\circ\mathbf{A}))} \left( \mathrm{expm}(\mathbf{A}\circ\mathbf{A})^\top \circ \mathbf{A} \right)
```

```math
\mathbf{A}_{\mathrm{SCM}}(t + \Delta t) = \mathrm{Clip}_{\rho(\mathbf{A}) \le 1 - \epsilon_{\mathrm{gap}}}^{\mathrm{Schur}}\left( \mathbf{A}_{\mathrm{SCM}}(t) - \frac{\Delta t}{\epsilon_1} \left[ -\nabla_{\mathbf{A}_{\mathrm{SCM}}} \mathcal{I}_{\mathrm{Pearl}}^{\mathrm{causal}} + \mu_{\mathrm{DAG}} \nabla_{\mathbf{A}_{\mathrm{SCM}}} h_{\mathrm{log}}(\mathbf{A}_{\mathrm{SCM}}) \right] \right)
```

where the spectral radius $`\rho(\mathbf{A})`$ is bounded strictly via real Schur decomposition:

```math
\mathbf{A} = \mathbf{Z} \mathbf{T} \mathbf{Z}^\top, \quad \mathbf{Z}^\top \mathbf{Z} = \mathbf{I}, \quad \mathbf{T} \text{ is real quasi-triangular Schur form.}
```

---
