[Back to theory contents](../THEORY.md)

### Algorithm 6: Gödelian Ego-Reflection via Adjoint Resolvents
* **Inputs:** Historical microstate stream, gradient-detached coordinates $`\mathrm{sg}[\boldsymbol{\mu}]`$, motor vector $`\mathbf{a}(t-\Delta t)`$.
* **Outputs:** Updated weight tensors $`\mathbf{W}_{\mathrm{self}, v}, \mathbf{W}_{\mathrm{self}, \mathrm{hist}}`$, introspection trajectory $`\overline{\mathcal{T}}_{\mathrm{introspect}}`$.
* **Complexity:** $`O((d-1)^3 + (d-1)^2 D_{\mathrm{hist}})`$.
* **Execution Steps:**
  1. Assemble historical macro-vector: $`\text{feat}_{\mathrm{hist}}(t) \in \mathbb{R}^{D_{\mathrm{hist}}}`$.
  2. Solve linear resolvent system:  

```math
((1+\delta_{\mathrm{reg}})\mathbf{I} - \mathbf{W}_{\mathrm{self}, v}) \cdot v_{\mathrm{self}}^{*} = \mathbf{W}_{\mathrm{self}, \mathrm{hist}} \text{feat}_{\mathrm{hist}} + \mathbf{b}_{\mathrm{self}}
```

  3. Lift to layer $`L`$ barycenter tangent space: $`\mathbf{v}_{\mathrm{ambient}}^{*} = E(\overline{\boldsymbol{\mu}}^{(L)}) v_{\mathrm{self}}^{*}`$.
  4. Execute ego-identity exponential step: $`\boldsymbol{\mu}_{\mathrm{self}}^{*} = \mathrm{Exp}_{\overline{\boldsymbol{\mu}}^{(L)}}(\mathbf{v}_{\mathrm{ambient}}^{*})`$.
  5. Update introspection trajectory $`\overline{\mathcal{T}}_{\mathrm{introspect}}`$ with geodesic momentum $`\gamma_{\mathrm{ema}}`$.
  6. Compute residual gradient: $`\mathbf{g}_{\mathrm{self}} = E(\overline{\mathcal{T}})^\top \boldsymbol{\eta}_{\mathcal{L}} \mathrm{Log}_{\overline{\mathcal{T}}}(\boldsymbol{\mu}_{\mathrm{self}}^{*})`$.
  7. Parallel transport $`\mathbf{g}_{\mathrm{self}}`$ to layer $`L`$ barycenter to form $`\tilde{\mathbf{g}}`$.
  8. Perform gradient descent scaled by adjoint resolvent $`((1+\delta_{\mathrm{reg}})\mathbf{I} - \mathbf{W}_{\mathrm{self}, v}^\top)^{-1}`$.
  9. Apply spectral norm clamping: $`\sigma_{\max} \le 1 - \epsilon_{\mathrm{gap}}`$.

---
