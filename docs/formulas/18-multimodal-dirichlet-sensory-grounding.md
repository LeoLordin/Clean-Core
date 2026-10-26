[Back to theory contents](../THEORY.md)

### Section 18: Multimodal Dirichlet Sensory Grounding

```math
\mathbf{v}_{\mathrm{bundle}} = E(\mathbf{o}) \cdot \pi_{\mathrm{bundle}}\left( \mathbf{W}_{\mathrm{vis}} \mathrm{Enc}_{\mathrm{vis}}(\mathbf{x}_{\mathrm{img}}) \parallel \mathbf{W}_{\mathrm{txt}} \mathrm{Enc}_{\mathrm{txt}}(\mathbf{x}_{\mathrm{text}}) \parallel \mathbf{W}_{\mathrm{aud}} \mathrm{Enc}_{\mathrm{aud}}(\mathbf{x}_{\mathrm{audio}}) \right) \in T_{\mathbf{o}}\mathcal{H}^{d-1}
```

```math
\boldsymbol{\mu}_0^{(0)}(t + \Delta t) = \mathrm{Exp}_{\mathbf{o}}^{\mathcal{H}^{d-1}}\left( \mathbf{v}_{\mathrm{bundle}} \right)
```

```math
\mathcal{I}_{\mathrm{grounding}}\big( \boldsymbol{\mu}^{(L)} ; \boldsymbol{\mu}_0^{(0)} \big) = \mathbb{E}_{\mathcal{B}}\left[ f_{\mathrm{ground}}\big(\boldsymbol{\mu}^{(L)}, \boldsymbol{\mu}_0^{(0)}\big) - \ln \sum_{\boldsymbol{\mu}' \in \mathcal{B}} \exp\Big( f_{\mathrm{ground}}\big(\boldsymbol{\mu}^{(L)}, \boldsymbol{\mu}'\big) \Big) \right]
```

---
