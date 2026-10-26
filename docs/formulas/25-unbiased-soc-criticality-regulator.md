[Back to theory contents](../THEORY.md)

### Section 25: Unbiased SOC Criticality Regulator
Pareto tail exponent tracking:

```math
\mathcal{S}_{\mathrm{tail}}(t) = \left\{ i \in \bigcup_{l=0}^L V^{(l)} \;\middle|\; \left\| \tilde{\boldsymbol{\varepsilon}}_i(t) \right\|_{\tilde{\mathbf{\Pi}}_i} \ge \varepsilon_{\min} \right\}
```

```math
\overline{L}_{\mathrm{tail}}(t) = \begin{cases} (1 - \gamma_{\mathrm{soc}}) \overline{L}_{\mathrm{tail}}(t - \Delta t) + \gamma_{\mathrm{soc}} \left( \frac{1}{|\mathcal{S}_{\mathrm{tail}}|} \sum_{i \in \mathcal{S}_{\mathrm{tail}}} \ln \left( \frac{\left\| \tilde{\boldsymbol{\varepsilon}}_i(t) \right\|_{\tilde{\mathbf{\Pi}}_i}}{\varepsilon_{\min}} \right) \right), & |\mathcal{S}_{\mathrm{tail}}| \ge N_{\min} \\ \overline{L}_{\mathrm{tail}}(t - \Delta t), & |\mathcal{S}_{\mathrm{tail}}| < N_{\min} \end{cases}
```

```math
\hat{\alpha}_{\mathrm{measured}}(t) = 1 + \frac{1}{\max\left( \delta_{\mathrm{soc}}, \overline{L}_{\mathrm{tail}}(t) \right)}
```

```math
\Delta \alpha = \mathrm{clip}\left( \Delta t \cdot \left(1.2 \cdot \eta_\alpha\right) \left( 1.5 - \hat{\alpha}_{\mathrm{measured}}(t) \right), -\Delta \alpha_{\max}, +\Delta \alpha_{\max} \right)
```

```math
\alpha_{\mathrm{damp}}(t + \Delta t) = \begin{cases} \max\left( \alpha_{\min}, \alpha_{\mathrm{damp}}(t) + \Delta \alpha \right), & |\mathcal{S}_{\mathrm{tail}}| \ge N_{\min} \\ \alpha_{\mathrm{damp}}(t), & |\mathcal{S}_{\mathrm{tail}}| < N_{\min} \end{cases}, \quad \alpha_{\min} = 0.05
```

---
