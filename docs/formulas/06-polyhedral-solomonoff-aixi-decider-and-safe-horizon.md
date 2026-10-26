[Back to theory contents](../THEORY.md)

### Section 6: Polyhedral Solomonoff-AIXI Decider & Safe Horizon
Decision hyperplanes partitioning the observation distribution simplex $`\Delta^{|\mathcal{O}|-1}`$:

```math
\mathcal{H}_{a, a'} = \left\{ \mathbf{p} \in \Delta^{|\mathcal{O}|-1} \;\middle|\; \sum_{o \in \mathcal{O}} p(o) \big( Q(h, a, o) - Q(h, a', o) \big) = 0 \right\}
```

Kraft inequality tail bound under Levin program interleaving:

```math
\Omega_{\mathrm{unresolved}}(k) = 1.0 - \sum_{p \in \mathrm{Halted}(k)} 2^{-l(p)}
```

```math
Q_{\min}(a) = \sum_{p \in \mathrm{Halted}(k)} \frac{2^{-l(p)}}{M_k} R(p, a), \quad Q_{\max}(a) = Q_{\min}(a) + \Omega_{\mathrm{unresolved}}(k) \cdot \frac{r_{\max}}{1 - \gamma}
```

Strict early-stopping decision boundary:

```math
Q_{\min}(a^{*}) > \max_{a \neq a^{*}} Q_{\max}(a) \implies a_{\mathrm{computable}}^{*} \equiv a_{\mathrm{AIXI}}^{*}
```

Polyhedral safety radius and continuous motor pressure vector:

```math
\varepsilon_{\mathrm{safe}} = \frac{Q_{\min}(a^{*}) - \max_{a \neq a^{*}} Q_{\max}(a)}{\sqrt{\sum_{o \in \mathcal{O}} \left( Q(a^{*}, o) - \max_{a \neq a^{*}} Q(a, o) \right)^2}} > 0
```

```math
\mathbf{m}_{\mathrm{poly}}(a^{*}) = \varepsilon_{\mathrm{safe}} \cdot \frac{\sum_{a \neq a^{*}} \frac{\mathbf{e}_{a^{*}} - \mathbf{e}_a}{Q_{\min}(a^{*}) - Q_{\max}(a) + \delta_{\mathrm{eps}}}}{\left\| \sum_{a \neq a^{*}} \frac{\mathbf{e}_{a^{*}} - \mathbf{e}_a}{Q_{\min}(a^{*}) - Q_{\max}(a) + \delta_{\mathrm{eps}}} \right\|_2} \in \mathbb{R}^{d_{\mathrm{act}}}
```

---
