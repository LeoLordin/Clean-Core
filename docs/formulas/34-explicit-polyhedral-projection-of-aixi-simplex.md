[Back to theory contents](../THEORY.md)

### Section 34: Explicit Polyhedral Projection of AIXI Simplex
Kraft lower and upper decision approximations:

```math
Q_{\min}(a) = \sum_{p \in \mathrm{Halted}} \frac{2^{-l(p)}}{M_k} R(p, a), \quad Q_{\max}(a) = Q_{\min}(a) + \Omega_{\mathrm{unresolved}}(k) \frac{r_{\max}}{1 - \gamma}
```

Analytical identity to the universal AIXI optimum:

```math
Q_{\min}(a^{*}) > \max_{a \neq a^{*}} Q_{\max}(a) \implies a_{\mathrm{computable}}^{*} \equiv a_{\mathrm{AIXI}}^{*}
```

Polyhedral pressure force on motor degrees of freedom:

```math
\mathbf{m}_{\mathrm{poly}}(a^{*}) = \varepsilon_{\mathrm{safe}} \cdot \frac{\sum_{a \neq a^{*}} \frac{\mathbf{e}_{a^{*}} - \mathbf{e}_a}{Q_{\min}(a^{*}) - Q_{\max}(a) + \delta_{\mathrm{eps}}}}{\left\| \sum_{a \neq a^{*}} \frac{\mathbf{e}_{a^{*}} - \mathbf{e}_a}{Q_{\min}(a^{*}) - Q_{\max}(a) + \delta_{\mathrm{eps}}} \right\|_2}
```

---
