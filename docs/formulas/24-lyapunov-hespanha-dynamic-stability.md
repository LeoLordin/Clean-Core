[Back to theory contents](../THEORY.md)

### Section 24: Lyapunov-Hespanha Dynamic Stability
Unified Lyapunov energy functional:

```math
V(\mathbf{\Xi}(t)) = \frac{1}{2} \sum_{l=1}^L \sum_{i \in V^{(l)}} d_{\mathcal{H}^d}^2\left( \boldsymbol{\mu}_i^{(l)}(t), \overline{\boldsymbol{\mu}}^{(l)}(t) \right) + \frac{1}{2} \|\mathbf{a}(t) - \mathbf{a}_{\mathrm{target}}(t)\|^2 \ge 0
```

```math
\frac{\mathrm{d}V}{\mathrm{d}t} \le -\alpha_{\mathrm{damp}}(t) \cdot V + \sigma_{\mathrm{noise}}^2
```

```math
\tau_{\mathrm{dwell}} > \frac{\ln(1 + \mu_{\mathrm{jump}})}{\alpha_{\min}}, \quad V(t^+) \le (1 + \mu_{\mathrm{jump}}) V(t^-)
```

---
