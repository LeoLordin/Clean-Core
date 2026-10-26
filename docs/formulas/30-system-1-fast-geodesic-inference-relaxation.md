[Back to theory contents](../THEORY.md)

### Section 30: System-1 Fast Geodesic Inference Relaxation

```math
\boldsymbol{\mu}_i^{(l)}(\tau + 1) = \mathrm{Exp}_{\boldsymbol{\mu}_i^{(l)}(\tau)}^{\mathcal{H}^{d-1}}\left( E(\boldsymbol{\mu}_i^{(l)}(\tau)) \cdot \mathbf{w}_{\mathrm{relax}, i}^{(l)}(tau) \right), \quad \tau \in \{0, 1, \dots, K_{\mathrm{relax}}-1\}
```

```math
\mathbf{w}_{\mathrm{relax}, i}^{(l)}(\tau) = \left( \mathbf{I}_{d-1} + \frac{\Delta t}{\epsilon_0} \pi_i \tilde{\mathbf{\Pi}}_i \right)^{-1} \left( \frac{\Delta t}{\epsilon_0} \tilde{\mathbf{v}}_{\mathrm{bottom-up}, i}^{(l)}(\tau) \right)
```

```math
\mathbf{w}_{\mathrm{relax}, i}^{(l)} \leftarrow \frac{\mathbf{w}_{\mathrm{relax}, i}^{(l)}}{\max\left(1.0, \frac{\|\mathbf{w}_{\mathrm{relax}, i}^{(l)}\|_2}{v_{\max}}\right)}
```

---
