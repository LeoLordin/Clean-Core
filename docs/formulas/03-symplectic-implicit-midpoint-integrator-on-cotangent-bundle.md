[Back to theory contents](../THEORY.md)

### Section 3: Symplectic Implicit Midpoint Integrator on Cotangent Bundle
The phase point $`\mathbf{\Gamma}_i^{(l)} = (\boldsymbol{\mu}_i^{(l)}, \mathbf{p}_i^{(l)}) \in T^{*}\mathcal{H}^d`$ obeys strict algebraic constraints:

```math
\sigma_1(\boldsymbol{\mu}) = \langle \boldsymbol{\mu}, \boldsymbol{\mu} \rangle_{\mathcal{L}} + 1.0 \equiv 0, \qquad \sigma_2(\boldsymbol{\mu}, \mathbf{p}) = \langle \boldsymbol{\mu}, \mathbf{p} \rangle_{\mathcal{L}} \equiv 0
```

The midpoint symplectic integrator evaluates forces at $`\boldsymbol{\mu}_{t+1/2} = \frac{\boldsymbol{\mu}_t + \boldsymbol{\mu}_{t+\Delta t}}{2}`$ and $`\mathbf{p}_{t+1/2} = \frac{\mathbf{p}_t + \mathbf{p}_{t+\Delta t}}{2}`$:

```math
\boldsymbol{\mu}_{i, t+\frac{\Delta t}{2}}^{(l)} = \frac{\boldsymbol{\mu}_{i, t}^{(l)} + \boldsymbol{\mu}_{i, t+\Delta t}^{(l)}}{2}, \qquad \mathbf{p}_{i, t+\frac{\Delta t}{2}}^{(l)} = \frac{\mathbf{p}_{i, t}^{(l)} + \mathbf{p}_{i, t+\Delta t}^{(l)}}{2}
```

```math
\mathbf{p}_{i, t+\Delta t}^{(l)} = \mathbf{p}_{i, t}^{(l)} - \Delta t \left( \nabla_{\mathcal{H}^d} \mathcal{F}_{\mathrm{pragmatic}}^{\mathrm{Sylvester}}\left(\boldsymbol{\mu}_{i, t+\frac{\Delta t}{2}}^{(l)}\right) + \lambda_{\mathrm{mid}} \cdot \boldsymbol{\eta}_{\mathcal{L}} \boldsymbol{\mu}_{i, t+\frac{\Delta t}{2}}^{(l)} \right)
```

```math
\boldsymbol{\mu}_{i, t+\Delta t}^{(l)} = \boldsymbol{\mu}_{i, t}^{(l)} + \Delta t \cdot \boldsymbol{\eta}_{\mathcal{L}}^{-1} \mathbf{p}_{i, t+\frac{\Delta t}{2}}^{(l)}
```

Constraint multipliers $`\lambda_{\mathrm{mid}}`$ and $`\mu_{\mathrm{mid}}`$ are determined uniquely such that:

```math
\left\langle \boldsymbol{\mu}_{i, t+\Delta t}^{(l)}, \boldsymbol{\mu}_{i, t+\Delta t}^{(l)} \right\rangle_{\mathcal{L}} \equiv -1.000000000000
```

```math
\left\langle \boldsymbol{\mu}_{i, t+\Delta t}^{(l)}, \mathbf{p}_{i, t+\Delta t}^{(l)} \right\rangle_{\mathcal{L}} \equiv 0.000000000000
```

Preservation of the canonical symplectic 2-form:

```math
\omega_{t+\Delta t} = \sum_{k=1}^d \mathrm{d}p_k(t+\Delta t) \wedge \mathrm{d}\mu^k(t+\Delta t) \equiv \sum_{k=1}^d \mathrm{d}p_k(t) \wedge \mathrm{d}\mu^k(t) = \omega_t
```

---
