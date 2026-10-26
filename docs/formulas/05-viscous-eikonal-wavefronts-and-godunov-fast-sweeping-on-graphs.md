[Back to theory contents](../THEORY.md)

### Section 5: Viscous Eikonal Wavefronts & Godunov Fast Sweeping on Graphs
Continuous Eikonal equation with Laplace-Beltrami viscosity regularization:

```math
\left\| \nabla_{\mathcal{H}^d} S(\boldsymbol{\mu}) \right\|_{\mathcal{L}}^2 - \varepsilon_{\mathrm{visc}} \Delta_{\mathcal{H}^d} S(\boldsymbol{\mu}) = n^2(\boldsymbol{\mu}) = \frac{1}{\Omega^2(\boldsymbol{\mu})}
```

Discrete Godunov upwind scheme on the neural connectivity graph $`\mathcal{N}(i)`$:

```math
\sum_{k=1}^m \frac{\left( S_i - S_{(k)} \right)^2}{d_{\mathcal{H}^d}^2\left( \boldsymbol{\mu}_i^{(l)}, \boldsymbol{\mu}_{(k)}^{(l)} \right)} - \varepsilon_{\mathrm{visc}} \left( \Delta_{\mathcal{H}^d} S \right)_i = \frac{1}{\Omega^2\left(\boldsymbol{\mu}_i^{(l)}\right)}, \quad S_{(1)} \le S_{(2)} \le \dots \le S_{(m)} < S_i
```

Analytical root update via the Fast Sweeping method:

```math
A_m = \sum_{k=1}^m \frac{1}{d_k^2}, \quad B_m = \sum_{k=1}^m \frac{S_{(k)}}{d_k^2}, \quad C_m = \sum_{k=1}^m \frac{S_{(k)}^2}{d_k^2}
```

```math
S_i = \frac{B_m + \sqrt{\max\left( 0.0, B_m^2 - A_m \left( C_m - \frac{1}{\Omega_i^2} \right) \right)}}{A_m}
```

Optical velocity vector:

```math
\mathbf{w}_{\mathrm{geodesic}, i}^{(l)}(t) = -v_{\max} \cdot \frac{E\left(\boldsymbol{\mu}_i^{(l)}\right)^\top \boldsymbol{\eta}_{\mathcal{L}} \nabla_{\mathcal{H}^d} S\left(\boldsymbol{\mu}_i^{(l)}\right)}{\max\left(\delta_{\mathrm{eps}}, \left\| \nabla_{\mathcal{H}^d} S\left(\boldsymbol{\mu}_i^{(l)}\right) \right\|_{\mathcal{L}}\right)}
```

---
