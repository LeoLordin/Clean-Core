[Back to theory contents](../THEORY.md)

### Section 14: Minkowski Metric & Geodesic Operators

```math
\langle \mathbf{x}, \mathbf{y} \rangle_{\mathcal{L}} = -x_0 y_0 + \sum_{k=1}^{d-1} x_k y_k, \quad \boldsymbol{\eta}_{\mathcal{L}} = \mathrm{diag}(-1, 1, \dots, 1)
```

```math
\|\mathbf{v}\|_{\mathcal{L}} = \sqrt{\max\left( 0.0, \langle \mathbf{v}, \mathbf{v} \rangle_{\mathcal{L}} \right)}
```

```math
d_{\mathcal{H}^{d-1}}(\mathbf{x}, \mathbf{y}) = 2 \mathrm{arcsinh}\left( \frac{1}{2} \sqrt{\max\left( 0.0, \langle \mathbf{x} - \mathbf{y}, \mathbf{x} - \mathbf{y} \rangle_{\mathcal{L}} \right)} \right)
```

```math
\mathrm{Exp}_{\boldsymbol{\mu}}^{\mathcal{H}^{d-1}}(\mathbf{v}) = \begin{cases} \left( 1 + \frac{1}{2}\|\mathbf{v}\|_{\mathcal{L}}^2 \right) \boldsymbol{\mu} + \left( 1 + \frac{1}{6}\|\mathbf{v}\|_{\mathcal{L}}^2 \right) \mathbf{v}, & \|\mathbf{v}\|_{\mathcal{L}} < 10^{-4} \\ \cosh\left( \|\mathbf{v}\|_{\mathcal{L}} \right) \boldsymbol{\mu} + \sinh\left( \|\mathbf{v}\|_{\mathcal{L}} \right) \frac{\mathbf{v}}{\|\mathbf{v}\|_{\mathcal{L}}}, & \|\mathbf{v}\|_{\mathcal{L}} \ge 10^{-4} \end{cases}
```

```math
\mathbf{u} = \mathbf{y} + \langle \boldsymbol{\mu}, \mathbf{y} \rangle_{\mathcal{L}} \boldsymbol{\mu}, \quad \mathrm{Log}_{\boldsymbol{\mu}}^{\mathcal{H}^{d-1}}(\mathbf{y}) = \begin{cases} \left( 1 - \frac{\|\mathbf{u}\|_{\mathcal{L}}^2}{6} + \frac{3 \|\mathbf{u}\|_{\mathcal{L}}^4}{40} \right) \mathbf{u}, & \|\mathbf{u}\|_{\mathcal{L}} < 10^{-4} \\ \frac{d_{\mathcal{H}^{d-1}}(\boldsymbol{\mu}, \mathbf{y})}{\|\mathbf{u}\|_{\mathcal{L}}} \cdot \mathbf{u}, & \|\mathbf{u}\|_{\mathcal{L}} \ge 10^{-4} \end{cases}
```

```math
\mathbf{P}_{\mathbf{x} \to \mathbf{y}}(\mathbf{v}) = \mathbf{v} + \frac{\langle \mathbf{v}, \mathbf{y} \rangle_{\mathcal{L}}}{1 - \langle \mathbf{x}, \mathbf{y} \rangle_{\mathcal{L}}} (\mathbf{x} + \mathbf{y})
```

---
