[Back to theory contents](../THEORY.md)

### Section 36: Sectional Lobachevsky-Riemann Curvature Adapter
Generalized distance in manifolds of dynamically variable sectional curvature $`\kappa`$:

```math
d_\kappa(x, y) = \frac{1}{\sqrt{|\kappa|}} \mathrm{arcosh}\left( 1 + \frac{2|\kappa| \|x - y\|^2}{(1 + \kappa \|x\|^2)(1 + \kappa \|y\|^2)} \right)
```

Layer sectional curvature $`\kappa_l(t) \in (-\infty, +\infty)`$ adapts via gradient descent along the free energy surface:

```math
\frac{\partial \kappa_l}{\partial t} = -\eta_\kappa \nabla_{\kappa_l} \mathcal{F}_{\mathrm{pragmatic}}^{\mathrm{Sylvester}}
```

---
