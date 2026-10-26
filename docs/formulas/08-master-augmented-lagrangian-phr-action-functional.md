[Back to theory contents](../THEORY.md)

### Section 8: Master Augmented Lagrangian (PHR) Action Functional

```math
\begin{aligned}
\mathcal{J}_{\mathrm{AGI\_Master}}\big[ \mathbf{\Xi}(t), \mathbf{a}(t) \big] &= \int_{0}^T \Bigg[ \mathcal{F}_{\mathrm{pragmatic}}^{\mathrm{Sylvester}}\big( \mathbf{\Xi} \big) + \mathcal{K}_{\mathrm{VIB}}\big( \mathbf{\Xi}, \mathbf{a} \big) + \lambda_{\mathrm{goal}}(t) \mathbb{D}_{\mathrm{KL}}\left( Q(s \mid \mathbf{a}) \parallel P^{*}(s) \right) - \lambda_{\mathrm{Pearl}}(t) \mathcal{I}_{\mathrm{Pearl}}^{\mathrm{causal}}\big( \mathbf{do}(\mathbf{a}) \big) \\
&\quad - \mu_{\mathrm{ground}}(t) \mathcal{I}_{\mathrm{grounding}} + \gamma_{\mathrm{G\ddot{o}del}}(t) \left( \frac{d_{\mathcal{H}^{d-1}}\left( \boldsymbol{\mu}_{\mathrm{self}}^{*}(t), \overline{\mathcal{T}}_{\mathrm{introspect}}(t) \right)}{R_{\mathcal{H}}} \right)^2 + \mathcal{L}_{\mathrm{kin}} + \mathcal{L}_{\mathrm{Lip}} + \mathcal{L}_{\mathrm{Eikonal}} \\
&\quad + \mathcal{L}_{\mathrm{Landauer}} + \sum_{k=1}^2 \left( \frac{\rho_k}{2} \max\left( 0.0, \mathcal{C}_k(\mathbf{\Xi}) - C_k^{*} + \frac{\lambda_k(t)}{\rho_k} \right)^2 - \frac{\lambda_k(t)^2}{2 \rho_k} \right) \Bigg] \mathrm{d}t
\end{aligned}
```

Eikonal residual penalty:

```math
\mathcal{L}_{\mathrm{Eikonal}} = \lambda_{\mathrm{Eik}} \sum_{l=1}^L \sum_{i \in V^{(l)}} \left( \left\| \nabla_{\mathcal{H}^{d-1}} S\left(\boldsymbol{\mu}_i^{(l)}\right) \right\|_{\mathcal{L}}^2 - \varepsilon_{\mathrm{visc}} \Delta_{\mathcal{H}^{d-1}} S\left(\boldsymbol{\mu}_i^{(l)}\right) - n^2\left(\boldsymbol{\mu}_i^{(l)}\right) \right)^2
```

Landauer information erasure dissipation penalty:

```math
\mathcal{L}_{\mathrm{Landauer}} = k_B \ln(2) \cdot \mathbf{T}_i \cdot \max\left(0, -\dot{u}_k\right) \cdot \mathbb{I}\left(u_k < \theta_{\mathrm{prune}}\right)
```

Kinetic limits and Moreau-Yosida multiplier update:

```math
\mathcal{L}_{\mathrm{kin}} = \sum_{l=1}^L \sum_{i \in V^{(l)}} \frac{\kappa_v}{2} \|\mathbf{w}_i^{(l)}(t)\|_2^2
```

```math
\mathcal{C}_1(\mathbf{\Xi}) = \max_{l \in \{1,\dots,L\}} \max_{i \in V^{(l)}} \max\left( \sigma_{\max}\left(\mathbf{W}_i^{(1, l)}\right), \sigma_{\max}\left(\mathbf{W}_i^{(2, l)}\right) \right) \le C_1^{*} = 1.0
```

```math
\mathcal{C}_2(\mathbf{\Xi}) = \frac{1}{\sum_{l=1}^L |V^{(l)}|} \sum_{l=1}^L \sum_{i \in V^{(l)}} \|\mathbf{w}_i^{(l)}(t)\|_2^2 \le C_2^{*} = v_{\mathrm{budget}}^2
```

```math
\lambda_k(t + \Delta t) = \max\left(0, \lambda_k(t) + \frac{\Delta t \cdot \eta_\lambda}{1 + \gamma_{\mathrm{prox}} \Delta t} \cdot \max\left( -\frac{\lambda_k(t)}{\rho_k}, \mathcal{C}_k(\mathbf{\Xi}(t)) - C_k^{*} \right)\right)
```

The outer projection keeps inequality multipliers non-negative; without it, an inactive constraint can drive a zero multiplier below zero.

---
