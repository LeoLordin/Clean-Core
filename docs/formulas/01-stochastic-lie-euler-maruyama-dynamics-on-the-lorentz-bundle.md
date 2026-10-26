[Back to theory contents](../THEORY.md)

### Section 1: Stochastic Lie-Euler-Maruyama Dynamics on the Lorentz Bundle
Evolution occurs across three distinct physical timescales:

```math
\epsilon_0 \sim 10^{-3}\text{ s} \quad (\text{inference}), \qquad \epsilon_1 \sim 1\text{ s} \quad (\text{weight adaptation}), \qquad \epsilon_2 \sim 10^5\text{ s} \quad (\text{synaptogenesis})
```

The sensory layer $`l=0`$ is pinned via a strict Dirichlet boundary condition:

```math
\boldsymbol{\mu}_0^{(0)}(t) = \mathrm{Exp}_{\mathbf{o}}^{\mathcal{H}^{d-1}}\left( \mathbf{v}_{\mathrm{bundle}}(t) \right) \in \mathcal{H}^{d-1}, \quad \mathbf{w}_0^{(0)}(t) \equiv \mathbf{0}
```

Zero-phase-lag Fréchet barycenter streaming estimator across layers $`l \in \{1, \dots, L\}`$:

```math
\overline{\boldsymbol{\mu}}^{(l)}(t) = \begin{cases} \mathbf{o} = (1, 0, \dots, 0)^\top, & t = 0 \\ \mathrm{Exp}_{\overline{\boldsymbol{\mu}}^{(l)}(t-\Delta t)}^{\mathcal{H}^{d-1}}\left( \gamma_{\mathrm{bary}} \cdot \mathbf{v}_{\mathrm{bary}}^{(l)}(t) \right), & t > 0 \end{cases}
```

```math
\mathbf{v}_{\mathrm{bary}}^{(l)}(t) = \frac{1}{|V^{(l)}|} \sum_{j \in V^{(l)}} \mathrm{Log}_{\overline{\boldsymbol{\mu}}^{(l)}(t-\Delta t)}^{\mathcal{H}^{d-1}}\left( \boldsymbol{\mu}_j^{(l)}(t) \right) \in T_{\overline{\boldsymbol{\mu}}^{(l)}}\mathcal{H}^{d-1}
```

Canonical closed-form orthonormal Lorentz boost tetrad frame:

```math
E\left(\boldsymbol{\mu}_i^{(l)}(t)\right)_{:, k} = \mathbf{e}_k + \frac{\left(\boldsymbol{\mu}_i^{(l)}(t)\right)_k}{1.0 + \left(\boldsymbol{\mu}_i^{(l)}(t)\right)_0} \left( \mathbf{o} + \boldsymbol{\mu}_i^{(l)}(t) \right), \quad k \in \{1, \dots, d-1\}
```

```math
E\left(\boldsymbol{\mu}_i^{(l)}\right)^\top \boldsymbol{\eta}_{\mathcal{L}} E\left(\boldsymbol{\mu}_i^{(l)}\right) \equiv \mathbf{I}_{d-1}, \quad \left(\boldsymbol{\mu}_i^{(l)}\right)^\top \boldsymbol{\eta}_{\mathcal{L}} E\left(\boldsymbol{\mu}_i^{(l)}\right) \equiv \mathbf{0}^\top
```

Chordal distance to the layer barycenter and kinematic Laplace-Beltrami drift regulator:

```math
r_i^{(l)}(t) = d_{\mathcal{H}^{d-1}}\left( \boldsymbol{\mu}_i^{(l)}(t), \overline{\boldsymbol{\mu}}^{(l)}(t) \right) = 2 \mathrm{arcsinh}\left( \frac{1}{2} \sqrt{\max\left( 0.0, \left\langle \boldsymbol{\mu}_i^{(l)}(t) - \overline{\boldsymbol{\mu}}^{(l)}(t), \boldsymbol{\mu}_i^{(l)}(t) - \overline{\boldsymbol{\mu}}^{(l)}(t) \right\rangle_{\mathcal{L}} \right)} \right)
```

```math
f_{\mathrm{drift\_safe}}(r) = \frac{1}{1 + r \cdot \tanh(r)}
```

Prediction error in the local orthonormal frame:

```math
\tilde{\boldsymbol{\varepsilon}}_i^{(l)}(t) = E\left(\boldsymbol{\mu}_i^{(l)}(t)\right)^\top \boldsymbol{\eta}_{\mathcal{L}} \boldsymbol{\varepsilon}_i^{(l)}(t) \in \mathbb{R}^{d-1}
```

Fisher-Rao Diagonal plus Low-Rank (DLR) error covariance factorization:

```math
\tilde{\mathbf{\Sigma}}_i^{(l)}(t) = \mathrm{diag}\left(\tilde{\mathbf{d}}_i^{(l)}(t)\right) + \tilde{\mathbf{V}}_i^{(l)}(t) \mathrm{diag}\left(\mathbf{s}_i^{(l)}(t)\right) \left( \tilde{\mathbf{V}}_i^{(l)}(t) \right)^\top \in \mathbb{R}^{(d-1) \times (d-1)}
```

Analytical precision matrix inversion via the Woodbury matrix identity:

```math
\tilde{\mathbf{\Pi}}_i^{(l)}(t) = \mathbf{D}^{-1} - \mathbf{D}^{-1} \tilde{\mathbf{V}}_i^{(l)} \left( \mathrm{diag}\left(\mathbf{s}_i^{(l)}\right)^{-1} + \left(\tilde{\mathbf{V}}_i^{(l)}\right)^\top \mathbf{D}^{-1} \tilde{\mathbf{V}}_i^{(l)} \right)^{-1} \left(\tilde{\mathbf{V}}_i^{(l)}\right)^\top \mathbf{D}^{-1}, \quad \mathbf{D} = \mathrm{diag}\left(\tilde{\mathbf{d}}_i^{(l)}(t)\right)
```

Temperature-scaled Itô-to-Stratonovich drift correction:

```math
\tilde{\mathbf{v}}_{\mathrm{drift\_Ito}, i}^{(l)}(t) = -0.5 \cdot \alpha_{\mathrm{temp}} \cdot \mathbf{T}_i(t) \cdot \mathrm{sech}^2\left( \frac{\alpha_{\mathrm{temp}}}{T_{\max}} \max\left(0.0, \left( \tilde{\boldsymbol{\varepsilon}}_i^{(l)}(t) \right)^\top \tilde{\mathbf{\Pi}}_i^{(l)}(t) \tilde{\boldsymbol{\varepsilon}}_i^{(l)}(t)\right) \right) \tilde{\mathbf{\Pi}}_i^{(l)}(t) \tilde{\boldsymbol{\varepsilon}}_i^{(l)}(t)
```

Canonical Hamiltonian equations on the conformal bundle:

```math
H_{\mathrm{neuron}}(\boldsymbol{\mu}, \mathbf{p}) = \frac{1}{2 \Omega^2(\boldsymbol{\mu})} \langle \mathbf{p}, \mathbf{p} \rangle_{\mathcal{L}} + \mathcal{F}_{\mathrm{pragmatic}}^{\mathrm{Sylvester}}(\boldsymbol{\mu})
```

```math
\frac{\mathrm{d}\boldsymbol{\mu}}{\mathrm{d}t} = \frac{1}{\Omega^2(\boldsymbol{\mu})} \boldsymbol{\eta}_{\mathcal{L}}^{-1} \mathbf{p}, \qquad \frac{\mathrm{d}\mathbf{p}}{\mathrm{d}t} = -\nabla_{\mathcal{H}^{d-1}} \mathcal{F}_{\mathrm{pragmatic}}^{\mathrm{Sylvester}}(\boldsymbol{\mu}) + \frac{\nabla \Omega}{\Omega^3} \langle \mathbf{p}, \mathbf{p} \rangle_{\mathcal{L}}
```

Total internal dynamic forces of inference:

```math
\begin{aligned}
\tilde{\mathbf{v}}_{\mathrm{forces}, i}^{(l)}(t) &= \pi_i^{(l)}(t) \sigma_\beta \tilde{\mathbf{\Pi}}_i^{(l)}(t) \tilde{\boldsymbol{\varepsilon}}_i^{(l)}(t) + \alpha_{\mathrm{damp}}(t) \cdot \epsilon_0 \cdot f_{\mathrm{drift\_safe}}\left( r_i^{(l)}(t) \right) E\left(\boldsymbol{\mu}_i^{(l)}\right)^\top \boldsymbol{\eta}_{\mathcal{L}} \mathrm{Log}_{\boldsymbol{\mu}_i^{(l)}}^{\mathcal{H}^{d-1}}\left( \overline{\boldsymbol{\mu}}^{(l)} \right) \\
&\quad + \tilde{\mathbf{\Delta}}_{\mathrm{fast}, i}^{(l)}(t) + \tilde{\mathbf{v}}_{\mathrm{bottom-up}, i}^{(l)}(t) + \tilde{\mathbf{v}}_{\mathrm{conformal}, i}^{(l)}(t) + E\left(\boldsymbol{\mu}_i^{(l)}\right)^\top \boldsymbol{\eta}_{\mathcal{L}} \frac{\nabla \Omega}{\Omega^3} \langle \mathbf{p}_i^{(l)}, \mathbf{p}_i^{(l)} \rangle_{\mathcal{L}} - \left( \lambda_2(t) + \rho_2 \max\left(0.0, \mathcal{C}_2(\mathbf{\Xi}) - C_2^{*}\right) \right) \mathbf{w}_i^{(l)}(t-\Delta t)
\end{aligned}
```

```math
\tilde{\mathbf{v}}_{\mathrm{bottom-up}, i}^{(l)}(t) = \sum_{k \in \mathrm{children}(i)} E_{ik}^{(l)}(t) \cdot E\left(\boldsymbol{\mu}_i^{(l)}(t)\right)^\top \boldsymbol{\eta}_{\mathcal{L}} \mathrm{Log}_{\boldsymbol{\mu}_i^{(l)}(t)}^{\mathcal{H}^{d-1}}\left( \boldsymbol{\mu}_{\mathrm{pred}, k}^{(l-1)}(t) \right)
```

Unconditionally $`A`$-stable semi-implicit velocity splitting with CFL-calibrated Langevin noise:

```math
\mathbf{w}_{\mathrm{det}, i}^{(l)}(t) = \left( \mathbf{I}_{d-1} + \frac{\Delta t}{\epsilon_0} \pi_i^{(l)}(t) \sigma_\beta \tilde{\mathbf{\Pi}}_i^{(l)}(t) \right)^{-1} \cdot \left( \frac{\Delta t}{\epsilon_0} \tilde{\mathbf{v}}_{\mathrm{forces}, i}^{(l)}(t) + \Delta t \cdot \tilde{\mathbf{v}}_{\mathrm{drift\_Ito}, i}^{(l)}(t) \right)
```

```math
\sigma_{\mathrm{discrete}, i}^{(l)}(t) = \sqrt{2 \mathbf{T}_i(t) \cdot \min\left(1.0, \frac{\epsilon_0}{\Delta t}\right)} \cdot 0.05 \cdot \gamma_{\mathrm{anneal}}^{\mathrm{epoch}}
```

```math
\mathbf{w}_{\mathrm{raw}, i}^{(l)}(t) = \mathbf{w}_{\mathrm{det}, i}^{(l)}(t) + \sigma_{\mathrm{discrete}, i}^{(l)}(t) \cdot \boldsymbol{\xi}_{i, t}, \quad \boldsymbol{\xi}_{i, t} \sim \mathcal{N}\left(\mathbf{0}, \mathbf{I}_{d-1}\right)
```

Nonlinear Euclidean velocity clamping:

```math
\mathbf{w}_i^{(l)}(t) = \frac{\mathbf{w}_{\mathrm{raw}, i}^{(l)}(t)}{\max\left( 1.0, \frac{\|\mathbf{w}_{\mathrm{raw}, i}^{(l)}(t)\|_2}{v_{\max}} \right)}, \quad v_{\max} = 5.0
```

Canonical tangent lift and geodesic exponential stepping:

```math
\mathbf{v}_i^{(l)}(t) = E\left(\boldsymbol{\mu}_i^{(l)}(t)\right) \mathbf{w}_i^{(l)}(t) \in T_{\boldsymbol{\mu}_i^{(l)}}\mathcal{H}^{d-1}, \quad \|\mathbf{v}_i^{(l)}(t)\|_{\mathcal{L}} \equiv \|\mathbf{w}_i^{(l)}(t)\|_2
```

```math
\boldsymbol{\mu}_i^{(l)}(t + \Delta t) = \cosh\left(\|\mathbf{w}_i^{(l)}(t)\|_2\right) \boldsymbol{\mu}_i^{(l)}(t) + \mathrm{sinhc}\left(\|\mathbf{w}_i^{(l)}(t)\|_2\right) \mathbf{v}_i^{(l)}(t)
```

```math
\mathrm{sinhc}(x) = \begin{cases} 1.0 + \frac{x^2}{6} + \frac{x^4}{120}, & x < 10^{-4} \\ \frac{\sinh(x)}{x}, & x \ge 10^{-4} \end{cases}
```

Continuous radial normalization projection onto the Lorentz hyperboloid:

```math
\boldsymbol{\mu}_i^{(l)}(t + \Delta t) \leftarrow \frac{\boldsymbol{\mu}_i^{(l)}(t + \Delta t)}{\sqrt{\max\left(\delta_{\mathrm{eps}}, -\left\langle \boldsymbol{\mu}_i^{(l)}(t + \Delta t), \boldsymbol{\mu}_i^{(l)}(t + \Delta t) \right\rangle_{\mathcal{L}}\right)}}
```

Adaptive variational scalar precision:

```math
\pi_i^{(l)}(t + \Delta t) = \mathrm{Softclip}_{[\pi_{\min}, \pi_{\max}]}\left( \frac{d - 1}{\max\left( \delta_{\mathrm{eps}}, \left( \tilde{\boldsymbol{\varepsilon}}_i^{(l)}(t) \right)^\top \tilde{\mathbf{\Pi}}_i^{(l)}(t) \left( \tilde{\boldsymbol{\varepsilon}}_i^{(l)}(t) \right) \right)} \right)
```

K-FAC natural gradient synaptic update with Tikhonov damping and Frobenius norm clipping:

```math
\Delta \mathbf{W}_i^{(1, l)} = \left( \tilde{\mathbf{B}}_i^{(1, l)} + \delta_{\mathrm{kfac}}\mathbf{I} \right)^{-1} \left( \nabla_{\mathbf{W}_i^{(1, l)}} \mathcal{F}_{\mathrm{pragmatic}}^{\mathrm{Sylvester}} + \nabla_{\mathbf{W}_i^{(1, l)}} \mathcal{L}_{\mathrm{Lip}} + \nabla_{\mathbf{W}_i^{(1, l)}} \mathcal{L}_{\mathrm{PHR}, 1} \right) \left( \tilde{\mathbf{A}}_i^{(1, l)} + \delta_{\mathrm{kfac}}\mathbf{I} \right)^{-1}
```

```math
\mathbf{W}_i^{(1, l)}(t + \Delta t) = \mathbf{W}_i^{(1, l)}(t) - \frac{\Delta t}{\epsilon_1} \frac{\Delta \mathbf{W}_i^{(1, l)}}{\max\left(1.0, \frac{\|\Delta \mathbf{W}_i^{(1, l)}\|_F}{M_{\mathrm{step}}}\right)}, \quad M_{\mathrm{step}} = 1.0, \quad \delta_{\mathrm{kfac}} = 0.2
```

```math
\Delta \mathbf{W}_i^{(2, l)} = \left( \tilde{\mathbf{B}}_i^{(2, l)} + \delta_{\mathrm{kfac}}\mathbf{I} \right)^{-1} \left( \nabla_{\mathbf{W}_i^{(2, l)}} \mathcal{F}_{\mathrm{pragmatic}}^{\mathrm{Sylvester}} + \nabla_{\mathbf{W}_i^{(2, l)}} \mathcal{L}_{\mathrm{Lip}} + \nabla_{\mathbf{W}_i^{(2, l)}} \mathcal{L}_{\mathrm{PHR}, 1} \right) \left( \tilde{\mathbf{A}}_i^{(2, l)} + \delta_{\mathrm{kfac}}\mathbf{I} \right)^{-1}
```

```math
\mathbf{W}_i^{(2, l)}(t + \Delta t) = \mathbf{W}_i^{(2, l)}(t) - \frac{\Delta t}{\epsilon_1} \frac{\Delta \mathbf{W}_i^{(2, l)}}{\max\left(1.0, \frac{\|\Delta \mathbf{W}_i^{(2, l)}\|_F}{M_{\mathrm{step}}}\right)}
```

```math
\mathbf{b}_i^{(1, l)}(t + \Delta t) = \mathbf{b}_i^{(1, l)}(t) - \frac{\Delta t}{\epsilon_1} \left( \tilde{\mathbf{B}}_i^{(1, l)} + \delta_{\mathrm{kfac}}\mathbf{I} \right)^{-1} \nabla_{\mathbf{b}_i^{(1, l)}} \mathcal{F}_{\mathrm{pragmatic}}^{\mathrm{Sylvester}}
```

```math
\mathbf{b}_i^{(2, l)}(t + \Delta t) = \mathbf{b}_i^{(2, l)}(t) - \frac{\Delta t}{\epsilon_1} \left( \tilde{\mathbf{B}}_i^{(2, l)} + \delta_{\mathrm{kfac}}\mathbf{I} \right)^{-1} \nabla_{\mathbf{b}_i^{(2, l)}} \mathcal{F}_{\mathrm{pragmatic}}^{\mathrm{Sylvester}}
```

Synaptogenesis of graph topology (timescale $`\epsilon_2`$):

```math
E_{ji}^{(l+1)}(t + \Delta t) = \mathrm{Softclip}_{[E_{\min}, 1.0]}\left( E_{ji}^{(l+1)}(t) + \Delta t \left( \eta_E \frac{1}{\max\left(\delta_{\mathrm{eps}}, -\left\langle \boldsymbol{\mu}_i^{(l)}(t), \boldsymbol{\mu}_j^{(l+1)}(t) \right\rangle_{\mathcal{L}}\right)} - \lambda_E E_{ji}^{(l+1)}(t) \right) \right)
```

---
