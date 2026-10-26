[Back to theory contents](../THEORY.md)

### Mathematical Operators and Notations
* $`\langle \mathbf{x}, \mathbf{y} \rangle_{\mathcal{L}} = -x_0 y_0 + \sum_{k=1}^{d-1} x_k y_k`$: Minkowski bilinear form of signature $`(-, +, \dots, +)`$.
* $`\|\mathbf{v}\|_{\mathcal{L}} = \sqrt{\max(0.0, \langle \mathbf{v}, \mathbf{v} \rangle_{\mathcal{L}})}`$: Minkowski seminorm for space-like vectors.
* $`\|\mathbf{w}\|_2 = \sqrt{\sum_{k=1}^{d-1} w_k^2}`$: Standard Euclidean norm in local orthonormal tetrad frame.
* $`\|\mathbf{W}\|_F = \sqrt{\sum_{i, j} W_{ij}^2}`$: Frobenius matrix norm.
* $`d_{\mathcal{H}^d}(\mathbf{x}, \mathbf{y}) = 2 \mathrm{arcsinh}\left(0.5 \sqrt{\max(0, \langle \mathbf{x}-\mathbf{y}, \mathbf{x}-\mathbf{y} \rangle_{\mathcal{L}})}\right)`$: Numerically stable chordal geodesic distance.
* $`\mathrm{Exp}_{\boldsymbol{\mu}}^{\mathcal{H}^d}(\mathbf{v})`$: Geodesic exponential mapping from $`T_{\boldsymbol{\mu}}\mathcal{H}^d`$ onto $`\mathcal{H}^d`$ with Taylor expansion for small velocities.
* $`\mathrm{Log}_{\boldsymbol{\mu}}^{\mathcal{H}^d}(\mathbf{y})`$: Riemannian logarithmic mapping projecting manifold coordinates into tangent vectors.
* $`\mathbf{P}_{\mathbf{x} \to \mathbf{y}}(\mathbf{v})`$: Isometric parallel transport operator along geodesics.
* $`h_{\mathrm{log}}(\mathbf{A}) = \ln\left(\frac{1}{d} \mathrm{Tr}\left(\mathrm{expm}(\mathbf{A}\circ\mathbf{A})\right)\right)`$: Smooth NOTEARS acyclicity constraint.
* $`\mathrm{expm}(\mathbf{M}) = \sum_{k=0}^\infty \frac{\mathbf{M}^k}{k!}`$: Matrix exponential.
* $`\mathrm{Tr}(\mathbf{M})`$: Trace of a square matrix.
* $`\det(\mathbf{M})`$: Determinant of a square matrix.
* $`\rho(\mathbf{M})`$: Spectral radius of a matrix (maximum eigenvalue modulus), evaluated via real Schur decomposition $`\mathbf{M} = \mathbf{Z} \mathbf{T} \mathbf{Z}^\top`$.
* $`\mathbb{D}_{\mathrm{KL}}(Q \parallel P)`$: Kullback-Leibler divergence between probability distributions.
* $`\mathcal{F}_{\mathrm{pragmatic}}^{\mathrm{Sylvester}}`$: Sylvester Pragmatic Free Energy functional summed from Dirichlet layer $`l=0`$ to layer $`L`$.
* $`\mathcal{J}_{\mathrm{AGI\_Master}}`$: Master Augmented Lagrangian (PHR) action functional with Landauer thermal dissipation.
* $`V(\mathbf{\Xi})`$: Unified Lyapunov-Hespanha dynamic stability candidate functional.
* $`\odot`$: Hadamard element-wise vector product.
* $`\circ`$: Hadamard element-wise matrix product.
* $`\parallel`$: Vector and tensor concatenation operator.
* $`\wedge`$: Grassmann exterior differential product.
* $`\mathrm{sg}[\cdot]`$: Gradient-stop (Stop-Gradient) operator.
* $`\mathrm{Softclip}_{[a, b]}(x)`$: Smooth saturation function bounding values within $`[a, b]`$.
* $`\mathrm{clip}(x, a, b)`$: Hard numerical clamping operator.
* $`\sigma(x) = \frac{1}{1 + e^{-x}}`$: Standard logistic sigmoid activation.
* $`\sigma'(x) = \sigma(x)(1 - \sigma(x))`$: First derivative of logistic sigmoid.
* $`\mathrm{sech}(x) = \frac{2}{e^x + e^{-x}}`$: Hyperbolic secant function.
* $`\mathrm{sinhc}(x) = \frac{\sinh(x)}{x}`$: Cardinal hyperbolic sine with Taylor series expansion around zero.
* $`\delta(t - t_k)`$: Dirac delta distribution.
* $`\mathbb{I}(\text{cond})`$: Indicator function ($`1`$ if condition is true, $`0`$ otherwise).
* $`\mathrm{step}(x)`$: Heaviside step function ($`1`$ if $`x \ge 0`$, $`0`$ otherwise).
* $`\mathrm{QR\_Q}(\mathbf{M})`$: Extraction of orthonormal basis via thin QR matrix decomposition.

---

## Conclusion & Architectural Parity
All equations in this specification are dimensionally consistent, Lorentz-invariant, and bounded. Continuous Weierstrass parameterization ensures that machine rounding errors cannot eject state trajectories from the hyperboloid manifold. 

The complete dynamic kernel executes in $`O(1)`$ operations per moving entity per timestep, operating deterministically within real-time AAA game engine and robotic control budgets.
