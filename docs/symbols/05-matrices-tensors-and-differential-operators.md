[Back to theory contents](../THEORY.md)

### Matrices, Tensors, and Differential Operators
* $`E(\boldsymbol{\mu}_i^{(l)}) \in \mathbb{R}^{d \times (d-1)}`$: Closed-form orthonormal Lorentz boost tetrad frame.
* $`\boldsymbol{\eta}_{\mathcal{L}} = \mathrm{diag}(-1, 1, \dots, 1) \in \mathbb{R}^{d \times d}`$: Minkowski metric tensor of signature $`(-, +, \dots, +)`$.
* $`\tilde{g}_{AB}(\boldsymbol{\mu}) \in \mathbb{R}^{d \times d}`$: Maupertuis-Jacobi conformal metric tensor.
* $`\tilde{\Gamma}_{BC}^A`$: Christoffel symbols of the second kind on the conformal manifold.
* $`\omega \in \Lambda^2(T^{*}\mathcal{H}^d)`$: Canonical symplectic 2-form of the cotangent phase space.
* $`\Delta_{\mathcal{H}^d}`$: Laplace-Beltrami differential operator on the Lorentz hyperboloid.
* $`\nabla_{\mathcal{H}^d}`$: Riemannian gradient operator: $`\nabla_{\mathcal{H}^d} f = \boldsymbol{\eta}_{\mathcal{L}}^{-1} \mathrm{d}f + \langle \boldsymbol{\mu}, \boldsymbol{\eta}_{\mathcal{L}}^{-1} \mathrm{d}f \rangle_{\mathcal{L}} \boldsymbol{\mu}`$.
* $`\tilde{\mathbf{\Sigma}}_i^{(l)} \in \mathbb{R}^{(d-1) \times (d-1)}`$: Fisher-Rao local prediction error covariance matrix ($`900\text{ bytes}`$ FP32).
* $`\tilde{\mathbf{\Pi}}_i^{(l)} = (\tilde{\mathbf{\Sigma}}_i^{(l)})^{-1}`$: Full local precision matrix inverted via Woodbury identity ($`900\text{ bytes}`$ FP32).
* $`\tilde{\mathbf{V}}_i^{(l)} \in \mathbb{R}^{(d-1) \times r}`$: Orthonormal basis of principal error covariance modes ($`240\text{ bytes}`$ FP32).
* $`\tilde{\mathbf{A}}_i^{(1, l)}, \tilde{\mathbf{B}}_i^{(1, l)}`$: Input and gradient Kronecker factors for K-FAC natural gradient update (FP32).
* $`\mathbf{P}_{\mathbf{x} \to \mathbf{y}}`$: Isometric parallel transport operator along geodesics.
* $`\mathbf{P}_{\mathbf{a}^\perp}`$: Tikhonov-regularized projection operator onto the subspace orthogonal to actions.
* $`\mathbf{W}_i^{(1, l)}, \mathbf{W}_i^{(2, l)}`$: Weight matrices of the two-layer projection perceptron (FP32).
* $`\mathbf{b}_i^{(1, l)}, \mathbf{b}_i^{(2, l)}`$: Bias vectors of the projection perceptron (FP32).
* $`\mathbf{W}_{\mathrm{self}, v}, \mathbf{W}_{\mathrm{self}, \mathrm{hist}}`$: Trainable weight matrices of the Gödelian ego-reflection engine (FP32).
* $`\mathbf{A}_{\mathrm{SCM}}`$: Differentiable adjacency matrix of the structural causal model ($`1024\text{ bytes}`$ FP32).
* $`\mathbf{W}_{\mathrm{motor}}, \mathbf{b}_{\mathrm{motor}}`$: Variational motor kinematic decoder parameters (FP32).
* $`\mathbf{W}_{\mathrm{social}}`$: Projection matrix coupling Theory of Mind vectors to physical motor action (FP32).
* $`\mathbf{W}_{\mathrm{poly}}`$: Projection matrix mapping polyhedral AIXI pressure to motor degrees of freedom (FP32).
* $`\mathbf{W}_{\mathrm{key}}, \mathbf{W}_{\mathrm{val}}`$: Key and value projection matrices of hyperbolic associative attention (FP32).
* $`\mathbf{W}_{\mathrm{analog\_crossbar}} \in \mathbb{R}^{(d-1) \times M_{\mathrm{crossbar}}}`$: Memristive crossbar conductance matrix ($`3840\text{ bytes}`$ FP32).
* $`\mathbf{W}_{\mathrm{enc}}, \mathbf{b}_{\mathrm{enc}}`$: Neuromorphic straight-through spike encoder parameters (FP32).
* $`\mathbf{I}_{d-1}`$: Identity matrix of dimension $`(d-1) \times (d-1)`$.
* $`\mathbf{e}_k`$: Standard Cartesian basis unit vector in Euclidean subspace.

---
