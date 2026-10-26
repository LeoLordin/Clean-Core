[Back to theory contents](../THEORY.md)

### State Variables and Dynamic Vector Fields
* $`\mathbf{\Xi}_i^{(l)}(t)`$: Full microstate supervector of neuron $`i`$ in layer $`l`$ ($`16\,768\text{ bytes}`$).
* $`\mathbf{\Gamma}_i^{(l)}(t) = (\boldsymbol{\mu}_i^{(l)}, \mathbf{p}_i^{(l)})`$: Canonical phase point in $`T^{*}\mathcal{H}^{d-1}`$ ($`256\text{ bytes}`$ FP64).
* $`\boldsymbol{\mu}_i^{(l)}(t) \in \mathcal{H}^{d-1}`$: Coordinate vector on the Lorentz hyperboloid ($`128\text{ bytes}`$ FP64).
* $`\mathbf{p}_i^{(l)}(t) \in T_{\boldsymbol{\mu}_i^{(l)}}^{*}\mathcal{H}^{d-1}`$: Canonical momentum vector ($`128\text{ bytes}`$ FP64).
* $`\overline{\boldsymbol{\mu}}^{(l)}(t) \in \mathcal{H}^{d-1}`$: Running Fréchet barycenter of layer $`l`$ ($`128\text{ bytes}`$ FP64).
* $`\boldsymbol{\mu}_{\mathrm{pred}, i}^{(l)}(t) \in \mathcal{H}^{d-1}`$: Downward top-down prediction coordinate on the manifold.
* $`\boldsymbol{\mu}_{\mathrm{self}}^{*}(t) \in \mathcal{H}^{d-1}`$: Instantaneous ego-identity coordinate of the Gödelian operator.
* $`\overline{\mathcal{T}}_{\mathrm{introspect}}(t) \in \mathcal{H}^{d-1}`$: Running continuous introspection trajectory.
* $`\boldsymbol{\mu}_0^{(0)}(t) \in \mathcal{H}^{d-1}`$: Pinned Dirichlet sensory input vector.
* $`\mathbf{v}_i^{(l)}(t) \in T_{\boldsymbol{\mu}_i}\mathcal{H}^{d-1}`$: Physical tangent velocity vector on the hyperboloid.
* $`\mathbf{w}_i^{(l)}(t) \in \mathbb{R}^{d-1}`$: Inference velocity in the local orthonormal tetrad frame ($`120\text{ bytes}`$ FP64).
* $`\mathbf{w}_{\mathrm{det}, i}^{(l)}(t), \mathbf{w}_{\mathrm{raw}, i}^{(l)}(t)`$: Deterministic and raw unclipped stochastic velocity registers ($`120\text{ bytes}`$ FP64).
* $`\mathbf{w}_{\mathrm{geodesic}, i}^{(l)}(t)`$: Optical geodesic velocity vector derived from the Eikonal action gradient.
* $`\mathbf{w}_{\mathrm{relax}, i}^{(l)}`$: Inertia-free System-1 relaxation velocity during semantic queries.
* $`\tilde{\mathbf{v}}_{\mathrm{forces}, i}^{(l)}(t)`$: Total internal inference force vector in local frame.
* $`\tilde{\mathbf{v}}_{\mathrm{drift\_Ito}, i}^{(l)}(t)`$: Temperature-scaled Itô-to-Stratonovich drift correction vector.
* $`\tilde{\mathbf{v}}_{\mathrm{conformal}, i}^{(l)}(t)`$: Fictitious conformal drift force vector.
* $`\tilde{\mathbf{v}}_{\mathrm{bottom-up}, i}^{(l)}(t)`$: Bottom-up prediction error forces from child neurons in layer $`l-1`$.
* $`v_{\mathrm{self}}^{*}(t) \in \mathbb{R}^{d-1}`$: Fixed-point coordinates of the ego-operator in the top layer barycenter frame.
* $`\boldsymbol{\varepsilon}_i^{(l)}(t) \in T_{\boldsymbol{\mu}_i}\mathcal{H}^{d-1}`$: Tangent prediction error vector between state and prediction.
* $`\tilde{\boldsymbol{\varepsilon}}_i^{(l)}(t) \in \mathbb{R}^{d-1}`$: Local frame prediction error vector ($`60\text{ bytes}`$ FP32).
* $`\mathbf{a}(t) \in [-1, 1]^{d_{\mathrm{act}}}`$: Continuous motor action vector ($`32\text{ bytes}`$ FP32).
* $`\mathbf{a}_{\mathrm{target}}(t)`$: Target motor action synthesized by the wave and ToM modules ($`32\text{ bytes}`$ FP32).
* $`\mathbf{a}_{\mathrm{counterfactual}}(t)`$: Counterfactual motor action with regularized orthogonal perturbation.
* $`\mathbf{c}_k \in \mathbb{R}^{d-1}`$: SoftVQ concept centroids in the pole tangent space ($`\mathbf{c}_0 = \mathbf{0}`$).
* $`m_{\mathrm{pruned}, k} \in \{0, 1\}`$: Centroid activity boolean mask ($`0`$ is pruned, $`1`$ is active; $`m_{\mathrm{pruned}, 0} \equiv 1`$).
* $`u_k(t) \in [0, \infty)`$: Exponential token usage frequency counter (FP32).
* $`\text{feat}_{\mathrm{hist}}(t)`$: Historical macro-feature representation vector ($`512\text{ bytes}`$ FP32).
* $`\tilde{\mathbf{d}}_i^{(l)} \in \mathbb{R}^{d-1}`$: Diagonal residual error variance vector ($`60\text{ bytes}`$ FP32).
* $`\mathbf{s}_i^{(l)} \in \mathbb{R}^r`$: Principal singular values of error covariance ($`16\text{ bytes}`$ FP32).
* $`\mathbf{s}_i(t) \in \{0, 1\}^{M_{\mathrm{crossbar}}}`$: Neuromorphic discrete spike vector ($`64\text{ bytes}`$ UINT8).
* $`\lambda_{\mathrm{spike}, i}^{(l)}(t)`$: Neuromorphic spike generation firing rate ($`256\text{ bytes}`$ FP32).
* $`\tilde{\mathbf{a}}_i^{(l)}`$: Context vector assembled from parent neurons in the pole tangent space.
* $`\mathbf{h}_i^{(l)}`$: Activation vector of the perceptron hidden layer ($`128\text{ bytes}`$ FP32).
* $`\mathbf{u}_{2, i}^{(l)}, \mathbf{u}_{2, \mathrm{safe}, i}^{(l)}`$: Perceptron output vector before and after Euclidean retraction ($`60\text{ bytes}`$ FP32).
* $`\mathbf{g}_i^{(1, l)}, \mathbf{g}_i^{(2, l)}`$: Perceptron gradient sensitivity vectors in layer barycenter frame.
* $`\mathbf{b}_{\mathrm{self}}(t) \in \mathbb{R}^{d-1}`$: Dynamic bias vector of the Gödelian ego-reflection engine ($`60\text{ bytes}`$ FP32).
* $`\mathbf{m}_{\mathrm{poly}}(a^{*}) \in \mathbb{R}^{d_{\mathrm{act}}}`$: Polyhedral AIXI decision pressure vector ($`32\text{ bytes}`$ FP32).
* $`\mathbf{T}, \mathbf{Z}`$: Real Schur form and orthogonal transformation buffers for bounding $`\rho(\mathbf{A}_{\mathrm{SCM}})`$.
* $`\kappa_l(t)`$: Adaptive sectional curvature of layer $`l`$ (Section 36).

---
