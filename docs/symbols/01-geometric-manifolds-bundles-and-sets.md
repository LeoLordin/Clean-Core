[Back to theory contents](../THEORY.md)

### Geometric Manifolds, Bundles, and Sets
* $`\mathcal{H}^{d-1}`$: Lorentz hyperboloid manifold (upper sheet of the two-sheeted hyperboloid in Minkowski space $`\mathbb{R}^{1, d-1}`$):

```math
\mathcal{H}^{d-1} = \left\{ \mathbf{x} \in \mathbb{R}^d \;\middle|\; \langle \mathbf{x}, \mathbf{x} \rangle_{\mathcal{L}} = -x_0^2 + \sum_{k=1}^{d-1} x_k^2 = -1, \quad x_0 > 0 \right\}
```

* $`T_{\boldsymbol{\mu}}\mathcal{H}^{d-1}`$: Tangent vector space orthogonal to $`\boldsymbol{\mu}`$ under the Minkowski bilinear form:

```math
T_{\boldsymbol{\mu}}\mathcal{H}^{d-1} = \left\{ \mathbf{v} \in \mathbb{R}^d \;\middle|\; \langle \boldsymbol{\mu}, \mathbf{v} \rangle_{\mathcal{L}} = 0 \right\}
```

* $`T_{\boldsymbol{\mu}}^{*}\mathcal{H}^{d-1}`$: Cotangent space (space of canonical momenta 1-forms) dual to $`T_{\boldsymbol{\mu}}\mathcal{H}^{d-1}`$.
* $`T^{*}\mathcal{H}^{d-1}`$: Cotangent bundle representing canonical position-momentum phase space:

```math
T^{*}\mathcal{H}^{d-1} = \bigcup_{\boldsymbol{\mu} \in \mathcal{H}^{d-1}} \left( \{\boldsymbol{\mu}\} \times T_{\boldsymbol{\mu}}^{*}\mathcal{H}^{d-1} \right)
```

* $`T_{\mathbf{o}}\mathcal{H}^{d-1}`$: Tangent space at the canonical origin $`\mathbf{o} = (1, 0, \dots, 0)^\top`$, isomorphic to $`\mathbb{R}^{d-1}`$.
* $`\mathbb{S}^{d-2}`$: Unit Euclidean sphere parameterizing directional vectors in the hyperboloid metric factorization.
* $`\Delta^{|\mathcal{O}|-1}`$: Probability simplex over the alphabet of environmental observations $`\mathcal{O}`$.
* $`\mathcal{P}^{*}`$: Polyhedral decision cell in the observation simplex inside which optimal actions remain invariant.
* $`\mathcal{M}_i^{(l)}`$: Riemannian configuration manifold of the full neuron microstate.
* $`\mathcal{C}`$: SoftVQ codebook of spatial centroids defined in the pole tangent space.
* $`\mathcal{C}_{\mathrm{active}}`$: Active subset of codebook centroids ($`m_{\mathrm{pruned}} = 1`$).
* $`\mathcal{K}_{\mathrm{protected}}`$: Protected semantic token indices exempt from pruning operations.
* $`\mathcal{B}`$: Mini-batch of environmental observation snapshots.
* $`\mathcal{M}_{\mathrm{fast}}`$: Rapid episodic memory buffer partitioned into stratified reservoirs $`\mathcal{M}_k`$.
* $`\mathcal{S}_{\mathrm{tail}}`$: Set of avalanche neurons whose precision-weighted error exceeds threshold $`\varepsilon_{\min}`$.
* $`V^{(l)}`$: Set of neurons comprising layer $`l`$ of the hierarchy.
* $`\mathrm{parents}(i)`$: Set of parent neurons in layer $`l+1`$ providing context to neuron $`i`$.
* $`\mathrm{children}(i)`$: Set of child neurons in layer $`l-1`$ providing bottom-up prediction errors to neuron $`i`$.

---
