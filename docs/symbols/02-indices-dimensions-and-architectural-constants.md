[Back to theory contents](../THEORY.md)

### Indices, Dimensions, and Architectural Constants
* $`l \in \{0, 1, \dots, L\}`$: Hierarchical layer index ($`0`$ is Dirichlet sensory layer, $`L`$ is ego-reflection layer).
* $`i \in V^{(l)}`$: Unique neuron index in layer $`l`$.
* $`j \in \mathrm{parents}(i)`$: Parent neuron index in layer $`l+1`$.
* $`k \in \mathrm{children}(i)`$: Child neuron index in layer $`l-1`$.
* $`d = 16`$: Ambient Minkowski dimension ($`\mathbb{R}^{16}`$).
* $`r = 4`$: Factorization rank of the DLR error covariance subspace ($`r \ll d-1`$).
* $`M_{\mathrm{crossbar}} = 64`$: Physical neuromorphic memristive crossbar lines.
* $`M_{\mathrm{trace}} = 32`$: Number of Rademacher vectors in Hutchinson's stochastic trace estimator.
* $`h_{\mathrm{dim}} = 32`$: Hidden dimension of projection perceptrons.
* $`d_{\mathrm{act}} = 8`$: Continuous motor action degrees of freedom.
* $`d_{\mathrm{scm}} = 16`$: Vertices in the structural causal model graph.
* $`D_{\mathrm{hist}} = 128`$: Historical macro-state feature dimension.
* $`K_{\mathrm{relax}} = 4`$: Fast System-1 relaxation steps for semantic inference queries.
* $`K_{\mathrm{infer}} = 4`$: Fast inference steps per single K-FAC natural gradient synaptic update.

---
