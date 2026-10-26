[Back to theory contents](../THEORY.md)

## Preamble: Binary Layout & Memory Topology

### 1. Reference Profile Configuration
* **Ambient Minkowski Dimension:** $`d = 16`$ (Hyperboloid manifold dimension $`d - 1 = 15`$).
* **Fisher-Rao Error Covariance Factorization Rank (DLR):** $`r = 4`$ ($`r \ll d-1`$).
* **Projection Perceptron Hidden Layer Dimension:** $`h_{\mathrm{dim}} = 32`$.
* **Continuous Motor Actuation Degrees of Freedom:** $`d_{\mathrm{act}} = 8`$.
* **Structural Causal Model (NOTEARS) Vertices:** $`d_{\mathrm{scm}} = 16`$.
* **Neuromorphic Memristive Crossbar Lines:** $`M_{\mathrm{crossbar}} = 64`$.
* **Historical Introspection Macro-Vector Dimension:** $`D_{\mathrm{hist}} = 128`$.
* **SoftVQ Codebook Maximum Capacity:** $`K_{\max} = 1024`$.
* **Episodic Replay Memory Capacity:** $`M_{\mathrm{capacity}} = 10\,000`$.
* **Anti-Livelock VRAM Checkpoint Ring Buffer Depth:** $`K_{\mathrm{check}} = 5`$.

### 2. Data Representation & IEEE 754 Standards
* **FP64 (IEEE 754 Double Precision, 8 bytes):** Strictly allocated for manifold coordinates $`\boldsymbol{\mu}`$, canonical momenta $`\mathbf{p}`$, conformal scaling factor $`\Omega`$, chordal distances $`d_{\mathcal{H}^d}`$, and symplectic constraints. Guarantees numerical preservation of the Minkowski pseudonorm $`\langle \boldsymbol{\mu}, \boldsymbol{\mu} \rangle_{\mathcal{L}} = -1.0`$ down to $`10^{-15}`$.
* **FP32 (IEEE 754 Single Precision, 4 bytes):** Used for synaptic weight matrices $`\mathbf{W}`$, K-FAC Kronecker factors, gradient tensors, prediction error vectors $`\tilde{\boldsymbol{\varepsilon}}`$, and precision matrices $`\tilde{\mathbf{\Pi}}`$.
* **Complex64 ($`2 \times \text{FP32}`$, 8 bytes):** Reserved for spectral modes of the Cole-Hopf value function $`\psi`$ and Harish-Chandra holographic interference memory.
* **INT8 / UINT8 (1 byte):** Used for neuromorphic discrete spikes $`\mathbf{s}_i \in \{0, 1\}`$, pruning boolean masks $`m_{\mathrm{pruned}}`$, and protected semantic token tags.

### 3. Single-Neuron Microstate Byte Layout (64-byte Aligned)
The unpadded payload of a single neuron microstate $`\mathbf{\Xi}_i^{(l)}`$ evaluates to **$`16\,760`$ bytes**. Aligned to 64-byte cache lines ($`L1/L2`$ processor boundaries), the padded structure occupies **$`16\,768`$ bytes** ($`16.38\text{ KB}`$, exactly 262 cache lines):

| Field Symbol | Data Type | Shape | Byte Size | Functional Description |
| :--- | :--- | :--- | :--- | :--- |
| $`\boldsymbol{\mu}_i^{(l)}`$ | FP64 | $`(16,)`$ | 128 | Spatiotemporal coordinate on the Lorentz hyperboloid $`\mathcal{H}^{16}`$ |
| $`\mathbf{p}_i^{(l)}`$ | FP64 | $`(16,)`$ | 128 | Canonical cotangent momentum vector in $`T^{*}\mathcal{H}^{16}`$ |
| $`\mathbf{w}_i^{(l)}`$ | FP64 | $`(15,)`$ | 120 | Kinematic inference velocity in orthonormal tetrad frame |
| $`\mathbf{w}_{\mathrm{det}, i}^{(l)}, \mathbf{w}_{\mathrm{raw}, i}^{(l)}`$ | FP64 | $`(15,) \times 2`$ | 240 | Intermediate deterministic and stochastic velocity registers |
| $`S_i, \nabla_{\mathcal{H}^d} S_i`$ | FP64 | $`(1,) + (15,)`$ | 128 | Scalar Eikonal optical action and its Riemannian gradient |
| $`\psi_i, \nabla_{\mathcal{H}^d} \psi_i`$ | FP64 | $`(1,) + (15,)`$ | 128 | Cole-Hopf wave potential and target superposition gradient |
| $`\Omega_i, \nabla_{\mathcal{H}^d} \ln \Omega_i`$ | FP64 | $`(1,) + (15,)`$ | 128 | Maupertuis conformal factor and conformal drift gradient |
| $`\tilde{\boldsymbol{\varepsilon}}_i^{(l)}`$ | FP32 | $`(15,)`$ | 60 | Prediction error vector projected into local tetrad frame |
| $`\tilde{\mathbf{d}}_i^{(l)}`$ | FP32 | $`(15,)`$ | 60 | Diagonal residual error variance (DLR model) |
| $`\mathbf{s}_i^{(l)}`$ | FP32 | $`(4,)`$ | 16 | Singular value spectrum of principal covariance modes |
| $`\tilde{\mathbf{V}}_i^{(l)}`$ | FP32 | $`(15, 4)`$ | 240 | Orthonormal subspace basis for DLR factorization (Oja-QR) |
| $`\tilde{\mathbf{\Pi}}_i^{(l)}`$ | FP32 | $`(15, 15)`$ | 900 | Woodbury-inverted full precision matrix |
| $`\pi_i^{(l)}, \mathbf{T}_i`$ | FP32 | $`(1,) \times 2`$ | 8 | Scalar variational precision and thermodynamic temperature |
| $`\mathbf{W}_i^{(1, l)}, \mathbf{b}_i^{(1, l)}`$ | FP32 | $`(32, 15), (32,)`$ | 2048 | Synaptic weights and biases of first projection layer |
| $`\mathbf{W}_i^{(2, l)}, \mathbf{b}_i^{(2, l)}`$ | FP32 | $`(15, 32), (15,)`$ | 1980 | Synaptic weights and biases of second projection layer |
| $`\tilde{\mathbf{A}}_i^{(1, l)}, \tilde{\mathbf{B}}_i^{(1, l)}`$ | FP32 | $`(15, 15), (32, 32)`$ | 4996 | Input/gradient Kronecker covariance factors (Layer 1) |
| $`\tilde{\mathbf{A}}_i^{(2, l)}, \tilde{\mathbf{B}}_i^{(2, l)}`$ | FP32 | $`(32, 32), (15, 15)`$ | 4996 | Input/gradient Kronecker covariance factors (Layer 2) |
| $`\lambda_{\mathrm{spike}}, p_{\mathrm{spike}}`$ | FP32 | $`(64,) \times 2`$ | 512 | Neuromorphic spike generation rate and probability |
| $`\mathbf{s}_i`$ | UINT8 | $`(64,)`$ | 64 | Discrete spike vector for memristive crossbar interface |
| Hardware Padding | — | — | 8 | Zero-padding to maintain exact 64-byte alignment |
| **Total per Neuron** | — | — | **$`16\,768`$ bytes** | **Fully aligned single-neuron microstate structure** |

### 4. Global Macro-Topology & VRAM Footprint
* **Structural Causal Graph ($`\mathbf{A}_{\mathrm{SCM}}`$):** Adjacency matrix, gradient buffers, and Padé matrix exponential cache: $`3 \times (16 \times 16 \times 4) = \mathbf{3072\text{ bytes}}`$.
* **Schur Decomposition ($`\mathbf{T}, \mathbf{Z}`$) & Landauer Buffers:** $`\mathbf{+2056\text{ bytes}}`$.
* **Gödelian Ego-Reflection:** Weight tensors $`\mathbf{W}_{\mathrm{self}, v}`$, $`\mathbf{W}_{\mathrm{self}, \mathrm{hist}}`$, resolvent caches, and historical features: $`\mathbf{10\,052\text{ bytes}}`$.
* **Motor EFE Circuit & Polyhedral AIXI Gateway:** Linear decoders and motor action buffers: $`\mathbf{960\text{ bytes}}`$.
* **SoftVQ Semantic Codebook ($`K_{\max} = 1024`$):** Centroids $`\mathbf{c}_k`$, usage counters $`u_k`$, and boolean masks: $`\mathbf{66\,560\text{ bytes}}`$.
* **Harish-Chandra Holographic Memory ($`M_{\mathrm{capacity}} = 10\,000`$):**
  * Complex spectral modes $`\tilde{A}_m`$ (Complex64): $`10\,000 \times 15 \times 8 = \mathbf{1\,200\,000\text{ bytes}}`$ ($`1.14\text{ MB}`$).
  * Episodic coordinate buffer $`\boldsymbol{\mu}_m`$ (FP64): $`10\,000 \times 16 \times 8 = \mathbf{1\,280\,000\text{ bytes}}`$ ($`1.22\text{ MB}`$).
* **Total Static Shared Memory:** $`\mathbf{2\,562\,700\text{ bytes}}`$ ($`\approx \mathbf{2.44\text{ MB}}`$).
* **Complete Network Snapshot ($`N = 1000`$ Neurons):**  

```math
M_{\mathrm{snapshot}} = 1000 \times 16\,768\text{ bytes} + 2\,562\,700\text{ bytes} = \mathbf{19\,330\,700\text{ bytes}} \approx \mathbf{18.432\text{ MB}}
```

* **Anti-Livelock Ring Buffer ($`K_{\mathrm{check}} = 5`$ Checkpoints):**  

```math
M_{\mathrm{ring\_buffer}} = 5 \times 18.432\text{ MB} = \mathbf{92.17\text{ MB}}
```

---
