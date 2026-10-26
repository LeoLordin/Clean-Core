# Clean Core 5.0: Formal Architectural Specification & Complete Mathematical Compendium

> **Document Status:** Reference Specification & Open RFC (`v0.1.0-alpha`)
> **Target Domains:** Non-Euclidean Geometric Deep Learning, Active Inference, Symplectic Mechanics, Algorithmic Information Theory, Causal Discovery, High-Performance Systems Architecture
> **Author:** [LeoLordin](https://github.com/LeoLordin)

---

## Table of Contents
* [Preamble: Binary Layout & Memory Topology](#preamble-binary-layout--memory-topology)
  * [1. Reference Profile Configuration](#1-reference-profile-configuration)
  * [2. Data Representation & IEEE 754 Standards](#2-data-representation--ieee-754-standards)
  * [3. Single-Neuron Microstate Byte Layout (64-byte Aligned)](#3-single-neuron-microstate-byte-layout-64-byte-aligned)
  * [4. Global Macro-Topology & VRAM Footprint](#4-global-macro-topology--vram-footprint)
* [Part I: Mathematical Foundations & Dynamical Equations](#part-i-mathematical-foundations--dynamical-equations)
  * [Section 1: Stochastic Lie-Euler-Maruyama Dynamics on the Lorentz Bundle](#section-1-stochastic-lie-euler-maruyama-dynamics-on-the-lorentz-bundle)
  * [Section 2: Maupertuis-Jacobi Conformal Differential Geometry](#section-2-maupertuis-jacobi-conformal-differential-geometry)
  * [Section 3: Symplectic Implicit Midpoint Integrator on Cotangent Bundle](#section-3-symplectic-implicit-midpoint-integrator-on-cotangent-bundle)
  * [Section 4: Linearized Cole-Hopf Active Inference (Log-Diffusion HJB)](#section-4-linearized-cole-hopf-active-inference-log-diffusion-hjb)
  * [Section 5: Viscous Eikonal Wavefronts & Godunov Fast Sweeping on Graphs](#section-5-viscous-eikonal-wavefronts--godunov-fast-sweeping-on-graphs)
  * [Section 6: Polyhedral Solomonoff-AIXI Decider & Safe Horizon](#section-6-polyhedral-solomonoff-aixi-decider--safe-horizon)
  * [Section 7: Full Microstate Supervector & Expected Free Energy Motor Loop](#section-7-full-microstate-supervector--expected-free-energy-motor-loop)
  * [Section 8: Master Augmented Lagrangian (PHR) Action Functional](#section-8-master-augmented-lagrangian-phr-action-functional)
  * [Section 9: Sylvester Pragmatic Free Energy](#section-9-sylvester-pragmatic-free-energy)
  * [Section 10: Retraction Perceptron & Tangent-Space SoftVQ](#section-10-retraction-perceptron--tangent-space-softvq)
  * [Section 11: Sigmoidal Spike Activation Masking](#section-11-sigmoidal-spike-activation-masking)
  * [Section 12: DLR Factorization with Oja-QR & K-FAC Natural Gradient](#section-12-dlr-factorization-with-oja-qr--k-fac-natural-gradient)
  * [Section 13: Spectral Lipschitz Regularization](#section-13-spectral-lipschitz-regularization)
  * [Section 14: Minkowski Metric & Geodesic Operators](#section-14-minkowski-metric--geodesic-operators)
  * [Section 15: Variational Information Bottleneck (VIB)](#section-15-variational-information-bottleneck-vib)
  * [Section 16: Differentiable Causal Do-Calculus & Schur-Capped NOTEARS](#section-16-differentiable-causal-do-calculus--schur-capped-notears)
  * [Section 17: Friston Active Inference Target Divergence](#section-17-friston-active-inference-target-divergence)
  * [Section 18: Multimodal Dirichlet Sensory Grounding](#section-18-multimodal-dirichlet-sensory-grounding)
  * [Section 19: Gödelian Ego-Reflection via Adjoint Resolvents](#section-19-gödelian-ego-reflection-via-adjoint-resolvents)
  * [Section 20: Sleep Replay Consolidation via Vitter Stratified Reservoirs](#section-20-sleep-replay-consolidation-via-vitter-stratified-reservoirs)
  * [Section 21: SoftVQ Semantic Codebook Evolution & Pruning](#section-21-softvq-semantic-codebook-evolution--pruning)
  * [Section 22: Thermodynamic Langevin Frame Noise](#section-22-thermodynamic-langevin-frame-noise)
  * [Section 23: Fast Associative Attention & Memristive Crossbar Currents](#section-23-fast-associative-attention--memristive-crossbar-currents)
  * [Section 24: Lyapunov-Hespanha Dynamic Stability](#section-24-lyapunov-hespanha-dynamic-stability)
  * [Section 25: Unbiased SOC Criticality Regulator](#section-25-unbiased-soc-criticality-regulator)
  * [Section 26: Poincaré Holographic Memory & Harish-Chandra Transforms](#section-26-poincaré-holographic-memory--harish-chandra-transforms)
  * [Section 27: Multi-Agent Conformal Theory of Mind (ToM)](#section-27-multi-agent-conformal-theory-of-mind-tom)
  * [Section 28: Anti-Livelock VRAM Ring Buffer with Thermal Annealing](#section-28-anti-livelock-vram-ring-buffer-with-thermal-annealing)
  * [Section 29: Neuromorphic STE Spike Encoder](#section-29-neuromorphic-ste-spike-encoder)
  * [Section 30: System-1 Fast Geodesic Inference Relaxation](#section-30-system-1-fast-geodesic-inference-relaxation)
  * [Section 31: Conformal Connection & Christoffel Symbols](#section-31-conformal-connection--christoffel-symbols)
  * [Section 32: Non-Separable Maupertuis-Jacobi Hamiltonian Mechanics](#section-32-non-separable-maupertuis-jacobi-hamiltonian-mechanics)
  * [Section 33: Generalized Cole-Hopf Diffusion on Riemannian Manifolds](#section-33-generalized-cole-hopf-diffusion-on-riemannian-manifolds)
  * [Section 34: Explicit Polyhedral Projection of AIXI Simplex](#section-34-explicit-polyhedral-projection-of-aixi-simplex)
  * [Section 35: Asynchronous Zero-Downtime Gödel JIT Engine Hot-Swapping](#section-35-asynchronous-zero-downtime-gödel-jit-engine-hot-swapping)
  * [Section 36: Sectional Lobachevsky-Riemann Curvature Adapter](#section-36-sectional-lobachevsky-riemann-curvature-adapter)
  * [Section 37: Lock-Free RCU Multi-Timescale Engine Dispatcher](#section-37-lock-free-rcu-multi-timescale-engine-dispatcher)
  * [Section 38: Dynamic Isometric Dimension Expansion](#section-38-dynamic-isometric-dimension-expansion)
* [Part II: Algorithmic Execution Specifications (Algorithms 1–13)](#part-ii-algorithmic-execution-specifications)
* [Part III: Complete Mathematical & Symbol Index](#part-iii-complete-mathematical--symbol-index)

---

## Preamble: Binary Layout & Memory Topology

### 1. Reference Profile Configuration
* **Ambient Minkowski Dimension:** $d = 16$ (Hyperboloid manifold dimension $d - 1 = 15$).
* **Fisher-Rao Error Covariance Factorization Rank (DLR):** $r = 4$ ($r \ll d-1$).
* **Projection Perceptron Hidden Layer Dimension:** $h_{\mathrm{dim}} = 32$.
* **Continuous Motor Actuation Degrees of Freedom:** $d_{\mathrm{act}} = 8$.
* **Structural Causal Model (NOTEARS) Vertices:** $d_{\mathrm{scm}} = 16$.
* **Neuromorphic Memristive Crossbar Lines:** $M_{\mathrm{crossbar}} = 64$.
* **Historical Introspection Macro-Vector Dimension:** $D_{\mathrm{hist}} = 128$.
* **SoftVQ Codebook Maximum Capacity:** $K_{\max} = 1024$.
* **Episodic Replay Memory Capacity:** $M_{\mathrm{capacity}} = 10\,000$.
* **Anti-Livelock VRAM Checkpoint Ring Buffer Depth:** $K_{\mathrm{check}} = 5$.

### 2. Data Representation & IEEE 754 Standards
* **FP64 (IEEE 754 Double Precision, 8 bytes):** Strictly allocated for manifold coordinates $\boldsymbol{\mu}$, canonical momenta $\mathbf{p}$, conformal scaling factor $\Omega$, chordal distances $d_{\mathcal{H}^d}$, and symplectic constraints. Guarantees numerical preservation of the Minkowski pseudonorm $\langle \boldsymbol{\mu}, \boldsymbol{\mu} \rangle_{\mathcal{L}} = -1.0$ down to $10^{-15}$.
* **FP32 (IEEE 754 Single Precision, 4 bytes):** Used for synaptic weight matrices $\mathbf{W}$, K-FAC Kronecker factors, gradient tensors, prediction error vectors $\tilde{\boldsymbol{\varepsilon}}$, and precision matrices $\tilde{\mathbf{\Pi}}$.
* **Complex64 ($2 \times \text{FP32}$, 8 bytes):** Reserved for spectral modes of the Cole-Hopf value function $\psi$ and Harish-Chandra holographic interference memory.
* **INT8 / UINT8 (1 byte):** Used for neuromorphic discrete spikes $\mathbf{s}_i \in \{0, 1\}$, pruning boolean masks $m_{\mathrm{pruned}}$, and protected semantic token tags.

### 3. Single-Neuron Microstate Byte Layout (64-byte Aligned)
The unpadded payload of a single neuron microstate $\mathbf{\Xi}_i^{(l)}$ evaluates to **$16\,760$ bytes**. Aligned to 64-byte cache lines ($L1/L2$ processor boundaries), the padded structure occupies **$16\,768$ bytes** ($16.38\text{ KB}$, exactly 262 cache lines):

| Field Symbol | Data Type | Shape | Byte Size | Functional Description |
| :--- | :--- | :--- | :--- | :--- |
| $\boldsymbol{\mu}_i^{(l)}$ | FP64 | $(16,)$ | 128 | Spatiotemporal coordinate on the Lorentz hyperboloid $\mathcal{H}^{16}$ |
| $\mathbf{p}_i^{(l)}$ | FP64 | $(16,)$ | 128 | Canonical cotangent momentum vector in $T^*\mathcal{H}^{16}$ |
| $\mathbf{w}_i^{(l)}$ | FP64 | $(15,)$ | 120 | Kinematic inference velocity in orthonormal tetrad frame |
| $\mathbf{w}_{\mathrm{det}, i}^{(l)}, \mathbf{w}_{\mathrm{raw}, i}^{(l)}$ | FP64 | $(15,) \times 2$ | 240 | Intermediate deterministic and stochastic velocity registers |
| $S_i, \nabla_{\mathcal{H}^d} S_i$ | FP64 | $(1,) + (15,)$ | 128 | Scalar Eikonal optical action and its Riemannian gradient |
| $\psi_i, \nabla_{\mathcal{H}^d} \psi_i$ | FP64 | $(1,) + (15,)$ | 128 | Cole-Hopf wave potential and target superposition gradient |
| $\Omega_i, \nabla_{\mathcal{H}^d} \ln \Omega_i$ | FP64 | $(1,) + (15,)$ | 128 | Maupertuis conformal factor and conformal drift gradient |
| $\tilde{\boldsymbol{\varepsilon}}_i^{(l)}$ | FP32 | $(15,)$ | 60 | Prediction error vector projected into local tetrad frame |
| $\tilde{\mathbf{d}}_i^{(l)}$ | FP32 | $(15,)$ | 60 | Diagonal residual error variance (DLR model) |
| $\mathbf{s}_i^{(l)}$ | FP32 | $(4,)$ | 16 | Singular value spectrum of principal covariance modes |
| $\tilde{\mathbf{V}}_i^{(l)}$ | FP32 | $(15, 4)$ | 240 | Orthonormal subspace basis for DLR factorization (Oja-QR) |
| $\tilde{\mathbf{\Pi}}_i^{(l)}$ | FP32 | $(15, 15)$ | 900 | Woodbury-inverted full precision matrix |
| $\pi_i^{(l)}, \mathbf{T}_i$ | FP32 | $(1,) \times 2$ | 8 | Scalar variational precision and thermodynamic temperature |
| $\mathbf{W}_i^{(1, l)}, \mathbf{b}_i^{(1, l)}$ | FP32 | $(32, 15), (32,)$ | 2048 | Synaptic weights and biases of first projection layer |
| $\mathbf{W}_i^{(2, l)}, \mathbf{b}_i^{(2, l)}$ | FP32 | $(15, 32), (15,)$ | 1980 | Synaptic weights and biases of second projection layer |
| $\tilde{\mathbf{A}}_i^{(1, l)}, \tilde{\mathbf{B}}_i^{(1, l)}$ | FP32 | $(15, 15), (32, 32)$ | 4996 | Input/gradient Kronecker covariance factors (Layer 1) |
| $\tilde{\mathbf{A}}_i^{(2, l)}, \tilde{\mathbf{B}}_i^{(2, l)}$ | FP32 | $(32, 32), (15, 15)$ | 4996 | Input/gradient Kronecker covariance factors (Layer 2) |
| $\lambda_{\mathrm{spike}}, p_{\mathrm{spike}}$ | FP32 | $(64,) \times 2$ | 512 | Neuromorphic spike generation rate and probability |
| $\mathbf{s}_i$ | UINT8 | $(64,)$ | 64 | Discrete spike vector for memristive crossbar interface |
| Hardware Padding | — | — | 8 | Zero-padding to maintain exact 64-byte alignment |
| **Total per Neuron** | — | — | **$16\,768$ bytes** | **Fully aligned single-neuron microstate structure** |

### 4. Global Macro-Topology & VRAM Footprint
* **Structural Causal Graph ($\mathbf{A}_{\mathrm{SCM}}$):** Adjacency matrix, gradient buffers, and Padé matrix exponential cache: $3 \times (16 \times 16 \times 4) = \mathbf{3072\text{ bytes}}$.
* **Schur Decomposition ($\mathbf{T}, \mathbf{Z}$) & Landauer Buffers:** $\mathbf{+2056\text{ bytes}}$.
* **Gödelian Ego-Reflection:** Weight tensors $\mathbf{W}_{\mathrm{self}, v}$, $\mathbf{W}_{\mathrm{self}, \mathrm{hist}}$, resolvent caches, and historical features: $\mathbf{10\,052\text{ bytes}}$.
* **Motor EFE Circuit & Polyhedral AIXI Gateway:** Linear decoders and motor action buffers: $\mathbf{960\text{ bytes}}$.
* **SoftVQ Semantic Codebook ($K_{\max} = 1024$):** Centroids $\mathbf{c}_k$, usage counters $u_k$, and boolean masks: $\mathbf{66\,560\text{ bytes}}$.
* **Harish-Chandra Holographic Memory ($M_{\mathrm{capacity}} = 10\,000$):**
  * Complex spectral modes $\tilde{A}_m$ (Complex64): $10\,000 \times 15 \times 8 = \mathbf{1\,200\,000\text{ bytes}}$ ($1.14\text{ MB}$).
  * Episodic coordinate buffer $\boldsymbol{\mu}_m$ (FP64): $10\,000 \times 16 \times 8 = \mathbf{1\,280\,000\text{ bytes}}$ ($1.22\text{ MB}$).
* **Total Static Shared Memory:** $\mathbf{2\,562\,700\text{ bytes}}$ ($\approx \mathbf{2.44\text{ MB}}$).
* **Complete Network Snapshot ($N = 1000$ Neurons):**

$$
M_{\mathrm{snapshot}} = 1000 \times 16\,768\text{ bytes} + 2\,562\,700\text{ bytes} = \mathbf{19\,330\,700\text{ bytes}} \approx \mathbf{18.432\text{ MB}}
$$

* **Anti-Livelock Ring Buffer ($K_{\mathrm{check}} = 5$ Checkpoints):**

$$
M_{\mathrm{ring\_buffer}} = 5 \times 18.432\text{ MB} = \mathbf{92.17\text{ MB}}
$$

---

## Part I: Mathematical Foundations & Dynamical Equations

---

### Section 1: Stochastic Lie-Euler-Maruyama Dynamics on the Lorentz Bundle
Evolution occurs across three distinct physical timescales:

$$
\epsilon_0 \sim 10^{-3}\text{ s} \quad (\text{inference}), \qquad \epsilon_1 \sim 1\text{ s} \quad (\text{weight adaptation}), \qquad \epsilon_2 \sim 10^5\text{ s} \quad (\text{synaptogenesis})
$$

The sensory layer $l=0$ is pinned via a strict Dirichlet boundary condition:

$$
\boldsymbol{\mu}_0^{(0)}(t) = \mathrm{Exp}_{\mathbf{o}}^{\mathcal{H}^d}\left( \mathbf{v}_{\mathrm{bundle}}(t) \right) \in \mathcal{H}^d, \quad \mathbf{w}_0^{(0)}(t) \equiv \mathbf{0}
$$

Zero-phase-lag Fréchet barycenter streaming estimator across layers $l \in \{1, \dots, L\}$:

$$
\overline{\boldsymbol{\mu}}^{(l)}(t) = \begin{cases} \mathbf{o} = (1, 0, \dots, 0)^\top, & t = 0 \\ \mathrm{Exp}_{\overline{\boldsymbol{\mu}}^{(l)}(t-\Delta t)}^{\mathcal{H}^d}\left( \gamma_{\mathrm{bary}} \cdot \mathbf{v}_{\mathrm{bary}}^{(l)}(t) \right), & t > 0 \end{cases}
$$

$$
\mathbf{v}_{\mathrm{bary}}^{(l)}(t) = \frac{1}{|V^{(l)}|} \sum_{j \in V^{(l)}} \mathrm{Log}_{\overline{\boldsymbol{\mu}}^{(l)}(t-\Delta t)}^{\mathcal{H}^d}\left( \boldsymbol{\mu}_j^{(l)}(t) \right) \in T_{\overline{\boldsymbol{\mu}}^{(l)}}\mathcal{H}^d
$$

Canonical closed-form orthonormal Lorentz boost tetrad frame:

$$
E\left(\boldsymbol{\mu}_i^{(l)}(t)\right)_{:, k} = \mathbf{e}_k + \frac{\left(\boldsymbol{\mu}_i^{(l)}(t)\right)_k}{1.0 + \left(\boldsymbol{\mu}_i^{(l)}(t)\right)_0} \left( \mathbf{o} + \boldsymbol{\mu}_i^{(l)}(t) \right), \quad k \in \{1, \dots, d-1\}
$$

$$
E\left(\boldsymbol{\mu}_i^{(l)}\right)^\top \boldsymbol{\eta}_{\mathcal{L}} E\left(\boldsymbol{\mu}_i^{(l)}\right) \equiv \mathbf{I}_{d-1}, \quad \left(\boldsymbol{\mu}_i^{(l)}\right)^\top \boldsymbol{\eta}_{\mathcal{L}} E\left(\boldsymbol{\mu}_i^{(l)}\right) \equiv \mathbf{0}^\top
$$

Chordal distance to the layer barycenter and kinematic Laplace-Beltrami drift regulator:

$$
r_i^{(l)}(t) = d_{\mathcal{H}^d}\left( \boldsymbol{\mu}_i^{(l)}(t), \overline{\boldsymbol{\mu}}^{(l)}(t) \right) = 2 \mathrm{arcsinh}\left( \frac{1}{2} \sqrt{\max\left( 0.0, \left\langle \boldsymbol{\mu}_i^{(l)}(t) - \overline{\boldsymbol{\mu}}^{(l)}(t), \boldsymbol{\mu}_i^{(l)}(t) - \overline{\boldsymbol{\mu}}^{(l)}(t) \right\rangle_{\mathcal{L}} \right)} \right)
$$

$$
f_{\mathrm{drift\_safe}}(r) = \frac{1}{1 + r \cdot \tanh(r)}
$$

Prediction error in the local orthonormal frame:

$$
\tilde{\boldsymbol{\varepsilon}}_i^{(l)}(t) = E\left(\boldsymbol{\mu}_i^{(l)}(t)\right)^\top \boldsymbol{\eta}_{\mathcal{L}} \boldsymbol{\varepsilon}_i^{(l)}(t) \in \mathbb{R}^{d-1}
$$

Fisher-Rao Diagonal plus Low-Rank (DLR) error covariance factorization:

$$
\tilde{\mathbf{\Sigma}}_i^{(l)}(t) = \mathrm{diag}\left(\tilde{\mathbf{d}}_i^{(l)}(t)\right) + \tilde{\mathbf{V}}_i^{(l)}(t) \mathrm{diag}\left(\mathbf{s}_i^{(l)}(t)\right) \left( \tilde{\mathbf{V}}_i^{(l)}(t) \right)^\top \in \mathbb{R}^{(d-1) \times (d-1)}
$$

Analytical precision matrix inversion via the Woodbury matrix identity:

$$
\tilde{\mathbf{\Pi}}_i^{(l)}(t) = \mathbf{D}^{-1} - \mathbf{D}^{-1} \tilde{\mathbf{V}}_i^{(l)} \left( \mathrm{diag}\left(\mathbf{s}_i^{(l)}\right)^{-1} + \left(\tilde{\mathbf{V}}_i^{(l)}\right)^\top \mathbf{D}^{-1} \tilde{\mathbf{V}}_i^{(l)} \right)^{-1} \left(\tilde{\mathbf{V}}_i^{(l)}\right)^\top \mathbf{D}^{-1}, \quad \mathbf{D} = \mathrm{diag}\left(\tilde{\mathbf{d}}_i^{(l)}(t)\right)
$$

Temperature-scaled Itô-to-Stratonovich drift correction:

$$
\tilde{\mathbf{v}}_{\mathrm{drift\_Ito}, i}^{(l)}(t) = -0.5 \cdot \alpha_{\mathrm{temp}} \cdot \mathbf{T}_i(t) \cdot \mathrm{sech}^2\left( \frac{\alpha_{\mathrm{temp}}}{T_{\max}} \max\left(0.0, \left( \tilde{\boldsymbol{\varepsilon}}_i^{(l)}(t) \right)^\top \tilde{\mathbf{\Pi}}_i^{(l)}(t) \tilde{\boldsymbol{\varepsilon}}_i^{(l)}(t)\right) \right) \tilde{\mathbf{\Pi}}_i^{(l)}(t) \tilde{\boldsymbol{\varepsilon}}_i^{(l)}(t)
$$

Canonical Hamiltonian equations on the conformal bundle:

$$
H_{\mathrm{neuron}}(\boldsymbol{\mu}, \mathbf{p}) = \frac{1}{2 \Omega^2(\boldsymbol{\mu})} \langle \mathbf{p}, \mathbf{p} \rangle_{\mathcal{L}} + \mathcal{F}_{\mathrm{pragmatic}}^{\mathrm{Sylvester}}(\boldsymbol{\mu})
$$

$$
\frac{\mathrm{d}\boldsymbol{\mu}}{\mathrm{d}t} = \frac{1}{\Omega^2(\boldsymbol{\mu})} \boldsymbol{\eta}_{\mathcal{L}}^{-1} \mathbf{p}, \qquad \frac{\mathrm{d}\mathbf{p}}{\mathrm{d}t} = -\nabla_{\mathcal{H}^d} \mathcal{F}_{\mathrm{pragmatic}}^{\mathrm{Sylvester}}(\boldsymbol{\mu}) + \frac{\nabla \Omega}{\Omega^3} \langle \mathbf{p}, \mathbf{p} \rangle_{\mathcal{L}}
$$

Total internal dynamic forces of inference:

$$
\begin{aligned}
\tilde{\mathbf{v}}_{\mathrm{forces}, i}^{(l)}(t) &= \pi_i^{(l)}(t) \sigma_\beta \tilde{\mathbf{\Pi}}_i^{(l)}(t) \tilde{\boldsymbol{\varepsilon}}_i^{(l)}(t) + \alpha_{\mathrm{damp}}(t) \cdot \epsilon_0 \cdot f_{\mathrm{drift\_safe}}\left( r_i^{(l)}(t) \right) E\left(\boldsymbol{\mu}_i^{(l)}\right)^\top \boldsymbol{\eta}_{\mathcal{L}} \mathrm{Log}_{\boldsymbol{\mu}_i^{(l)}}^{\mathcal{H}^d}\left( \overline{\boldsymbol{\mu}}^{(l)} \right) \\
&\quad + \tilde{\mathbf{\Delta}}_{\mathrm{fast}, i}^{(l)}(t) + \tilde{\mathbf{v}}_{\mathrm{bottom-up}, i}^{(l)}(t) + \tilde{\mathbf{v}}_{\mathrm{conformal}, i}^{(l)}(t) + E\left(\boldsymbol{\mu}_i^{(l)}\right)^\top \boldsymbol{\eta}_{\mathcal{L}} \frac{\nabla \Omega}{\Omega^3} \langle \mathbf{p}_i^{(l)}, \mathbf{p}_i^{(l)} \rangle_{\mathcal{L}} - \left( \lambda_2(t) + \rho_2 \max\left(0.0, \mathcal{C}_2(\mathbf{\Xi}) - C_2^*\right) \right) \mathbf{w}_i^{(l)}(t-\Delta t)
\end{aligned}
$$

$$
\tilde{\mathbf{v}}_{\mathrm{bottom-up}, i}^{(l)}(t) = \sum_{k \in \mathrm{children}(i)} E_{ik}^{(l)}(t) \cdot E\left(\boldsymbol{\mu}_i^{(l)}(t)\right)^\top \boldsymbol{\eta}_{\mathcal{L}} \mathrm{Log}_{\boldsymbol{\mu}_i^{(l)}(t)}^{\mathcal{H}^d}\left( \boldsymbol{\mu}_{\mathrm{pred}, k}^{(l-1)}(t) \right)
$$

Unconditionally $A$-stable semi-implicit velocity splitting with CFL-calibrated Langevin noise:

$$
\mathbf{w}_{\mathrm{det}, i}^{(l)}(t) = \left( \mathbf{I}_{d-1} + \frac{\Delta t}{\epsilon_0} \pi_i^{(l)}(t) \sigma_\beta \tilde{\mathbf{\Pi}}_i^{(l)}(t) \right)^{-1} \cdot \left( \frac{\Delta t}{\epsilon_0} \tilde{\mathbf{v}}_{\mathrm{forces}, i}^{(l)}(t) + \Delta t \cdot \tilde{\mathbf{v}}_{\mathrm{drift\_Ito}, i}^{(l)}(t) \right)
$$

$$
\sigma_{\mathrm{discrete}, i}^{(l)}(t) = \sqrt{2 \mathbf{T}_i(t) \cdot \min\left(1.0, \frac{\epsilon_0}{\Delta t}\right)} \cdot 0.05 \cdot \gamma_{\mathrm{anneal}}^{\mathrm{epoch}}
$$

$$
\mathbf{w}_{\mathrm{raw}, i}^{(l)}(t) = \mathbf{w}_{\mathrm{det}, i}^{(l)}(t) + \sigma_{\mathrm{discrete}, i}^{(l)}(t) \cdot \boldsymbol{\xi}_{i, t}, \quad \boldsymbol{\xi}_{i, t} \sim \mathcal{N}\left(\mathbf{0}, \mathbf{I}_{d-1}\right)
$$

Nonlinear Euclidean velocity clamping:

$$
\mathbf{w}_i^{(l)}(t) = \frac{\mathbf{w}_{\mathrm{raw}, i}^{(l)}(t)}{\max\left( 1.0, \frac{\|\mathbf{w}_{\mathrm{raw}, i}^{(l)}(t)\|_2}{v_{\max}} \right)}, \quad v_{\max} = 5.0
$$

Canonical tangent lift and geodesic exponential stepping:

$$
\mathbf{v}_i^{(l)}(t) = E\left(\boldsymbol{\mu}_i^{(l)}(t)\right) \mathbf{w}_i^{(l)}(t) \in T_{\boldsymbol{\mu}_i^{(l)}}\mathcal{H}^d, \quad \|\mathbf{v}_i^{(l)}(t)\|_{\mathcal{L}} \equiv \|\mathbf{w}_i^{(l)}(t)\|_2
$$

$$
\boldsymbol{\mu}_i^{(l)}(t + \Delta t) = \cosh\left(\|\mathbf{w}_i^{(l)}(t)\|_2\right) \boldsymbol{\mu}_i^{(l)}(t) + \mathrm{sinhc}\left(\|\mathbf{w}_i^{(l)}(t)\|_2\right) \mathbf{v}_i^{(l)}(t)
$$

$$
\mathrm{sinhc}(x) = \begin{cases} 1.0 + \frac{x^2}{6} + \frac{x^4}{120}, & x < 10^{-4} \\ \frac{\sinh(x)}{x}, & x \ge 10^{-4} \end{cases}
$$

Continuous radial normalization projection onto the Lorentz hyperboloid:

$$
\boldsymbol{\mu}_i^{(l)}(t + \Delta t) \leftarrow \frac{\boldsymbol{\mu}_i^{(l)}(t + \Delta t)}{\sqrt{\max\left(\delta_{\mathrm{eps}}, -\left\langle \boldsymbol{\mu}_i^{(l)}(t + \Delta t), \boldsymbol{\mu}_i^{(l)}(t + \Delta t) \right\rangle_{\mathcal{L}}\right)}}
$$

Adaptive variational scalar precision:

$$
\pi_i^{(l)}(t + \Delta t) = \mathrm{Softclip}_{[\pi_{\min}, \pi_{\max}]}\left( \frac{d - 1}{\max\left( \delta_{\mathrm{eps}}, \left( \tilde{\boldsymbol{\varepsilon}}_i^{(l)}(t) \right)^\top \tilde{\mathbf{\Pi}}_i^{(l)}(t) \left( \tilde{\boldsymbol{\varepsilon}}_i^{(l)}(t) \right) \right)} \right)
$$

K-FAC natural gradient synaptic update with Tikhonov damping and Frobenius norm clipping:

$$
\Delta \mathbf{W}_i^{(1, l)} = \left( \tilde{\mathbf{B}}_i^{(1, l)} + \delta_{\mathrm{kfac}}\mathbf{I} \right)^{-1} \left( \nabla_{\mathbf{W}_i^{(1, l)}} \mathcal{F}_{\mathrm{pragmatic}}^{\mathrm{Sylvester}} + \nabla_{\mathbf{W}_i^{(1, l)}} \mathcal{L}_{\mathrm{Lip}} + \nabla_{\mathbf{W}_i^{(1, l)}} \mathcal{L}_{\mathrm{PHR}, 1} \right) \left( \tilde{\mathbf{A}}_i^{(1, l)} + \delta_{\mathrm{kfac}}\mathbf{I} \right)^{-1}
$$

$$
\mathbf{W}_i^{(1, l)}(t + \Delta t) = \mathbf{W}_i^{(1, l)}(t) - \frac{\Delta t}{\epsilon_1} \frac{\Delta \mathbf{W}_i^{(1, l)}}{\max\left(1.0, \frac{\|\Delta \mathbf{W}_i^{(1, l)}\|_F}{M_{\mathrm{step}}}\right)}, \quad M_{\mathrm{step}} = 1.0, \quad \delta_{\mathrm{kfac}} = 0.2
$$

$$
\Delta \mathbf{W}_i^{(2, l)} = \left( \tilde{\mathbf{B}}_i^{(2, l)} + \delta_{\mathrm{kfac}}\mathbf{I} \right)^{-1} \left( \nabla_{\mathbf{W}_i^{(2, l)}} \mathcal{F}_{\mathrm{pragmatic}}^{\mathrm{Sylvester}} + \nabla_{\mathbf{W}_i^{(2, l)}} \mathcal{L}_{\mathrm{Lip}} + \nabla_{\mathbf{W}_i^{(2, l)}} \mathcal{L}_{\mathrm{PHR}, 1} \right) \left( \tilde{\mathbf{A}}_i^{(2, l)} + \delta_{\mathrm{kfac}}\mathbf{I} \right)^{-1}
$$

$$
\mathbf{W}_i^{(2, l)}(t + \Delta t) = \mathbf{W}_i^{(2, l)}(t) - \frac{\Delta t}{\epsilon_1} \frac{\Delta \mathbf{W}_i^{(2, l)}}{\max\left(1.0, \frac{\|\Delta \mathbf{W}_i^{(2, l)}\|_F}{M_{\mathrm{step}}}\right)}
$$

$$
\mathbf{b}_i^{(1, l)}(t + \Delta t) = \mathbf{b}_i^{(1, l)}(t) - \frac{\Delta t}{\epsilon_1} \left( \tilde{\mathbf{B}}_i^{(1, l)} + \delta_{\mathrm{kfac}}\mathbf{I} \right)^{-1} \nabla_{\mathbf{b}_i^{(1, l)}} \mathcal{F}_{\mathrm{pragmatic}}^{\mathrm{Sylvester}}
$$

$$
\mathbf{b}_i^{(2, l)}(t + \Delta t) = \mathbf{b}_i^{(2, l)}(t) - \frac{\Delta t}{\epsilon_1} \left( \tilde{\mathbf{B}}_i^{(2, l)} + \delta_{\mathrm{kfac}}\mathbf{I} \right)^{-1} \nabla_{\mathbf{b}_i^{(2, l)}} \mathcal{F}_{\mathrm{pragmatic}}^{\mathrm{Sylvester}}
$$

Synaptogenesis of graph topology (timescale $\epsilon_2$):

$$
E_{ji}^{(l+1)}(t + \Delta t) = \mathrm{Softclip}_{[E_{\min}, 1.0]}\left( E_{ji}^{(l+1)}(t) + \Delta t \left( \eta_E \frac{1}{\max\left(\delta_{\mathrm{eps}}, -\left\langle \boldsymbol{\mu}_i^{(l)}(t), \boldsymbol{\mu}_j^{(l+1)}(t) \right\rangle_{\mathcal{L}}\right)} - \lambda_E E_{ji}^{(l+1)}(t) \right) \right)
$$

---

### Section 2: Maupertuis-Jacobi Conformal Differential Geometry
Conformal deformation of the pseudo-Euclidean metric by the pragmatic free energy target potential:

$$
\tilde{g}_{AB}(\boldsymbol{\mu}) = \Omega^2(\boldsymbol{\mu}) (\boldsymbol{\eta}_{\mathcal{L}})_{AB}, \quad \tilde{g}^{AB}(\boldsymbol{\mu}) = \Omega^{-2}(\boldsymbol{\mu}) (\boldsymbol{\eta}_{\mathcal{L}})^{AB}
$$

$$
\Omega(\boldsymbol{\mu}) = \max\left( \delta_{\mathrm{eps}}, \mathcal{F}_{\mathrm{threshold}} - \mathcal{F}_{\mathrm{pragmatic}}^{\mathrm{Sylvester}}(\boldsymbol{\mu}) \right)
$$

Levi-Civita connection transformation on the conformal hyperboloid:

$$
\tilde{\Gamma}_{BC}^A = \Gamma_{BC}^A + \delta_B^A \partial_C \ln \Omega(\boldsymbol{\mu}) + \delta_C^A \partial_B \ln \Omega(\boldsymbol{\mu}) - (\boldsymbol{\eta}_{\mathcal{L}})_{BC} (\boldsymbol{\eta}_{\mathcal{L}})^{AK} \partial_K \ln \Omega(\boldsymbol{\mu})
$$

Closed-form fictitious conformal drift force in local orthonormal tetrad coordinates:

$$
\tilde{\mathbf{v}}_{\mathrm{conformal}, i}^{(l)}(t) = -\|\mathbf{w}_i^{(l)}(t)\|_2^2 \cdot E\left(\boldsymbol{\mu}_i^{(l)}\right)^\top \boldsymbol{\eta}_{\mathcal{L}} \nabla_{\mathcal{H}^d} \ln \Omega\left(\boldsymbol{\mu}_i^{(l)}\right) + 2 \left( \left(\mathbf{w}_i^{(l)}(t)\right)^\top E\left(\boldsymbol{\mu}_i^{(l)}\right)^\top \boldsymbol{\eta}_{\mathcal{L}} \nabla_{\mathcal{H}^d} \ln \Omega\left(\boldsymbol{\mu}_i^{(l)}\right) \right) \mathbf{w}_i^{(l)}(t)
$$

---

### Section 3: Symplectic Implicit Midpoint Integrator on Cotangent Bundle
The phase point $\mathbf{\Gamma}_i^{(l)} = (\boldsymbol{\mu}_i^{(l)}, \mathbf{p}_i^{(l)}) \in T^*\mathcal{H}^d$ obeys strict algebraic constraints:

$$
\sigma_1(\boldsymbol{\mu}) = \langle \boldsymbol{\mu}, \boldsymbol{\mu} \rangle_{\mathcal{L}} + 1.0 \equiv 0, \qquad \sigma_2(\boldsymbol{\mu}, \mathbf{p}) = \langle \boldsymbol{\mu}, \mathbf{p} \rangle_{\mathcal{L}} \equiv 0
$$

The midpoint symplectic integrator evaluates forces at $\boldsymbol{\mu}_{t+1/2} = \frac{\boldsymbol{\mu}_t + \boldsymbol{\mu}_{t+\Delta t}}{2}$ and $\mathbf{p}_{t+1/2} = \frac{\mathbf{p}_t + \mathbf{p}_{t+\Delta t}}{2}$:

$$
\boldsymbol{\mu}_{i, t+\frac{\Delta t}{2}}^{(l)} = \frac{\boldsymbol{\mu}_{i, t}^{(l)} + \boldsymbol{\mu}_{i, t+\Delta t}^{(l)}}{2}, \qquad \mathbf{p}_{i, t+\frac{\Delta t}{2}}^{(l)} = \frac{\mathbf{p}_{i, t}^{(l)} + \mathbf{p}_{i, t+\Delta t}^{(l)}}{2}
$$

$$
\mathbf{p}_{i, t+\Delta t}^{(l)} = \mathbf{p}_{i, t}^{(l)} - \Delta t \left( \nabla_{\mathcal{H}^d} \mathcal{F}_{\mathrm{pragmatic}}^{\mathrm{Sylvester}}\left(\boldsymbol{\mu}_{i, t+\frac{\Delta t}{2}}^{(l)}\right) + \lambda_{\mathrm{mid}} \cdot \boldsymbol{\eta}_{\mathcal{L}} \boldsymbol{\mu}_{i, t+\frac{\Delta t}{2}}^{(l)} \right)
$$

$$
\boldsymbol{\mu}_{i, t+\Delta t}^{(l)} = \boldsymbol{\mu}_{i, t}^{(l)} + \Delta t \cdot \boldsymbol{\eta}_{\mathcal{L}}^{-1} \mathbf{p}_{i, t+\frac{\Delta t}{2}}^{(l)}
$$

Constraint multipliers $\lambda_{\mathrm{mid}}$ and $\mu_{\mathrm{mid}}$ are determined uniquely such that:

$$
\left\langle \boldsymbol{\mu}_{i, t+\Delta t}^{(l)}, \boldsymbol{\mu}_{i, t+\Delta t}^{(l)} \right\rangle_{\mathcal{L}} \equiv -1.000000000000
$$

$$
\left\langle \boldsymbol{\mu}_{i, t+\Delta t}^{(l)}, \mathbf{p}_{i, t+\Delta t}^{(l)} \right\rangle_{\mathcal{L}} \equiv 0.000000000000
$$

Preservation of the canonical symplectic 2-form:

$$
\omega_{t+\Delta t} = \sum_{k=1}^d \mathrm{d}p_k(t+\Delta t) \wedge \mathrm{d}\mu^k(t+\Delta t) \equiv \sum_{k=1}^d \mathrm{d}p_k(t) \wedge \mathrm{d}\mu^k(t) = \omega_t
$$

---

### Section 4: Linearized Cole-Hopf Active Inference (Log-Diffusion HJB)
Nonlinear Hamilton-Jacobi-Bellman stochastic dynamic programming equation:

$$
\frac{\partial V(\boldsymbol{\mu})}{\partial t} + \min_{\mathbf{w}} \left[ \langle \nabla_{\mathcal{H}^d} V(\boldsymbol{\mu}), \mathbf{w} \rangle_{\mathcal{L}} + \frac{1}{2 \lambda} \|\mathbf{w}\|_2^2 \right] + \mathcal{F}_{\mathrm{pragmatic}}(\boldsymbol{\mu}) + \frac{\lambda \sigma^2}{2} \Delta_{\mathcal{H}^d} V(\boldsymbol{\mu}) = 0
$$

Logarithmic Cole-Hopf transformation:

$$
\psi(\boldsymbol{\mu}, t) = \exp\left( -\frac{V(\boldsymbol{\mu}, t)}{\lambda \sigma^2} \right) \Longleftrightarrow V(\boldsymbol{\mu}, t) = -\lambda \sigma^2 \ln \psi(\boldsymbol{\mu}, t)
$$

Linear parabolic Schrödinger-Beltrami value diffusion:

$$
\frac{\partial \psi(\boldsymbol{\mu}, t)}{\partial t} = \frac{\lambda \sigma^2}{2} \Delta_{\mathcal{H}^d} \psi(\boldsymbol{\mu}, t) - \frac{\mathcal{F}_{\mathrm{pragmatic}}(\boldsymbol{\mu})}{\lambda \sigma^2} \psi(\boldsymbol{\mu}, t)
$$

where the radial-spherical Laplace-Beltrami operator on $\mathcal{H}^d$ evaluates to:

$$
\Delta_{\mathcal{H}^d} \psi = \frac{1}{\sinh^{d-1}(r)} \frac{\partial}{\partial r} \left( \sinh^{d-1}(r) \frac{\partial \psi}{\partial r} \right) + \frac{1}{\sinh^2(r)} \Delta_{\mathbb{S}^{d-2}} \psi
$$

Linear goal superposition and analytical geodesic velocity optimum:

$$
\psi_{\mathrm{total}}(\boldsymbol{\mu}) = \sum_{k=1}^K w_k \psi_k(\boldsymbol{\mu})
$$

$$
\mathbf{w}_{\mathrm{optimal}}(\boldsymbol{\mu}) = \lambda \sigma^2 \cdot E(\boldsymbol{\mu})^\top \boldsymbol{\eta}_{\mathcal{L}} \frac{\nabla_{\mathcal{H}^d} \psi_{\mathrm{total}}(\boldsymbol{\mu})}{\psi_{\mathrm{total}}(\boldsymbol{\mu})} = \lambda \sigma^2 \cdot E(\boldsymbol{\mu})^\top \boldsymbol{\eta}_{\mathcal{L}} \nabla_{\mathcal{H}^d} \ln \psi_{\mathrm{total}}(\boldsymbol{\mu})
$$

---

### Section 5: Viscous Eikonal Wavefronts & Godunov Fast Sweeping on Graphs
Continuous Eikonal equation with Laplace-Beltrami viscosity regularization:

$$
\left\| \nabla_{\mathcal{H}^d} S(\boldsymbol{\mu}) \right\|_{\mathcal{L}}^2 - \varepsilon_{\mathrm{visc}} \Delta_{\mathcal{H}^d} S(\boldsymbol{\mu}) = n^2(\boldsymbol{\mu}) = \frac{1}{\Omega^2(\boldsymbol{\mu})}
$$

Discrete Godunov upwind scheme on the neural connectivity graph $\mathcal{N}(i)$:

$$
\sum_{k=1}^m \frac{\left( S_i - S_{(k)} \right)^2}{d_{\mathcal{H}^d}^2\left( \boldsymbol{\mu}_i^{(l)}, \boldsymbol{\mu}_{(k)}^{(l)} \right)} - \varepsilon_{\mathrm{visc}} \left( \Delta_{\mathcal{H}^d} S \right)_i = \frac{1}{\Omega^2\left(\boldsymbol{\mu}_i^{(l)}\right)}, \quad S_{(1)} \le S_{(2)} \le \dots \le S_{(m)} < S_i
$$

Analytical root update via the Fast Sweeping method:

$$
A_m = \sum_{k=1}^m \frac{1}{d_k^2}, \quad B_m = \sum_{k=1}^m \frac{S_{(k)}}{d_k^2}, \quad C_m = \sum_{k=1}^m \frac{S_{(k)}^2}{d_k^2}
$$

$$
S_i = \frac{B_m + \sqrt{\max\left( 0.0, B_m^2 - A_m \left( C_m - \frac{1}{\Omega_i^2} \right) \right)}}{A_m}
$$

Optical velocity vector:

$$
\mathbf{w}_{\mathrm{geodesic}, i}^{(l)}(t) = -v_{\max} \cdot \frac{E\left(\boldsymbol{\mu}_i^{(l)}\right)^\top \boldsymbol{\eta}_{\mathcal{L}} \nabla_{\mathcal{H}^d} S\left(\boldsymbol{\mu}_i^{(l)}\right)}{\max\left(\delta_{\mathrm{eps}}, \left\| \nabla_{\mathcal{H}^d} S\left(\boldsymbol{\mu}_i^{(l)}\right) \right\|_{\mathcal{L}}\right)}
$$

---

### Section 6: Polyhedral Solomonoff-AIXI Decider & Safe Horizon
Decision hyperplanes partitioning the observation distribution simplex $\Delta^{|\mathcal{O}|-1}$:

$$
\mathcal{H}_{a, a'} = \left\{ \mathbf{p} \in \Delta^{|\mathcal{O}|-1} \;\middle|\; \sum_{o \in \mathcal{O}} p(o) \big( Q(h, a, o) - Q(h, a', o) \big) = 0 \right\}
$$

Kraft inequality tail bound under Levin program interleaving:

$$
\Omega_{\mathrm{unresolved}}(k) = 1.0 - \sum_{p \in \mathrm{Halted}(k)} 2^{-l(p)}
$$

$$
Q_{\min}(a) = \sum_{p \in \mathrm{Halted}(k)} \frac{2^{-l(p)}}{M_k} R(p, a), \quad Q_{\max}(a) = Q_{\min}(a) + \Omega_{\mathrm{unresolved}}(k) \cdot \frac{r_{\max}}{1 - \gamma}
$$

Strict early-stopping decision boundary:

$$
Q_{\min}(a^*) > \max_{a \neq a^*} Q_{\max}(a) \implies a_{\mathrm{computable}}^* \equiv a_{\mathrm{AIXI}}^*
$$

Polyhedral safety radius and continuous motor pressure vector:

$$
\varepsilon_{\mathrm{safe}} = \frac{Q_{\min}(a^*) - \max_{a \neq a^*} Q_{\max}(a)}{\sqrt{\sum_{o \in \mathcal{O}} \left( Q(a^*, o) - \max_{a \neq a^*} Q(a, o) \right)^2}} > 0
$$

$$
\mathbf{m}_{\mathrm{poly}}(a^*) = \varepsilon_{\mathrm{safe}} \cdot \frac{\sum_{a \neq a^*} \frac{\mathbf{e}_{a^*} - \mathbf{e}_a}{Q_{\min}(a^*) - Q_{\max}(a) + \delta_{\mathrm{eps}}}}{\left\| \sum_{a \neq a^*} \frac{\mathbf{e}_{a^*} - \mathbf{e}_a}{Q_{\min}(a^*) - Q_{\max}(a) + \delta_{\mathrm{eps}}} \right\|_2} \in \mathbb{R}^{d_{\mathrm{act}}}
$$

---

### Section 7: Full Microstate Supervector & Expected Free Energy Motor Loop
Full microstate configuration vector:

$$
\mathbf{\Xi}_i^{(l)}(t) = \begin{pmatrix} \boldsymbol{\mu}_i^{(l)}(t) \\ \mathbf{p}_i^{(l)}(t) \\ \mathbf{\Theta}_i^{(l)}(t) \\ \pi_i^{(l)}(t) \\ E_{ji}^{(l+1)}(t) \\ \mathbf{A}_{\mathrm{SCM}}(t) \end{pmatrix} \in \mathcal{M}_i^{(l)}, \quad \mathbf{\Theta}_i^{(l)}(t) = \left\{ \mathbf{W}_i^{(1, l)}, \mathbf{W}_i^{(2, l)}, \mathbf{b}_i^{(1, l)}, \mathbf{b}_i^{(2, l)} \right\}
$$

Motor actuation update via Expected Free Energy (EFE) gradient descent:

$$
\frac{\mathrm{d} \mathbf{a}}{\mathrm{d} t} = -\eta_a \nabla_{\mathbf{a}} \left[ \mathbb{D}_{\mathrm{KL}}\left( Q(s_{t+\Delta t} \mid \mathbf{a}(t)) \parallel P^*(s_{\mathrm{target}}) \right) - \lambda_{\mathrm{Pearl}}(t) \mathcal{I}_{\mathrm{Pearl}}^{\mathrm{causal}}\big( \mathbf{do}(\mathbf{a}(t)) \big) + \frac{\lambda_{\mathrm{policy}}}{2} \|\mathbf{a}(t) - \mathbf{a}_{\mathrm{target}}(t)\|^2 \right]
$$

$$
\mathbf{a}(t + \Delta t) = \mathrm{Softclip}_{[-1, 1]}\left( \mathbf{a}(t) + \Delta t \cdot \frac{\mathrm{d} \mathbf{a}}{\mathrm{d} t} \right) \in [-1, 1]^{d_{\mathrm{act}}}
$$

$$
\mathbf{a}_{\mathrm{target}}(t) = \tanh\left( \mathbf{W}_{\mathrm{motor}} \cdot \mathbf{w}_{\mathrm{geodesic}}^{(L)}(t) + \mathbf{W}_{\mathrm{poly}} \cdot \mathbf{m}_{\mathrm{poly}}(a^*) + \mathbf{W}_{\mathrm{social}} \cdot \mathbf{a}_{\mathrm{ToM}}^*(t) + \mathbf{b}_{\mathrm{motor}} \right)
$$

---

### Section 8: Master Augmented Lagrangian (PHR) Action Functional

$$
\begin{aligned}
\mathcal{J}_{\mathrm{AGI\_Master}}\big[ \mathbf{\Xi}(t), \mathbf{a}(t) \big] &= \int_{0}^T \Bigg[ \mathcal{F}_{\mathrm{pragmatic}}^{\mathrm{Sylvester}}\big( \mathbf{\Xi} \big) + \mathcal{K}_{\mathrm{VIB}}\big( \mathbf{\Xi}, \mathbf{a} \big) + \lambda_{\mathrm{goal}}(t) \mathbb{D}_{\mathrm{KL}}\left( Q(s \mid \mathbf{a}) \parallel P^*(s) \right) - \lambda_{\mathrm{Pearl}}(t) \mathcal{I}_{\mathrm{Pearl}}^{\mathrm{causal}}\big( \mathbf{do}(\mathbf{a}) \big) \\
&\quad - \mu_{\mathrm{ground}}(t) \mathcal{I}_{\mathrm{grounding}} + \gamma_{\mathrm{G\ddot{o}del}}(t) \left( \frac{d_{\mathcal{H}^d}\left( \boldsymbol{\mu}_{\mathrm{self}}^*(t), \overline{\mathcal{T}}_{\mathrm{introspect}}(t) \right)}{R_{\mathcal{H}}} \right)^2 + \mathcal{L}_{\mathrm{kin}} + \mathcal{L}_{\mathrm{Lip}} + \mathcal{L}_{\mathrm{Eikonal}}
&\quad + \mathcal{L}_{\mathrm{Landauer}} + \sum_{k=1}^2 \left( \frac{\rho_k}{2} \max\left( 0.0, \mathcal{C}_k(\mathbf{\Xi}) - C_k^* + \frac{\lambda_k(t)}{\rho_k} \right)^2 - \frac{\lambda_k(t)^2}{2 \rho_k} \right) \Bigg] \mathrm{d}t
\end{aligned}
$$

Eikonal residual penalty:

$$
\mathcal{L}_{\mathrm{Eikonal}} = \lambda_{\mathrm{Eik}} \sum_{l=1}^L \sum_{i \in V^{(l)}} \left( \left\| \nabla_{\mathcal{H}^d} S\left(\boldsymbol{\mu}_i^{(l)}\right) \right\|_{\mathcal{L}}^2 - \varepsilon_{\mathrm{visc}} \Delta_{\mathcal{H}^d} S\left(\boldsymbol{\mu}_i^{(l)}\right) - n^2\left(\boldsymbol{\mu}_i^{(l)}\right) \right)^2
$$

Landauer information erasure dissipation penalty:

$$
\mathcal{L}_{\mathrm{Landauer}} = k_B \ln(2) \cdot \mathbf{T}_i \cdot \max\left(0, -\dot{u}_k\right) \cdot \mathbb{I}\left(u_k < \theta_{\mathrm{prune}}\right)
$$

Kinetic limits and Moreau-Yosida multiplier update:

$$
\mathcal{L}_{\mathrm{kin}} = \sum_{l=1}^L \sum_{i \in V^{(l)}} \frac{\kappa_v}{2} \|\mathbf{w}_i^{(l)}(t)\|_2^2
$$

$$
\mathcal{C}_1(\mathbf{\Xi}) = \max_{l \in \{1,\dots,L\}} \max_{i \in V^{(l)}} \max\left( \sigma_{\max}\left(\mathbf{W}_i^{(1, l)}\right), \sigma_{\max}\left(\mathbf{W}_i^{(2, l)}\right) \right) \le C_1^* = 1.0
$$

$$
\mathcal{C}_2(\mathbf{\Xi}) = \frac{1}{\sum_{l=1}^L |V^{(l)}|} \sum_{l=1}^L \sum_{i \in V^{(l)}} \|\mathbf{w}_i^{(l)}(t)\|_2^2 \le C_2^* = v_{\mathrm{budget}}^2
$$

$$
\lambda_k(t + \Delta t) = \lambda_k(t) + \frac{\Delta t \cdot \eta_\lambda}{1 + \gamma_{\mathrm{prox}} \Delta t} \cdot \max\left( -\frac{\lambda_k(t)}{\rho_k}, \mathcal{C}_k(\mathbf{\Xi}(t)) - C_k^* \right)
$$

---

### Section 9: Sylvester Pragmatic Free Energy

$$
\begin{aligned}
\mathcal{F}_{\mathrm{pragmatic}}^{\mathrm{Sylvester}}\big( \mathbf{\Xi}(t) \big) &= \sum_{l=0}^L \sum_{i \in V^{(l)}} \Bigg[ \frac{\pi_i^{(l)}(t)}{2} \left( \tilde{\boldsymbol{\varepsilon}}_i^{(l)}(t) \right)^\top \tilde{\mathbf{\Pi}}_i^{(l)}(t) \left( \tilde{\boldsymbol{\varepsilon}}_i^{(l)}(t) \right) - \frac{d-1}{2} \ln \pi_i^{(l)}(t) \\
&\quad + \frac{1}{2} \left( \sum_{k=1}^{d-1} \ln \tilde{d}_{i, k}^{(l)} + \sum_{p=1}^r \ln s_{i, p}^{(l)} + \ln \det\left( \mathrm{diag}\left(\mathbf{s}_i^{(l)}\right)^{-1} + \left( \tilde{\mathbf{V}}_i^{(l)} \right)^\top \mathrm{diag}\left(\tilde{\mathbf{d}}_i^{(l)}\right)^{-1} \tilde{\mathbf{V}}_i^{(l)} \right) \right) \\
&\quad + \frac{1}{2 M_{\mathrm{trace}}} \sum_{m=1}^{M_{\mathrm{trace}}} \mathbf{u}_m^\top \left( \tilde{\mathbf{\Pi}}_i^{(l)} - \mathbf{I}_{d-1} \right) \mathbf{u}_m \Bigg]
\end{aligned}
$$

$$
\mathbf{u}_m \sim \{-1, +1\}^{d-1}
$$

---

### Section 10: Retraction Perceptron & Tangent-Space SoftVQ
Parent context assembly:

$$
\tilde{\mathbf{a}}_i^{(l)}(t) = \begin{cases} \sum_{j \in \mathrm{parents}(i)} E_{ji}^{(l+1)} \mathrm{SoftVQ}_{\mathrm{tangent}}\left( \boldsymbol{\mu}_j^{(l+1)}(t) \right), & 0 \le l < L \\ E\left(\overline{\boldsymbol{\mu}}^{(L)}(t)\right)^\top \boldsymbol{\eta}_{\mathcal{L}} \mathrm{Log}_{\overline{\boldsymbol{\mu}}^{(L)}(t)}^{\mathcal{H}^d}\left( \overline{\mathcal{T}}_{\mathrm{introspect}}(t) \right), & l = L \end{cases} \in \mathbb{R}^{d-1}
$$

Two-layer prediction perceptron with tangent Euclidean retraction ($R_{\mathrm{pred\_max}} = 3.0$):

$$
\mathbf{h}_i^{(l)}(t) = \sigma\left( \mathbf{W}_i^{(1, l)} \tilde{\mathbf{a}}_i^{(l)}(t) + \mathbf{b}_i^{(1, l)} \right) \in \mathbb{R}^{h_{\mathrm{dim}}}
$$

$$
\mathbf{u}_{2, i}^{(l)}(t) = \mathbf{W}_i^{(2, l)} \mathbf{h}_i^{(l)}(t) + \mathbf{b}_i^{(2, l)} \in \mathbb{R}^{d-1}
$$

$$
\mathbf{u}_{2, \mathrm{safe}, i}^{(l)}(t) = \mathbf{u}_{2, i}^{(l)}(t) \cdot \min\left(1.0, \frac{R_{\mathrm{pred\_max}}}{\|\mathbf{u}_{2, i}^{(l)}(t)\|_2}\right), \quad R_{\mathrm{pred\_max}} = 3.0
$$

$$
\mathbf{v}_{\mathrm{pred}, i}^{(l)}(t) = E\left(\overline{\boldsymbol{\mu}}^{(l)}(t)\right) \cdot \mathbf{u}_{2, \mathrm{safe}, i}^{(l)}(t) \in T_{\overline{\boldsymbol{\mu}}^{(l)}}\mathcal{H}^d, \quad \|\mathbf{v}_{\mathrm{pred}, i}^{(l)}(t)\|_{\mathcal{L}} \equiv \|\mathbf{u}_{2, \mathrm{safe}, i}^{(l)}(t)\|_2 \le 3.0
$$

$$
\boldsymbol{\mu}_{\mathrm{pred}, i}^{(l)}(t) = \mathrm{Exp}_{\overline{\boldsymbol{\mu}}^{(l)}(t)}^{\mathcal{H}^d}\left( \mathbf{v}_{\mathrm{pred}, i}^{(l)}(t) \right) \in \mathcal{H}^d
$$

$$
\boldsymbol{\varepsilon}_i^{(l)}(t) = \mathrm{Log}_{\boldsymbol{\mu}_i^{(l)}(t)}^{\mathcal{H}^d}\left( \boldsymbol{\mu}_{\mathrm{pred}, i}^{(l)}(t) \right) \in T_{\boldsymbol{\mu}_i^{(l)}}\mathcal{H}^d
$$

SoftVQ tangent quantization with active token preservation:

$$
d_{\mathrm{min\_act}}^2 = \min_{m: \, m_{\mathrm{pruned}, m} = 1} d_{\mathcal{H}^d}^2\left( \boldsymbol{\mu}_j^{(l+1)}, \mathrm{Exp}_{\mathbf{o}}^{\mathcal{H}^d}(E(\mathbf{o})\mathbf{c}_m) \right)
$$

$$
w_k = \frac{m_{\mathrm{pruned}, k} \cdot \exp\left( \frac{-d_{\mathcal{H}^d}^2\left( \boldsymbol{\mu}_j^{(l+1)}, \mathrm{Exp}_{\mathbf{o}}(E(\mathbf{o})\mathbf{c}_k) \right) + d_{\mathrm{min\_act}}^2}{T_{\mathrm{vq}}} \right)}{\sum_{k'=0}^{|\mathcal{C}|} m_{\mathrm{pruned}, k'} \exp\left( \frac{-d_{\mathcal{H}^d}^2\left( \boldsymbol{\mu}_j^{(l+1)}, \mathrm{Exp}_{\mathbf{o}}(E(\mathbf{o})\mathbf{c}_{k'}) \right) + d_{\mathrm{min\_act}}^2}{T_{\mathrm{vq}}} \right) + \delta_{\mathrm{eps}}}
$$

$$
\mathrm{SoftVQ}_{\mathrm{tangent}}\left( \boldsymbol{\mu}_j^{(l+1)} \right) = \sum_{k=0}^{|\mathcal{C}|} w_k \mathbf{c}_k \in \mathbb{R}^{d-1}, \quad \mathbf{c}_0 = \mathbf{0}, \quad m_{\mathrm{pruned}, 0} \equiv 1
$$

---

### Section 11: Sigmoidal Spike Activation Masking

$$
\sigma_\beta\left( \left\| \tilde{\boldsymbol{\varepsilon}}_i^{(l)} \right\|_{\tilde{\mathbf{\Pi}}_i^{(l)}}^2 - \tau_i^{(l)} \right) = \frac{1}{1 + \exp\left( -\beta \left( \left( \tilde{\boldsymbol{\varepsilon}}_i^{(l)} \right)^\top \tilde{\mathbf{\Pi}}_i^{(l)} \left( \tilde{\boldsymbol{\varepsilon}}_i^{(l)} \right) - \tau_i^{(l)} \right) \right)}
$$

---

### Section 12: DLR Factorization with Oja-QR & K-FAC Natural Gradient
Streaming Oja subspace tracking:

$$
\mathbf{y}_i^{(l)}(t) = \left( \tilde{\mathbf{V}}_i^{(l)}(t) \right)^\top \tilde{\boldsymbol{\varepsilon}}_i^{(l)}(t) \in \mathbb{R}^{r}
$$

$$
\mathbf{s}_i^{(l)}(t + \Delta t) = (1 - \eta_m) \mathbf{s}_i^{(l)}(t) + \eta_m \left( \mathbf{y}_i^{(l)}(t) \odot \mathbf{y}_i^{(l)}(t) + \delta_{\min} \mathbf{1} \right) \in \mathbb{R}^r
$$

$$
\tilde{\mathbf{V}}_{\mathrm{raw}, i}^{(l)} = \tilde{\mathbf{V}}_i^{(l)}(t) + \eta_m \left[ \tilde{\boldsymbol{\varepsilon}}_i^{(l)}(t) \left( \mathbf{y}_i^{(l)}(t) \right)^\top - \tilde{\mathbf{V}}_i^{(l)}(t) \mathbf{y}_i^{(l)}(t) \left( \mathbf{y}_i^{(l)}(t) \right)^\top \right]
$$

$$
\tilde{\mathbf{V}}_i^{(l)}(t + \Delta t) = \mathrm{QR\_Q}\left( \tilde{\mathbf{V}}_{\mathrm{raw}, i}^{(l)} \right) \in \mathbb{R}^{(d-1) \times r}
$$

Residual variance tracking:

$$
\tilde{\boldsymbol{\varepsilon}}_{\mathrm{residual}, i}^{(l)}(t) = \tilde{\boldsymbol{\varepsilon}}_i^{(l)}(t) - \tilde{\mathbf{V}}_i^{(l)}(t + \Delta t) \left( \left( \tilde{\mathbf{V}}_i^{(l)}(t + \Delta t) \right)^\top \tilde{\boldsymbol{\varepsilon}}_i^{(l)}(t) \right) \in \mathbb{R}^{d-1}
$$

$$
\tilde{\mathbf{d}}_i^{(l)}(t + \Delta t) = (1 - \eta_m) \tilde{\mathbf{d}}_i^{(l)}(t) + \eta_m \left( \tilde{\boldsymbol{\varepsilon}}_{\mathrm{residual}, i}^{(l)}(t) \odot \tilde{\boldsymbol{\varepsilon}}_{\mathrm{residual}, i}^{(l)}(t) + \delta_{\min} \mathbf{1} \right)
$$

Sensitivity vectors under isometric Lorentz parallel transport:

$$
\mathbf{g}_i^{(2, \mathrm{local})} = +\pi_i^{(l)}(t) \cdot \tilde{\mathbf{\Pi}}_i^{(l)}(t) \tilde{\boldsymbol{\varepsilon}}_i^{(l)}(t) \in \mathbb{R}^{d-1}
$$

$$
\mathbf{g}_i^{(2, l)} = E\left(\overline{\boldsymbol{\mu}}^{(l)}(t)\right)^\top \boldsymbol{\eta}_{\mathcal{L}} \mathbf{P}_{\boldsymbol{\mu}_i^{(l)}(t) \to \overline{\boldsymbol{\mu}}^{(l)}(t)}\left( E\left(\boldsymbol{\mu}_i^{(l)}(t)\right) \mathbf{g}_i^{(2, \mathrm{local})} \right) \in \mathbb{R}^{d-1}
$$

$$
\mathbf{g}_i^{(1, l)} = \left( \left(\mathbf{W}_i^{(2, l)}\right)^\top \mathbf{g}_i^{(2, l)} \right) \odot \sigma'\left( \mathbf{W}_i^{(1, l)} \tilde{\mathbf{a}}_i^{(l)} + \mathbf{b}_i^{(1, l)} \right) \in \mathbb{R}^{h_{\mathrm{dim}}}
$$

Kronecker covariance factors with identity initialization $\tilde{\mathbf{A}}(0) = \mathbf{I}$, $\tilde{\mathbf{B}}(0) = \mathbf{I}$:

$$
\tilde{\mathbf{A}}_i^{(1, l)}(t + \Delta t) = (1 - \eta_k) \tilde{\mathbf{A}}_i^{(1, l)}(t) + \eta_k \left( \frac{\tilde{\mathbf{a}}_i^{(l)} \left( \tilde{\mathbf{a}}_i^{(l)} \right)^\top}{1.0 + \|\tilde{\mathbf{a}}_i^{(l)}\|^2} \right)
$$

$$
\tilde{\mathbf{B}}_i^{(1, l)}(t + \Delta t) = (1 - \eta_k) \tilde{\mathbf{B}}_i^{(1, l)}(t) + \eta_k \left( \frac{\mathbf{g}_i^{(1, l)} \left( \mathbf{g}_i^{(1, l)} \right)^\top}{1.0 + \|\mathbf{g}_i^{(1, l)}\|^2} \right)
$$

$$
\tilde{\mathbf{A}}_i^{(2, l)}(t + \Delta t) = (1 - \eta_k) \tilde{\mathbf{A}}_i^{(2, l)}(t) + \eta_k \left( \frac{\mathbf{h}_i^{(l)} \left( \mathbf{h}_i^{(l)} \right)^\top}{1.0 + \|\mathbf{h}_i^{(l)}\|^2} \right)
$$

$$
\tilde{\mathbf{B}}_i^{(2, l)}(t + \Delta t) = (1 - \eta_k) \tilde{\mathbf{B}}_i^{(2, l)}(t) + \eta_k \left( \frac{\mathbf{g}_i^{(2, l)} \left( \mathbf{g}_i^{(2, l)} \right)^\top}{1.0 + \|\mathbf{g}_i^{(2, l)}\|^2} \right)
$$

---

### Section 13: Spectral Lipschitz Regularization

$$
\mathcal{L}_{\mathrm{Lip}} = \lambda_{\mathrm{Lip}} \sum_{l=1}^L \sum_{i \in V^{(l)}} \left[ \max\left( 0.0, \sigma_{\max}\left( \mathbf{W}_i^{(1, l)} \right) - 1.0 \right)^2 + \max\left( 0.0, \sigma_{\max}\left( \mathbf{W}_i^{(2, l)} \right) - 1.0 \right)^2 \right]
$$

---

### Section 14: Minkowski Metric & Geodesic Operators

$$
\langle \mathbf{x}, \mathbf{y} \rangle_{\mathcal{L}} = -x_0 y_0 + \sum_{k=1}^{d-1} x_k y_k, \quad \boldsymbol{\eta}_{\mathcal{L}} = \mathrm{diag}(-1, 1, \dots, 1)
$$

$$
\|\mathbf{v}\|_{\mathcal{L}} = \sqrt{\max\left( 0.0, \langle \mathbf{v}, \mathbf{v} \rangle_{\mathcal{L}} \right)}
$$

$$
d_{\mathcal{H}^d}(\mathbf{x}, \mathbf{y}) = 2 \mathrm{arcsinh}\left( \frac{1}{2} \sqrt{\max\left( 0.0, \langle \mathbf{x} - \mathbf{y}, \mathbf{x} - \mathbf{y} \rangle_{\mathcal{L}} \right)} \right)
$$

$$
\mathrm{Exp}_{\boldsymbol{\mu}}^{\mathcal{H}^d}(\mathbf{v}) = \begin{cases} \left( 1 + \frac{1}{2}\|\mathbf{v}\|_{\mathcal{L}}^2 \right) \boldsymbol{\mu} + \left( 1 + \frac{1}{6}\|\mathbf{v}\|_{\mathcal{L}}^2 \right) \mathbf{v}, & \|\mathbf{v}\|_{\mathcal{L}} < 10^{-4} \\ \cosh\left( \|\mathbf{v}\|_{\mathcal{L}} \right) \boldsymbol{\mu} + \sinh\left( \|\mathbf{v}\|_{\mathcal{L}} \right) \frac{\mathbf{v}}{\|\mathbf{v}\|_{\mathcal{L}}}, & \|\mathbf{v}\|_{\mathcal{L}} \ge 10^{-4} \end{cases}
$$

$$
\mathbf{u} = \mathbf{y} + \langle \boldsymbol{\mu}, \mathbf{y} \rangle_{\mathcal{L}} \boldsymbol{\mu}, \quad \mathrm{Log}_{\boldsymbol{\mu}}^{\mathcal{H}^d}(\mathbf{y}) = \begin{cases} \left( 1 - \frac{\|\mathbf{u}\|_{\mathcal{L}}^2}{6} + \frac{3 \|\mathbf{u}\|_{\mathcal{L}}^4}{40} \right) \mathbf{u}, & \|\mathbf{u}\|_{\mathcal{L}} < 10^{-4} \\ \frac{d_{\mathcal{H}^d}(\boldsymbol{\mu}, \mathbf{y})}{\|\mathbf{u}\|_{\mathcal{L}}} \cdot \mathbf{u}, & \|\mathbf{u}\|_{\mathcal{L}} \ge 10^{-4} \end{cases}
$$

$$
\mathbf{P}_{\mathbf{x} \to \mathbf{y}}(\mathbf{v}) = \mathbf{v} + \frac{\langle \mathbf{v}, \mathbf{y} \rangle_{\mathcal{L}}}{1 - \langle \mathbf{x}, \mathbf{y} \rangle_{\mathcal{L}}} (\mathbf{x} + \mathbf{y})
$$

---

### Section 15: Variational Information Bottleneck (VIB)

$$
\mathcal{K}_{\mathrm{VIB}}\big( \mathbf{\Xi}(t), \mathbf{a}(t) \big) = \sum_{l=0}^L \sum_{i \in V^{(l)}} \mathbb{D}_{\mathrm{KL}}\left( Q\left( \mathbf{\Theta}_i^{(l)} \;\middle|\; \mathcal{D}_{\mathrm{data}} \right) \;\middle\|\; P_0\left( \mathbf{\Theta}_i^{(l)} \right) \right) - \beta_{\mathrm{info}}(t) \mathcal{I}_{\mathrm{var}}\left( \boldsymbol{\mu}_i^{(l)} ; \mathbf{x}_{\mathrm{world}} \;\middle|\; \mathbf{a}(t) \right)
$$

$$
\mathcal{I}_{\mathrm{var}}\left( \boldsymbol{\mu}_i^{(l)} ; \mathbf{x}_{\mathrm{world}} \;\middle|\; \mathbf{a}(t) \right) \ge \mathbb{E}_{\mathcal{B}}\left[ f_{\psi}^{\mathrm{VIB}}\left(\mathbf{x}_{\mathrm{world}}, \boldsymbol{\mu}_i^{(l)}, \mathbf{a}(t)\right) - \ln \sum_{\mathbf{x}' \in \mathcal{B}} \exp\left( f_{\psi}^{\mathrm{VIB}}\left(\mathbf{x}', \boldsymbol{\mu}_i^{(l)}, \mathbf{a}(t)\right) \right) \right]
$$

---

### Section 16: Differentiable Causal Do-Calculus & Schur-Capped NOTEARS
Structural causal forward model:

$$
s_{t+\Delta t} = f_\phi\left( \left( (1 + \delta_{\mathrm{scm}})\mathbf{I}_{d_{\mathrm{scm}}} - \mathbf{A}_{\mathrm{SCM}}^\top \right)^{-1} g(s_t, \mathbf{a}(t)) \right) + \boldsymbol{\epsilon}_{\mathrm{scm}}
$$

Causal counterfactual mutual information:

$$
\mathcal{I}_{\mathrm{Pearl}}^{\mathrm{causal}}\big( \mathbf{do}(\mathbf{a}(t)) \big) = \mathbb{E}_{\mathcal{B}}\left[ f_\phi\big(s_{t+\Delta t}, s_t, \mathbf{do}(\mathbf{a}(t))\big) - \ln \sum_{s' \in \mathcal{B}} \exp\Big( f_\phi\big(s', s_t, \mathbf{a}_{\mathrm{counterfactual}}(t)\big) \Big) \right]
$$

$$
\mathbf{P}_{\mathbf{a}^\perp}(t) = \mathbf{I}_{d_{\mathrm{act}}} - \frac{\mathbf{a}(t) \mathbf{a}(t)^\top}{\|\mathbf{a}(t)\|^2 + \delta_{\mathrm{proj}}^2}, \quad \delta_{\mathrm{proj}} = 10^{-5}
$$

$$
\mathbf{a}_{\mathrm{counterfactual}}(t) = \mathrm{Softclip}_{[-1, 1]}\left( \mathbf{a}(t) + \sigma_{\mathrm{cf}} \mathbf{P}_{\mathbf{a}^\perp}(t) \boldsymbol{\zeta} \right), \quad \boldsymbol{\zeta} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{d_{\mathrm{act}}})
$$

Log-determinant acyclicity constraint:

$$
h_{\mathrm{log}}(\mathbf{A}_{\mathrm{SCM}}) = \ln \left( \frac{1}{d_{\mathrm{scm}}} \mathrm{Tr}\left( \mathrm{expm}(\mathbf{A}_{\mathrm{SCM}} \circ \mathbf{A}_{\mathrm{SCM}}) \right) \right) = 0
$$

$$
\nabla_{\mathbf{A}} h_{\mathrm{log}}(\mathbf{A}) = \frac{2}{\mathrm{Tr}(\mathrm{expm}(\mathbf{A}\circ\mathbf{A}))} \left( \mathrm{expm}(\mathbf{A}\circ\mathbf{A})^\top \circ \mathbf{A} \right)
$$

$$
\mathbf{A}_{\mathrm{SCM}}(t + \Delta t) = \mathrm{Clip}_{\rho(\mathbf{A}) \le 1 - \epsilon_{\mathrm{gap}}}^{\mathrm{Schur}}\left( \mathbf{A}_{\mathrm{SCM}}(t) - \frac{\Delta t}{\epsilon_1} \left[ -\nabla_{\mathbf{A}_{\mathrm{SCM}}} \mathcal{I}_{\mathrm{Pearl}}^{\mathrm{causal}} + \mu_{\mathrm{DAG}} \nabla_{\mathbf{A}_{\mathrm{SCM}}} h_{\mathrm{log}}(\mathbf{A}_{\mathrm{SCM}}) \right] \right)
$$

where the spectral radius $\rho(\mathbf{A})$ is bounded strictly via real Schur decomposition:

$$
\mathbf{A} = \mathbf{Z} \mathbf{T} \mathbf{Z}^\top, \quad \mathbf{Z}^\top \mathbf{Z} = \mathbf{I}, \quad \mathbf{T} \text{ is real quasi-triangular Schur form.}
$$

---

### Section 17: Friston Active Inference Target Divergence

$$
\mathbb{D}_{\mathrm{KL}}\left( Q\big(s_{t+\Delta t} \mid \mathbf{a}(t)\big) \;\middle\|\; P^*\big(s_{\mathrm{target}}\big) \right) = \sum_{k} Q\big(s_{t+\Delta t} = k \mid \mathbf{a}(t)\big) \ln \frac{Q\big(s_{t+\Delta t} = k \mid \mathbf{a}(t)\big)}{P^*\big(s_{\mathrm{target}} = k\big)}
$$

---

### Section 18: Multimodal Dirichlet Sensory Grounding

$$
\mathbf{v}_{\mathrm{bundle}} = E(\mathbf{o}) \cdot \pi_{\mathrm{bundle}}\left( \mathbf{W}_{\mathrm{vis}} \mathrm{Enc}_{\mathrm{vis}}(\mathbf{x}_{\mathrm{img}}) \parallel \mathbf{W}_{\mathrm{txt}} \mathrm{Enc}_{\mathrm{txt}}(\mathbf{x}_{\mathrm{text}}) \parallel \mathbf{W}_{\mathrm{aud}} \mathrm{Enc}_{\mathrm{aud}}(\mathbf{x}_{\mathrm{audio}}) \right) \in T_{\mathbf{o}}\mathcal{H}^d
$$

$$
\boldsymbol{\mu}_0^{(0)}(t + \Delta t) = \mathrm{Exp}_{\mathbf{o}}^{\mathcal{H}^d}\left( \mathbf{v}_{\mathrm{bundle}} \right)
$$

$$
\mathcal{I}_{\mathrm{grounding}}\big( \boldsymbol{\mu}^{(L)} ; \boldsymbol{\mu}_0^{(0)} \big) = \mathbb{E}_{\mathcal{B}}\left[ f_{\mathrm{ground}}\big(\boldsymbol{\mu}^{(L)}, \boldsymbol{\mu}_0^{(0)}\big) - \ln \sum_{\boldsymbol{\mu}' \in \mathcal{B}} \exp\Big( f_{\mathrm{ground}}\big(\boldsymbol{\mu}^{(L)}, \boldsymbol{\mu}'\big) \Big) \right]
$$

---

### Section 19: Gödelian Ego-Reflection via Adjoint Resolvents
Historical state macro-representation:

$$
\text{feat}_{\mathrm{hist}}(t) = \left[ \mathrm{Concat}_{l=1, i}^L\left( \mathrm{sg}\big[\boldsymbol{\mu}_i^{(l)}(t-\Delta t)\big], \pi_i^{(l)}(t-\Delta t) \right) \parallel \mathbf{a}(t-\Delta t) \right] \in \mathbb{R}^{D_{\mathrm{hist}}}
$$

Ego-resolvent fixed point:

$$
v_{\mathrm{self}}^*(t) = \left( (1 + \delta_{\mathrm{reg}})\mathbf{I}_{d-1} - \mathbf{W}_{\mathrm{self}, v}(t) \right)^{-1} \cdot \left( \mathbf{W}_{\mathrm{self}, \mathrm{hist}}(t) \cdot \text{feat}_{\mathrm{hist}}(t) + \mathbf{b}_{\mathrm{self}}(t) \right) \in \mathbb{R}^{d-1}
$$

$$
\mathbf{v}_{\mathrm{ambient}}^*(t) = E\left( \overline{\boldsymbol{\mu}}^{(L)}(t) \right) \cdot v_{\mathrm{self}}^*(t) \in T_{\overline{\boldsymbol{\mu}}^{(L)}}\mathcal{H}^d
$$

$$
\boldsymbol{\mu}_{\mathrm{self}}^*(t) = \mathrm{Exp}_{\overline{\boldsymbol{\mu}}^{(L)}(t)}^{\mathcal{H}^d}\left( \mathbf{v}_{\mathrm{ambient}}^*(t) \right) \in \mathcal{H}^d
$$

$$
\overline{\mathcal{T}}_{\mathrm{introspect}}(t) = \mathrm{Exp}_{\overline{\mathcal{T}}(t-\Delta t)}^{\mathcal{H}^d}\left( \gamma_{\mathrm{ema}} \mathrm{Log}_{\overline{\mathcal{T}}(t-\Delta t)}^{\mathcal{H}^d}\Big( \boldsymbol{\mu}_{\mathrm{self}}^*(t) \Big) \right)
$$

$$
\mathbf{g}_{\mathrm{self}} = E\left(\overline{\mathcal{T}}_{\mathrm{introspect}}\right)^\top \boldsymbol{\eta}_{\mathcal{L}} \mathrm{Log}_{\overline{\mathcal{T}}_{\mathrm{introspect}}(t)}^{\mathcal{H}^d}\left( \boldsymbol{\mu}_{\mathrm{self}}^*(t) \right) \in \mathbb{R}^{d-1}
$$

$$
\tilde{\mathbf{g}} = E\left(\overline{\boldsymbol{\mu}}^{(L)}(t)\right)^\top \boldsymbol{\eta}_{\mathcal{L}} \mathbf{P}_{\overline{\mathcal{T}}_{\mathrm{introspect}}(t) \to \overline{\boldsymbol{\mu}}^{(L)}(t)}\left( E\left(\overline{\mathcal{T}}_{\mathrm{introspect}}\right) \mathbf{g}_{\mathrm{self}} \right) \in \mathbb{R}^{d-1}
$$

$$
\mathbf{W}_{\mathrm{self}, v}(t + \Delta t) = \mathrm{Clip}_{\sigma_{\max} \le 1 - \epsilon_{\mathrm{gap}}}\left( \mathbf{W}_{\mathrm{self}, v}(t) - \Delta t \eta_{\mathrm{self}} \left( (1 + \delta_{\mathrm{reg}})\mathbf{I}_{d-1} - \mathbf{W}_{\mathrm{self}, v}(t)^\top \right)^{-1} \tilde{\mathbf{g}} \cdot \left( v_{\mathrm{self}}^*(t) \right)^\top \right)
$$

$$
\mathbf{W}_{\mathrm{self}, \mathrm{hist}}(t + \Delta t) = \mathbf{W}_{\mathrm{self}, \mathrm{hist}}(t) - \Delta t \eta_{\mathrm{self}} \left( (1 + \delta_{\mathrm{reg}})\mathbf{I}_{d-1} - \mathbf{W}_{\mathrm{self}, v}(t)^\top \right)^{-1} \tilde{\mathbf{g}} \cdot \left( \text{feat}_{\mathrm{hist}}(t) \right)^\top
$$

$$
\mathbf{b}_{\mathrm{self}}(t + \Delta t) = \mathbf{b}_{\mathrm{self}}(t) - \Delta t \eta_{\mathrm{self}} \left( (1 + \delta_{\mathrm{reg}})\mathbf{I}_{d-1} - \mathbf{W}_{\mathrm{self}, v}(t)^\top \right)^{-1} \tilde{\mathbf{g}}
$$

---

### Section 20: Sleep Replay Consolidation via Vitter Stratified Reservoirs

$$
\mathcal{M}_{\mathrm{fast}} = \bigcup_{k=1}^K \mathcal{M}_k, \quad |\mathcal{M}_k| \le \left\lfloor \frac{M_{\mathrm{capacity}}}{K} \right\rfloor
$$

$$
\mathbb{E}_{\mathbf{m} \sim \mathcal{M}_{\mathrm{fast}}} [\nabla \mathcal{L}_{\mathrm{recon}}] = \frac{1}{K} \sum_{k=1}^K \mathbb{E}_{\mathbf{m} \sim \mathcal{M}_k} [\nabla \mathcal{L}_{\mathrm{recon}}(\mathbf{m})]
$$

$$
\left( \frac{\partial \mathbf{W}_i^{(1, l)}}{\partial t} \right)_{\mathrm{replay}} = -\eta_{\mathrm{replay}} \cdot \frac{1}{K} \sum_{k=1}^K \mathbb{E}_{\mathbf{m} \sim \mathcal{M}_k}\left[ \left( \tilde{\mathbf{B}}_i^{(1, l)} + \delta_{\mathrm{kfac}}\mathbf{I} \right)^{-1} \nabla_{\mathbf{W}_i^{(1, l)}} \mathcal{L}_{\mathrm{recon}}\big( \mathbf{m} \big) \left( \tilde{\mathbf{A}}_i^{(1, l)} + \delta_{\mathrm{kfac}}\mathbf{I} \right)^{-1} \right]
$$

$$
\left( \frac{\partial \mathbf{W}_i^{(2, l)}}{\partial t} \right)_{\mathrm{replay}} = -\eta_{\mathrm{replay}} \cdot \frac{1}{K} \sum_{k=1}^K \mathbb{E}_{\mathbf{m} \sim \mathcal{M}_k}\left[ \left( \tilde{\mathbf{B}}_i^{(2, l)} + \delta_{\mathrm{kfac}}\mathbf{I} \right)^{-1} \nabla_{\mathbf{W}_i^{(2, l)}} \mathcal{L}_{\mathrm{recon}}\big( \mathbf{m} \big) \left( \tilde{\mathbf{A}}_i^{(2, l)} + \delta_{\mathrm{kfac}}\mathbf{I} \right)^{-1} \right]
$$

---

### Section 21: SoftVQ Semantic Codebook Evolution & Pruning
Dynamic usage tracking and thresholded pruning:

$$
u_k(t+\Delta t) = \gamma_{\mathrm{usage}} \cdot u_k(t) + \sum_{l=0}^L \sum_{j \in V^{(l)}} \mathbb{I}\left( k = \arg\min_{m: \, m_{\mathrm{pruned}, m} = 1} d_{\mathcal{H}^d}\left(\boldsymbol{\mu}_j^{(l)}(t), \mathrm{Exp}_{\mathbf{o}}^{\mathcal{H}^d}(E(\mathbf{o})\mathbf{c}_m)\right) \right)
$$

$$
m_{\mathrm{pruned}, k} = \begin{cases} 1, & k \in \mathcal{K}_{\mathrm{protected}} \lor k = 0 \\ 0, & u_k(t+\Delta t) < \theta_{\mathrm{prune}} = 0.05 \land k \notin \mathcal{K}_{\mathrm{protected}} \\ 1, & u_k(t+\Delta t) \ge 0.05 \end{cases}
$$

Spontaneous concept spawning under epistemic prediction divergence:

$$
\text{If } \left( \left( \tilde{\boldsymbol{\varepsilon}}_i^{(l)}(t) \right)^\top \tilde{\mathbf{\Pi}}_i^{(l)}(t) \left( \tilde{\boldsymbol{\varepsilon}}_i^{(l)}(t) \right) > \theta_{\mathrm{spawn}} \right) \land \left( |\mathcal{C}_{\mathrm{active}}| < K_{\max} \right) \land \left( \min_{m} d_{\mathcal{H}^d}\left(\boldsymbol{\mu}_i^{(l)}, \mathrm{Exp}_{\mathbf{o}}^{\mathcal{H}^d}(E(\mathbf{o})\mathbf{c}_m)\right) > d_{\min} \right):
$$

$$
\mathcal{C} \leftarrow \mathcal{C} \cup \left\{ E(\mathbf{o})^\top \boldsymbol{\eta}_{\mathcal{L}} \mathrm{Log}_{\mathbf{o}}^{\mathcal{H}^d}\left( \boldsymbol{\mu}_i^{(l)}(t) \right) + \boldsymbol{\eta}_{\mathrm{noise}} \right\}, \quad u_{\mathrm{new}} \leftarrow 1.0, \quad m_{\mathrm{pruned}, \mathrm{new}} \leftarrow 1
$$

---

### Section 22: Thermodynamic Langevin Frame Noise

$$
\mathbf{T}_i\left( \tilde{\boldsymbol{\varepsilon}}_i^{(l)} \right) = T_{\max} \cdot \tanh\left( \frac{\alpha_{\mathrm{temp}}}{T_{\max}} \max\left(0.0, \left( \tilde{\boldsymbol{\varepsilon}}_i^{(l)} \right)^\top \tilde{\mathbf{\Pi}}_i^{(l)} \left( \tilde{\boldsymbol{\varepsilon}}_i^{(l)} \right) \right) \right), \quad T_{\max} = 1.0
$$

$$
\boldsymbol{\zeta}_{i, t}^{\mathrm{tangent}} = E\left(\boldsymbol{\mu}_i^{(l)}(t)\right) \left[ \sigma_{\mathrm{discrete}, i}^{(l)}(t) \boldsymbol{\xi}_{i, t} \right], \quad \boldsymbol{\xi}_{i, t} \sim \mathcal{N}\left(\mathbf{0}, \mathbf{I}_{d-1}\right)
$$

---

### Section 23: Fast Associative Attention & Memristive Crossbar Currents
Key-value manifold projections:

$$
\mathbf{m}_m^{\mathrm{key}} = \mathrm{Exp}_{\mathbf{o}}^{\mathcal{H}^d}\left( E(\mathbf{o}) \cdot \mathbf{W}_{\mathrm{key}} \mathbf{h}_m \right), \quad \mathbf{m}_m^{\mathrm{value}} = \mathrm{Exp}_{\mathbf{o}}^{\mathcal{H}^d}\left( E(\mathbf{o}) \cdot \mathbf{W}_{\mathrm{val}} \mathbf{h}_m \right), \quad \mathbf{h}_m \in \mathbf{H}_{\mathrm{memory}}
$$

$$
\text{score}_m = -\frac{d_{\mathcal{H}^d}^2\left( \boldsymbol{\mu}_i^{(l)}, \mathbf{m}_m^{\mathrm{key}} \right)}{T_{\mathrm{attn}} \sqrt{d-1}}, \quad w_m = \frac{\exp\left( \text{score}_m - \max_k(\text{score}_k) \right)}{\sum_{k \in \mathcal{M}_{\mathrm{fast}}} \exp\left( \text{score}_k - \max_k(\text{score}_k) \right)}
$$

$$
\tilde{\mathbf{\Delta}}_{\mathrm{fast}, i}^{\mathrm{attn}}(t) = \eta_{\mathrm{dopamine}} \sum_{m \in \mathcal{M}_{\mathrm{fast}}} w_m E\left(\boldsymbol{\mu}_i^{(l)}\right)^\top \boldsymbol{\eta}_{\mathcal{L}} \mathrm{Log}_{\boldsymbol{\mu}_i^{(l)}}\left( \mathbf{m}_m^{\mathrm{value}} \right) \in \mathbb{R}^{d-1}
$$

$$
\tilde{\mathbf{\Delta}}_{\mathrm{fast}, i}^{(l)}(t) = \tilde{\mathbf{\Delta}}_{\mathrm{fast}, i}^{\mathrm{attn}}(t) + E\left(\boldsymbol{\mu}_i^{(l)}\right)^\top \boldsymbol{\eta}_{\mathcal{L}} \mathbf{P}_{\mathbf{o} \to \boldsymbol{\mu}_i^{(l)}(t)}\left( E(\mathbf{o}) \cdot \mathbf{W}_{\mathrm{analog\_crossbar}} \cdot \mathbf{s}_i(t) \right) \in \mathbb{R}^{d-1}
$$

---

### Section 24: Lyapunov-Hespanha Dynamic Stability
Unified Lyapunov energy functional:

$$
V(\mathbf{\Xi}(t)) = \frac{1}{2} \sum_{l=1}^L \sum_{i \in V^{(l)}} d_{\mathcal{H}^d}^2\left( \boldsymbol{\mu}_i^{(l)}(t), \overline{\boldsymbol{\mu}}^{(l)}(t) \right) + \frac{1}{2} \|\mathbf{a}(t) - \mathbf{a}_{\mathrm{target}}(t)\|^2 \ge 0
$$

$$
\frac{\mathrm{d}V}{\mathrm{d}t} \le -\alpha_{\mathrm{damp}}(t) \cdot V + \sigma_{\mathrm{noise}}^2
$$

$$
\tau_{\mathrm{dwell}} > \frac{\ln(1 + \mu_{\mathrm{jump}})}{\alpha_{\min}}, \quad V(t^+) \le (1 + \mu_{\mathrm{jump}}) V(t^-)
$$

---

### Section 25: Unbiased SOC Criticality Regulator
Pareto tail exponent tracking:

$$
\mathcal{S}_{\mathrm{tail}}(t) = \left\{ i \in \bigcup_{l=0}^L V^{(l)} \;\middle|\; \left\| \tilde{\boldsymbol{\varepsilon}}_i(t) \right\|_{\tilde{\mathbf{\Pi}}_i} \ge \varepsilon_{\min} \right\}
$$

$$
\overline{L}_{\mathrm{tail}}(t) = \begin{cases} (1 - \gamma_{\mathrm{soc}}) \overline{L}_{\mathrm{tail}}(t - \Delta t) + \gamma_{\mathrm{soc}} \left( \frac{1}{|\mathcal{S}_{\mathrm{tail}}|} \sum_{i \in \mathcal{S}_{\mathrm{tail}}} \ln \left( \frac{\left\| \tilde{\boldsymbol{\varepsilon}}_i(t) \right\|_{\tilde{\mathbf{\Pi}}_i}}{\varepsilon_{\min}} \right) \right), & |\mathcal{S}_{\mathrm{tail}}| \ge N_{\min} \\ \overline{L}_{\mathrm{tail}}(t - \Delta t), & |\mathcal{S}_{\mathrm{tail}}| < N_{\min} \end{cases}
$$

$$
\hat{\alpha}_{\mathrm{measured}}(t) = 1 + \frac{1}{\max\left( \delta_{\mathrm{soc}}, \overline{L}_{\mathrm{tail}}(t) \right)}
$$

$$
\Delta \alpha = \mathrm{clip}\left( \Delta t \cdot \left(1.2 \cdot \eta_\alpha\right) \left( 1.5 - \hat{\alpha}_{\mathrm{measured}}(t) \right), -\Delta \alpha_{\max}, +\Delta \alpha_{\max} \right)
$$

$$
\alpha_{\mathrm{damp}}(t + \Delta t) = \begin{cases} \max\left( \alpha_{\min}, \alpha_{\mathrm{damp}}(t) + \Delta \alpha \right), & |\mathcal{S}_{\mathrm{tail}}| \ge N_{\min} \\ \alpha_{\mathrm{damp}}(t), & |\mathcal{S}_{\mathrm{tail}}| < N_{\min} \end{cases}, \quad \alpha_{\min} = 0.05
$$

---

### Section 26: Poincaré Holographic Memory & Harish-Chandra Transforms
Lorentz group spherical basis ($\mathrm{SO}^+(1, d-1)$):

$$
\phi_{\lambda, \mathbf{b}}(\boldsymbol{\mu}) = \left( \langle \boldsymbol{\mu}, \boldsymbol{\xi}(\mathbf{b}) \rangle_{\mathcal{L}} \right)^{-\frac{d-2}{2} + i \lambda}, \quad \lambda \in \mathbb{R}^+, \quad \mathbf{b} \in \mathbb{S}^{d-2}, \quad \boldsymbol{\xi}(\mathbf{b}) = (1, \mathbf{b})^\top
$$

Direct Harish-Chandra episode encoding:

$$
\tilde{A}_m(\lambda, \mathbf{b}) = \int_{\mathcal{H}^d} \mathrm{Log}_{\mathbf{o}}^{\mathcal{H}^d}\left(\boldsymbol{\mu}_m\right) \cdot \phi_{-\lambda, \mathbf{b}}(\boldsymbol{\mu}) \, \mathrm{d}\mathrm{Vol}_{\mathcal{H}^d}(\boldsymbol{\mu})
$$

Interference retrieval via Plancherel measure $|c(\lambda)|^{-2}$:

$$
\mathbf{H}_{\mathrm{recall}}\left(\boldsymbol{\mu}_i^{(l)}\right) = \int_{0}^\infty \mathrm{d}\lambda \, |c(\lambda)|^{-2} \int_{\mathbb{S}^{d-2}} \mathrm{d}\mathbf{b} \left[ \sum_{m \in \mathcal{M}_{\mathrm{fast}}} \tilde{A}_m(\lambda, \mathbf{b}) \exp\left( -i \frac{\lambda \cdot d_{\mathcal{H}^d}\left( \boldsymbol{\mu}_i^{(l)}, \boldsymbol{\mu}_m \right)}{\hbar_{\mathrm{cog}}} \right) \right] \phi_{\lambda, \mathbf{b}}\left(\boldsymbol{\mu}_i^{(l)}\right)
$$

$$
|c(\lambda)|^{-2} = \frac{1}{2\pi} \cdot \left| \frac{\Gamma\left( \frac{d-1}{2} \right) \Gamma(i \lambda)}{\Gamma\left( \frac{d-1}{2} + i \lambda \right)} \right|^{-2}
$$

---

### Section 27: Multi-Agent Conformal Theory of Mind (ToM)

$$
\mathbf{a}_{\mathrm{ToM}}^*(t) = \arg\max_{\mathbf{a}} \left[ \mathcal{I}_{\mathrm{curiosity}}\big( \mathbf{a} \big) + \sum_{j \neq \mathrm{self}} \mathbb{E}_{\boldsymbol{\mu}_j}\Big[ U_{\mathrm{agent}}(\mathbf{a}, \mathbf{a}_j) \;\Big|\; \mathcal{T}_{\mathrm{ToM}}\big( \boldsymbol{\mu}_j^{(L)} \big) \Big] \right]
$$

---

### Section 28: Anti-Livelock VRAM Ring Buffer with Thermal Annealing
Dynamic thought step-size:

$$
\Delta t_{\mathrm{thought}}(\mathbf{\Xi}) = \Delta t_0 \cdot \left( 1 + \gamma_{\mathrm{sys2}} \cdot \mathrm{Var}_{\mathbf{\Theta}}\left[ \boldsymbol{\mu}^{(L)} \right] \right)
$$

Emergency state rollback and exponential backoff:

$$
\text{If } \Big( \dot{\bar{\mathcal{J}}}(t) > \theta_{\mathrm{impasse}} \Big) \land \Big( t - t_{\mathrm{last\_reset}} \ge \tau_{\mathrm{cooldown}}(t) \Big) \implies \begin{cases} \left( \mathbf{\Xi}_{\mathrm{internal}}, \tilde{\mathbf{A}}, \tilde{\mathbf{B}}, \mathcal{M}_{\mathrm{fast}} \right)_{t+\Delta t} \leftarrow \mathcal{B}_{\mathrm{ring\_buffer}}(K_{\mathrm{check}} = 5) \\ \boldsymbol{\mu}_0^{(0)}(t+\Delta t) \leftarrow \text{grounded sensory stream} \\ \mathbf{T}_i \leftarrow T_{\max} \cdot \left( 0.8 \right)^{N_{\mathrm{consecutive\_resets}}} \\ \tau_{\mathrm{cooldown}}(t+\Delta t) \leftarrow \min\left( \tau_{\max}, 2.0 \cdot \tau_{\mathrm{cooldown}}(t) \right) \\ N_{\mathrm{consecutive\_resets}} \leftarrow N_{\mathrm{consecutive\_resets}} + 1 \\ t_{\mathrm{last\_reset}} \leftarrow t \end{cases}
$$

$$
\text{If } \dot{\bar{\mathcal{J}}}(t) \le 0 \implies N_{\mathrm{consecutive\_resets}} \leftarrow 0, \quad \tau_{\mathrm{cooldown}} \leftarrow \tau_{\mathrm{cooldown}, 0}
$$

---

### Section 29: Neuromorphic STE Spike Encoder

$$
\lambda_{\mathrm{spike}, i}^{(l)}(t) = f_{\max} \cdot \sigma\left( \mathbf{W}_{\mathrm{enc}} \cdot E(\mathbf{o})^\top \boldsymbol{\eta}_{\mathcal{L}} \mathrm{Log}_{\mathbf{o}}^{\mathcal{H}^d}\left( \boldsymbol{\mu}_i^{(l)}(t) \right) + \mathbf{b}_{\mathrm{enc}} \right) \in \mathbb{R}^{M_{\mathrm{crossbar}}}
$$

$$
p_{i, m}(t) = 1.0 - \exp\left( -\lambda_{\mathrm{spike}, i, m}^{(l)}(t) \cdot \Delta t \right) \in [0, 1]
$$

$$
\mathbf{s}_i(t) = \mathrm{step}\left( p_i(t) - \boldsymbol{\xi}_{\mathrm{uniform}} \right) - \mathrm{sg}\left[ p_i(t) \right] + p_i(t) \in \{0, 1\}^{M_{\mathrm{crossbar}}}, \quad \boldsymbol{\xi}_{\mathrm{uniform}} \sim \mathcal{U}(0, 1)^{M_{\mathrm{crossbar}}}
$$

---

### Section 30: System-1 Fast Geodesic Inference Relaxation

$$
\boldsymbol{\mu}_i^{(l)}(\tau + 1) = \mathrm{Exp}_{\boldsymbol{\mu}_i^{(l)}(\tau)}^{\mathcal{H}^d}\left( E(\boldsymbol{\mu}_i^{(l)}(\tau)) \cdot \mathbf{w}_{\mathrm{relax}, i}^{(l)}(tau) \right), \quad \tau \in \{0, 1, \dots, K_{\mathrm{relax}}-1\}
$$

$$
\mathbf{w}_{\mathrm{relax}, i}^{(l)}(\tau) = \left( \mathbf{I}_{d-1} + \frac{\Delta t}{\epsilon_0} \pi_i \tilde{\mathbf{\Pi}}_i \right)^{-1} \left( \frac{\Delta t}{\epsilon_0} \tilde{\mathbf{v}}_{\mathrm{bottom-up}, i}^{(l)}(\tau) \right)
$$

$$
\mathbf{w}_{\mathrm{relax}, i}^{(l)} \leftarrow \frac{\mathbf{w}_{\mathrm{relax}, i}^{(l)}}{\max\left(1.0, \frac{\|\mathbf{w}_{\mathrm{relax}, i}^{(l)}\|_2}{v_{\max}}\right)}
$$

---

### Section 31: Conformal Connection & Christoffel Symbols
The conformal deformation tensor:

$$
\tilde{g}_{AB}(\boldsymbol{\mu}) = \Omega^2(\boldsymbol{\mu}) (\boldsymbol{\eta}_{\mathcal{L}})_{AB}, \quad \Omega(\boldsymbol{\mu}) = \max\left( \delta_{\mathrm{eps}}, \mathcal{F}_{\mathrm{threshold}} - \mathcal{F}_{\mathrm{pragmatic}}^{\mathrm{Sylvester}}(\boldsymbol{\mu}) \right)
$$

Christoffel symbols of the second kind:

$$
\tilde{\Gamma}_{BC}^A = \Gamma_{BC}^A + \delta_B^A \partial_C \ln \Omega(\boldsymbol{\mu}) + \delta_C^A \partial_B \ln \Omega(\boldsymbol{\mu}) - (\boldsymbol{\eta}_{\mathcal{L}})_{BC} (\boldsymbol{\eta}_{\mathcal{L}})^{AK} \partial_K \ln \Omega(\boldsymbol{\mu})
$$

Conformal fictitious force:

$$
\tilde{\mathbf{v}}_{\mathrm{conformal}, i}^{(l)}(t) = -\|\mathbf{w}_i^{(l)}(t)\|_2^2 \cdot E\left(\boldsymbol{\mu}_i^{(l)}\right)^\top \boldsymbol{\eta}_{\mathcal{L}} \nabla_{\mathcal{H}^d} \ln \Omega\left(\boldsymbol{\mu}_i^{(l)}\right) + 2 \left( \left(\mathbf{w}_i^{(l)}(t)\right)^\top E\left(\boldsymbol{\mu}_i^{(l)}\right)^\top \boldsymbol{\eta}_{\mathcal{L}} \nabla_{\mathcal{H}^d} \ln \Omega\left(\boldsymbol{\mu}_i^{(l)}\right) \right) \mathbf{w}_i^{(l)}(t)
$$

---

### Section 32: Non-Separable Maupertuis-Jacobi Hamiltonian Mechanics
Cotangent bundle Hamiltonian $T^*\mathcal{H}^d$:

$$
H_{\mathrm{neuron}}(\boldsymbol{\mu}, \mathbf{p}) = \frac{1}{2 \Omega^2(\boldsymbol{\mu})} \langle \mathbf{p}, \mathbf{p} \rangle_{\mathcal{L}} + \mathcal{F}_{\mathrm{pragmatic}}^{\mathrm{Sylvester}}(\boldsymbol{\mu}) + \frac{\kappa_v}{2 \Omega^2(\boldsymbol{\mu})} \|\mathbf{p}\|_{\mathcal{L}}^2
$$

Canonical equations of motion:

$$
\frac{\mathrm{d}\boldsymbol{\mu}}{\mathrm{d}t} = \frac{\boldsymbol{\eta}_{\mathcal{L}}^{-1} \mathbf{p}}{\Omega^2(\boldsymbol{\mu})}, \qquad \frac{\mathrm{d}\mathbf{p}}{\mathrm{d}t} = -\nabla_{\mathcal{H}^d} \mathcal{F}_{\mathrm{pragmatic}}^{\mathrm{Sylvester}}(\boldsymbol{\mu}) + \frac{\nabla_{\mathcal{H}^d} \Omega(\boldsymbol{\mu})}{\Omega^3(\boldsymbol{\mu})} \langle \mathbf{p}, \mathbf{p} \rangle_{\mathcal{L}} + \lambda_{\mathrm{mid}} \boldsymbol{\eta}_{\mathcal{L}} \boldsymbol{\mu}
$$

Symplectic midpoint step ($\boldsymbol{\mu}_{1/2} = \frac{\boldsymbol{\mu}_t + \boldsymbol{\mu}_{t+\Delta t}}{2}$, $\mathbf{p}_{1/2} = \frac{\mathbf{p}_t + \mathbf{p}_{t+\Delta t}}{2}$):

$$
\frac{\boldsymbol{\mu}_{t+\Delta t} - \boldsymbol{\mu}_t}{\Delta t} = \frac{\boldsymbol{\eta}_{\mathcal{L}}^{-1} \mathbf{p}_{1/2}}{\Omega^2(\boldsymbol{\mu}_{1/2})}
$$

$$
\frac{\mathbf{p}_{t+\Delta t} - \mathbf{p}_t}{\Delta t} = -\nabla_{\mathcal{H}^d} \mathcal{F}(\boldsymbol{\mu}_{1/2}) + \frac{\nabla \Omega(\boldsymbol{\mu}_{1/2})}{\Omega^3(\boldsymbol{\mu}_{1/2})} \langle \mathbf{p}_{1/2}, \mathbf{p}_{1/2} \rangle_{\mathcal{L}} + \lambda_{\mathrm{mid}} \boldsymbol{\eta}_{\mathcal{L}} \boldsymbol{\mu}_{1/2}
$$

---

### Section 33: Generalized Cole-Hopf Diffusion on Riemannian Manifolds
Logarithmic value potential substitution:

$$
\psi(\boldsymbol{\mu}, t) = \exp\left( -\frac{V(\boldsymbol{\mu}, t)}{\lambda \sigma^2} \right)
$$

Parabolic Schrödinger-Beltrami diffusion equation:

$$
\frac{\partial \psi(\boldsymbol{\mu}, t)}{\partial t} = \frac{\lambda \sigma^2}{2} \Delta_{\mathcal{H}^d} \psi(\boldsymbol{\mu}, t) - \frac{\mathcal{F}_{\mathrm{pragmatic}}(\boldsymbol{\mu})}{\lambda \sigma^2} \psi(\boldsymbol{\mu}, t)
$$

Linear superposition of conflicting goal wavefronts:

$$
\psi_{\mathrm{total}}(\boldsymbol{\mu}) = \sum_{k=1}^K w_k \psi_k(\boldsymbol{\mu})
$$

Instantaneous geodesic inference velocity optimum:

$$
\mathbf{w}_{\mathrm{optimal}}(\boldsymbol{\mu}) = \lambda \sigma^2 \cdot E(\boldsymbol{\mu})^\top \boldsymbol{\eta}_{\mathcal{L}} \nabla_{\mathcal{H}^d} \ln \psi_{\mathrm{total}}(\boldsymbol{\mu})
$$

---

### Section 34: Explicit Polyhedral Projection of AIXI Simplex
Kraft lower and upper decision approximations:

$$
Q_{\min}(a) = \sum_{p \in \mathrm{Halted}} \frac{2^{-l(p)}}{M_k} R(p, a), \quad Q_{\max}(a) = Q_{\min}(a) + \Omega_{\mathrm{unresolved}}(k) \frac{r_{\max}}{1 - \gamma}
$$

Analytical identity to the universal AIXI optimum:

$$
Q_{\min}(a^*) > \max_{a \neq a^*} Q_{\max}(a) \implies a_{\mathrm{computable}}^* \equiv a_{\mathrm{AIXI}}^*
$$

Polyhedral pressure force on motor degrees of freedom:

$$
\mathbf{m}_{\mathrm{poly}}(a^*) = \varepsilon_{\mathrm{safe}} \cdot \frac{\sum_{a \neq a^*} \frac{\mathbf{e}_{a^*} - \mathbf{e}_a}{Q_{\min}(a^*) - Q_{\max}(a) + \delta_{\mathrm{eps}}}}{\left\| \sum_{a \neq a^*} \frac{\mathbf{e}_{a^*} - \mathbf{e}_a}{Q_{\min}(a^*) - Q_{\max}(a) + \delta_{\mathrm{eps}}} \right\|_2}
$$

---

### Section 35: Asynchronous Zero-Downtime Gödel JIT Engine Hot-Swapping
Deductive self-improvement decision criterion:

$$
\mathcal{P} \vdash \left[ \mathbb{E}[U(F_{\mathrm{new}})] > \mathbb{E}[U(F_{\mathrm{old}})] \right] \implies \mathrm{atomic\_store}(\& \mathrm{ActiveEngine}, \mathrm{Engine}_{\mathrm{new}})
$$

Double-buffering guarantees bounded execution jitter ($\Delta t = 0$).

---

### Section 36: Sectional Lobachevsky-Riemann Curvature Adapter
Generalized distance in manifolds of dynamically variable sectional curvature $\kappa$:

$$
d_\kappa(x, y) = \frac{1}{\sqrt{|\kappa|}} \mathrm{arcosh}\left( 1 + \frac{2|\kappa| \|x - y\|^2}{(1 + \kappa \|x\|^2)(1 + \kappa \|y\|^2)} \right)
$$

Layer sectional curvature $\kappa_l(t) \in (-\infty, +\infty)$ adapts via gradient descent along the free energy surface:

$$
\frac{\partial \kappa_l}{\partial t} = -\eta_\kappa \nabla_{\kappa_l} \mathcal{F}_{\mathrm{pragmatic}}^{\mathrm{Sylvester}}
$$

---

### Section 37: Lock-Free RCU Multi-Timescale Engine Dispatcher
Decoupling execution across temporal scales ($\epsilon_0 \sim 10^{-3}\text{ s}$, $\epsilon_1 \sim 1\text{ s}$, $\epsilon_2 \sim 10^5\text{ s}$) using Read-Copy-Update (RCU):

$$
\mathrm{Thread}_{\mathrm{fast}} \to \mathrm{Read}(\mathrm{ActivePtr})
$$

$$
\mathrm{Thread}_{\mathrm{slow}} \to \mathrm{Write}(\mathrm{ShadowCopy}) \to \mathrm{AtomicSwap}()
$$

Guarantees deadlock-free and livelock-free continuous inference.

---

### Section 38: Dynamic Isometric Dimension Expansion
Isometric embedding upon codebook capacity saturation:

$$
\iota: \mathcal{H}^d \hookrightarrow \mathcal{H}^{d+k}, \quad \boldsymbol{\mu} \mapsto (\mu_0, \dots, \mu_{d-1}, \underbrace{0, \dots, 0}_{k})^\top
$$

$$
\langle \iota(x), \iota(y) \rangle_{\mathcal{L}^{d+k}} \equiv \langle x, y \rangle_{\mathcal{L}^d} = -1.0
$$

Preserves historical relational representations while expanding capacity by a factor of $e^{k \cdot R}$.

---

## Part II: Algorithmic Execution Specifications

---

### Algorithm 1: Multimodal Dirichlet Sensory Grounding ($l=0$)
* **Inputs:** Multimodal observation tensors $\mathbf{x}_{\mathrm{img}} \in \mathbb{R}^{C \times H \times W}$, $\mathbf{x}_{\mathrm{aud}} \in \mathbb{R}^{T_{\mathrm{aud}}}$, $\mathbf{x}_{\mathrm{text}} \in \mathbb{Z}^{T_{\mathrm{txt}}}$.
* **Outputs:** Grounded Dirichlet manifold coordinate $\boldsymbol{\mu}_0^{(0)}(t+\Delta t) \in \mathcal{H}^d$, mutual information bound $\mathcal{I}_{\mathrm{grounding}}$.
* **Complexity:** $O(d \cdot \sum \mathrm{dim}(\text{inputs}))$.
* **Execution Steps:**
  1. Extract multimodal latent embeddings: $\mathbf{z}_{\mathrm{vis}} = \mathbf{W}_{\mathrm{vis}} \mathrm{Enc}_{\mathrm{vis}}(\mathbf{x}_{\mathrm{img}})$, $\mathbf{z}_{\mathrm{aud}} = \mathbf{W}_{\mathrm{aud}} \mathrm{Enc}_{\mathrm{aud}}(\mathbf{x}_{\mathrm{aud}})$, $\mathbf{z}_{\mathrm{txt}} = \mathbf{W}_{\mathrm{txt}} \mathrm{Enc}_{\mathrm{txt}}(\mathbf{x}_{\mathrm{text}})$.
  2. Compute bundle projection: $\mathbf{a}_{\mathrm{bundle}} = \pi_{\mathrm{bundle}}(\mathbf{z}_{\mathrm{vis}} \parallel \mathbf{z}_{\mathrm{aud}} \parallel \mathbf{z}_{\mathrm{txt}}) \in \mathbb{R}^{d-1}$.
  3. Lift to tangent pole space: $\mathbf{v}_{\mathrm{bundle}} = E(\mathbf{o})\mathbf{a}_{\mathrm{bundle}} \in T_{\mathbf{o}}\mathcal{H}^d$, where $E(\mathbf{o}) = [\mathbf{0}_{1 \times (d-1)}; \mathbf{I}_{d-1}]$.
  4. Evaluate pole exponential map: $\boldsymbol{\mu}_0^{(0)}(t+\Delta t) = \mathrm{Exp}_{\mathbf{o}}^{\mathcal{H}^d}(\mathbf{v}_{\mathrm{bundle}})$.
  5. Enforce boundary conditions: $\mathbf{w}_0^{(0)} \leftarrow \mathbf{0}$, $\mathbf{p}_0^{(0)} \leftarrow \mathbf{0}$.
  6. Compute contrastive InfoNCE lower bound $\mathcal{I}_{\mathrm{grounding}}$ across batch $\mathcal{B}$.

---

### Algorithm 2: System-1 Fast Geodesic Inference Step ($\epsilon_0 \sim 1\text{ ms}$)
* **Inputs:** Current state $\boldsymbol{\mu}_i^{(l)}(t)$, context $\tilde{\mathbf{a}}_i^{(l)}(t)$, layer barycenter $\overline{\boldsymbol{\mu}}^{(l)}(t)$.
* **Outputs:** Updated coordinates $\boldsymbol{\mu}_i^{(l)}(t+\Delta t)$, velocity $\mathbf{w}_i^{(l)}(t)$, variational precision $\pi_i^{(l)}(t+\Delta t)$.
* **Complexity:** $O(|V| \cdot (d-1) \cdot (h_{\mathrm{dim}} + r^2))$.
* **Execution Steps:**
  1. Stream layer barycenter $\overline{\boldsymbol{\mu}}^{(l)}$ via Fréchet exponential integration.
  2. Construct analytical Lorentz boost tetrad frame $E(\boldsymbol{\mu}_i^{(l)})$.
  3. SoftVQ context quantization: evaluate distances $d_{\mathcal{H}^d}^2$ to centroids $\mathbf{c}_k$, normalize softmax weights, assemble $\tilde{\mathbf{a}}_i^{(l)}$.
  4. Forward pass through two-layer perceptron: hidden state $\mathbf{h} = \sigma(\mathbf{W}^{(1)}\tilde{\mathbf{a}} + \mathbf{b}^{(1)})$, output $\mathbf{u}_2 = \mathbf{W}^{(2)}\mathbf{h} + \mathbf{b}^{(2)}$.
  5. Enforce Euclidean retraction: $\mathbf{u}_{2, \mathrm{safe}} = \mathbf{u}_2 \cdot \min(1.0, 3.0 / \|\mathbf{u}_2\|_2)$.
  6. Compute target prediction: $\boldsymbol{\mu}_{\mathrm{pred}} = \mathrm{Exp}_{\overline{\boldsymbol{\mu}}}(E(\overline{\boldsymbol{\mu}})\mathbf{u}_{2, \mathrm{safe}})$.
  7. Compute frame-projected prediction error: $\boldsymbol{\varepsilon}_i = \mathrm{Log}_{\boldsymbol{\mu}_i}(\boldsymbol{\mu}_{\mathrm{pred}})$, $\tilde{\boldsymbol{\varepsilon}}_i = E(\boldsymbol{\mu}_i)^\top \boldsymbol{\eta}_{\mathcal{L}} \boldsymbol{\varepsilon}_i$.
  8. Oja-QR subspace update: project $\mathbf{y} = \tilde{\mathbf{V}}^\top \tilde{\boldsymbol{\varepsilon}}$, update singular values $\mathbf{s}$, QR factorize $\tilde{\mathbf{V}}$, compute residual $\tilde{\boldsymbol{\varepsilon}}_{\mathrm{residual}}$, smooth $\tilde{\mathbf{d}}$.
  9. Woodbury inversion: compute precision matrix $\tilde{\mathbf{\Pi}}_i$.
  10. Calculate Maupertuis conformal drift $\tilde{\mathbf{v}}_{\mathrm{conformal}, i}^{(l)}$ and momentum term $E(\boldsymbol{\mu}_i)^\top \boldsymbol{\eta}_{\mathcal{L}} \frac{\nabla \Omega}{\Omega^3} \langle \mathbf{p}_i, \mathbf{p}_i \rangle_{\mathcal{L}}$.
  11. Assemble total forces $\tilde{\mathbf{v}}_{\mathrm{forces}}$, apply semi-implicit $A$-stable solver, add Langevin noise $\sigma_{\mathrm{discrete}} \boldsymbol{\xi}$.
  12. Clamp Euclidean velocity norm to $v_{\max} = 5.0$.
  13. Execute exponential step $\mathrm{Exp}_{\boldsymbol{\mu}_i}(E(\boldsymbol{\mu}_i)\mathbf{w}_i)$ and project coordinates onto hyperboloid.
  14. Update adaptive scalar precision $\pi_i(t+\Delta t)$.

---

### Algorithm 3: Symplectic Implicit Midpoint Step on Cotangent Bundle $T^*\mathcal{H}^{16}$
* **Inputs:** Phase state $(\boldsymbol{\mu}_{i, t}^{(l)}, \mathbf{p}_{i, t}^{(l)})$, free energy gradient $\nabla_{\mathcal{H}^d} \mathcal{F}_{\mathrm{pragmatic}}$.
* **Outputs:** Updated symplectic phase coordinates $(\boldsymbol{\mu}_{i, t+\Delta t}^{(l)}, \mathbf{p}_{i, t+\Delta t}^{(l)})$.
* **Complexity:** $O(d)$ elementary operations per neuron.
* **Execution Steps:**
  1. Compute midpoint coordinates and momenta:

$$
\boldsymbol{\mu}_{i, t+\frac{\Delta t}{2}}^{(l)} = \frac{\boldsymbol{\mu}_{i, t}^{(l)} + \boldsymbol{\mu}_{i, t+\Delta t}^{(l)}}{2}, \quad \mathbf{p}_{i, t+\frac{\Delta t}{2}}^{(l)} = \frac{\mathbf{p}_{i, t}^{(l)} + \mathbf{p}_{i, t+\Delta t}^{(l)}}{2}
$$

  2. Compute physical force at midpoint: $\mathbf{g}_{t+1/2} = \nabla_{\mathcal{H}^d} \mathcal{F}_{\mathrm{pragmatic}}(\boldsymbol{\mu}_{i, t+\frac{\Delta t}{2}}^{(l)})$.
  3. Implicit momentum kick:

$$
\mathbf{p}_{i, t+\Delta t}^{(l)} = \mathbf{p}_{i, t}^{(l)} - \Delta t \left( \mathbf{g}_{t+1/2} + \lambda_{\mathrm{mid}} \boldsymbol{\eta}_{\mathcal{L}} \boldsymbol{\mu}_{i, t+\frac{\Delta t}{2}}^{(l)} \right)
$$

  4. Implicit coordinate step:

$$
\boldsymbol{\mu}_{i, t+\Delta t}^{(l)} = \boldsymbol{\mu}_{i, t}^{(l)} + \Delta t \boldsymbol{\eta}_{\mathcal{L}}^{-1} \mathbf{p}_{i, t+\frac{\Delta t}{2}}^{(l)}
$$

  5. Solve scalar constraint equation for $\lambda_{\mathrm{mid}}$ to satisfy $\langle \boldsymbol{\mu}_{i, t+\Delta t}^{(l)}, \boldsymbol{\mu}_{i, t+\Delta t}^{(l)} \rangle_{\mathcal{L}} = -1.0$.
  6. Execute momentum tangent projection: $\mu_{\mathrm{mid}} = \langle \boldsymbol{\mu}_{i, t+\Delta t}^{(l)}, \boldsymbol{\eta}_{\mathcal{L}}^{-1} \mathbf{p}_{i, t+\Delta t}^{(l)} \rangle_{\mathcal{L}}$, then update $\mathbf{p}_{i, t+\Delta t}^{(l)} \leftarrow \mathbf{p}_{i, t+\Delta t}^{(l)} - \mu_{\mathrm{mid}} \boldsymbol{\eta}_{\mathcal{L}} \boldsymbol{\mu}_{i, t+\Delta t}^{(l)}$.
  7. Verify canonical symplectic 2-form invariance $\omega_{t+\Delta t} = \omega_t$.

---

### Algorithm 4: Fast Sweeping Eikonal Wavefront Solver
* **Inputs:** Adjacency matrix $E_{ij}$, vertex coordinates $\boldsymbol{\mu}_i$, refractive index $n_i = 1/\Omega_i$, target coordinate $\boldsymbol{\mu}_{\mathrm{target}}$.
* **Outputs:** Discrete scalar action field $S_i$ for all vertices $i \in V$.
* **Complexity:** $O(K_{\mathrm{sweep}} \cdot |E|)$.
* **Execution Steps:**
  1. Initialize action field: $S(\boldsymbol{\mu}_{\mathrm{target}}) \leftarrow 0.0$, all other vertices $S_i \leftarrow +\infty$.
  2. Iterate across graph topological sweeps:
     * Extract neighbor values $\mathcal{N}(i)$ and chordal distances $d_k = d_{\mathcal{H}^d}(\boldsymbol{\mu}_i, \boldsymbol{\mu}_k)$.
     * Sort neighbor values in ascending order: $S_{(1)} \le S_{(2)} \le \dots \le S_{(m)}$.
     * Determine active neighbor count $m$ satisfying Godunov causality $S_i > S_{(m)}$.
     * Compute candidate: $S_i^{\mathrm{new}} = \frac{B_m + \sqrt{B_m^2 - A_m(C_m - n_i^2)}}{A_m}$ incorporating viscosity $\varepsilon_{\mathrm{visc}} (\Delta_{\mathcal{H}^d} S)_i$.
     * Update monotonically: $S_i \leftarrow \min(S_i, S_i^{\mathrm{new}})$.
  3. Check convergence criterion: $\max_i |S_i^{(k+1)} - S_i^{(k)}| < 10^{-6}$.
  4. Compute discrete Riemannian gradient $\nabla_{\mathcal{H}^d} S_i$ via upwind finite differences.

---

### Algorithm 5: Linearized Cole-Hopf & Polyhedral AIXI Motor Synthesis
* **Inputs:** Top-layer coordinate $\boldsymbol{\mu}^{(L)}$, goal potentials $\psi_k$, optimal discrete action $a^*$, social vector $\mathbf{a}_{\mathrm{ToM}}^*$.
* **Outputs:** Continuous motor control vector $\mathbf{a}(t+\Delta t) \in [-1, 1]^{d_{\mathrm{act}}}$.
* **Complexity:** $O(d_{\mathrm{act}} \cdot (d-1))$.
* **Execution Steps:**
  1. Superimpose goal wavefunctions: $\psi_{\mathrm{total}} = \sum_k w_k \psi_k$.
  2. Compute optical geodesic gradient: $\mathbf{w}_{\mathrm{geodesic}}^{(L)} = \lambda \sigma^2 E(\boldsymbol{\mu}^{(L)})^\top \boldsymbol{\eta}_{\mathcal{L}} \frac{\nabla \psi_{\mathrm{total}}}{\psi_{\mathrm{total}}}$.
  3. Calculate polyhedral AIXI safety margin $\varepsilon_{\mathrm{safe}}$ and pressure vector $\mathbf{m}_{\mathrm{poly}}(a^*)$.
  4. Synthesize target motor action:

$$
\mathbf{a}_{\mathrm{target}} = \tanh\left( \mathbf{W}_{\mathrm{motor}} \mathbf{w}_{\mathrm{geodesic}}^{(L)} + \mathbf{W}_{\mathrm{poly}} \mathbf{m}_{\mathrm{poly}}(a^*) + \mathbf{W}_{\mathrm{social}} \mathbf{a}_{\mathrm{ToM}}^* + \mathbf{b}_{\mathrm{motor}} \right)
$$

  5. Evaluate motor EFE gradient: KL divergence to target priors plus causal intervention $\mathcal{I}_{\mathrm{Pearl}}^{\mathrm{causal}}$ via projector $\mathbf{P}_{\mathbf{a}^\perp}$.
  6. Integrate velocity: $\frac{\mathrm{d}\mathbf{a}}{\mathrm{d}t} = -\eta_a \nabla_{\mathbf{a}} \mathrm{EFE}$.
  7. Apply Euler integration and continuous clamping: $\mathbf{a}(t+\Delta t) = \mathrm{Softclip}_{[-1, 1]}(\mathbf{a}(t) + \Delta t \frac{\mathrm{d}\mathbf{a}}{\mathrm{d}t})$.

---

### Algorithm 6: Gödelian Ego-Reflection via Adjoint Resolvents
* **Inputs:** Historical microstate stream, gradient-detached coordinates $\mathrm{sg}[\boldsymbol{\mu}]$, motor vector $\mathbf{a}(t-\Delta t)$.
* **Outputs:** Updated weight tensors $\mathbf{W}_{\mathrm{self}, v}, \mathbf{W}_{\mathrm{self}, \mathrm{hist}}$, introspection trajectory $\overline{\mathcal{T}}_{\mathrm{introspect}}$.
* **Complexity:** $O((d-1)^3 + (d-1)^2 D_{\mathrm{hist}})$.
* **Execution Steps:**
  1. Assemble historical macro-vector: $\text{feat}_{\mathrm{hist}}(t) \in \mathbb{R}^{D_{\mathrm{hist}}}$.
  2. Solve linear resolvent system:

$$
((1+\delta_{\mathrm{reg}})\mathbf{I} - \mathbf{W}_{\mathrm{self}, v}) \cdot v_{\mathrm{self}}^* = \mathbf{W}_{\mathrm{self}, \mathrm{hist}} \text{feat}_{\mathrm{hist}} + \mathbf{b}_{\mathrm{self}}
$$

  3. Lift to layer $L$ barycenter tangent space: $\mathbf{v}_{\mathrm{ambient}}^* = E(\overline{\boldsymbol{\mu}}^{(L)}) v_{\mathrm{self}}^*$.
  4. Execute ego-identity exponential step: $\boldsymbol{\mu}_{\mathrm{self}}^* = \mathrm{Exp}_{\overline{\boldsymbol{\mu}}^{(L)}}(\mathbf{v}_{\mathrm{ambient}}^*)$.
  5. Update introspection trajectory $\overline{\mathcal{T}}_{\mathrm{introspect}}$ with geodesic momentum $\gamma_{\mathrm{ema}}$.
  6. Compute residual gradient: $\mathbf{g}_{\mathrm{self}} = E(\overline{\mathcal{T}})^\top \boldsymbol{\eta}_{\mathcal{L}} \mathrm{Log}_{\overline{\mathcal{T}}}(\boldsymbol{\mu}_{\mathrm{self}}^*)$.
  7. Parallel transport $\mathbf{g}_{\mathrm{self}}$ to layer $L$ barycenter to form $\tilde{\mathbf{g}}$.
  8. Perform gradient descent scaled by adjoint resolvent $((1+\delta_{\mathrm{reg}})\mathbf{I} - \mathbf{W}_{\mathrm{self}, v}^\top)^{-1}$.
  9. Apply spectral norm clamping: $\sigma_{\max} \le 1 - \epsilon_{\mathrm{gap}}$.

---

### Algorithm 7: Riemannian K-FAC Natural Gradient Update ($\epsilon_1 \sim 1\text{ s}$)
* **Inputs:** Error sensitivity vectors $\tilde{\boldsymbol{\varepsilon}}_i$, perceptron activation matrices, Augmented Lagrangian gradients.
* **Outputs:** Updated synaptic tensors $\mathbf{W}_i^{(1, l)}, \mathbf{W}_i^{(2, l)}$.
* **Complexity:** $O(|V| \cdot (h_{\mathrm{dim}}^3 + (d-1)^3))$.
* **Execution Steps:**
  1. Parallel transport local gradient $\mathbf{g}_i^{(2, \mathrm{local})}$ to layer barycenter: $\mathbf{g}_i^{(2, l)} = \mathbf{P}_{\boldsymbol{\mu}_i \to \overline{\boldsymbol{\mu}}}(\mathbf{g}_i^{(2, \mathrm{local})})$.
  2. Backpropagate sensitivity: $\mathbf{g}_i^{(1, l)} = ((\mathbf{W}_i^{(2)})^\top \mathbf{g}_i^{(2)}) \odot \sigma'$.
  3. Update running Kronecker factors $\tilde{\mathbf{A}}^{(1)}, \tilde{\mathbf{B}}^{(1)}, \tilde{\mathbf{A}}^{(2)}, \tilde{\mathbf{B}}^{(2)}$ with momentum $\eta_k$.
  4. Adiabatic decimation check: step executes strictly every $K_{\mathrm{infer}} = 4$ inference ticks.
  5. Invert Tikhonov-damped factors: $(\tilde{\mathbf{A}} + \delta_{\mathrm{kfac}}\mathbf{I})^{-1}$ and $(\tilde{\mathbf{B}} + \delta_{\mathrm{kfac}}\mathbf{I})^{-1}$.
  6. Synthesize natural gradient matrix tensor: $\Delta \mathbf{W} = \tilde{\mathbf{B}}^{-1} \nabla \mathbf{W} \tilde{\mathbf{A}}^{-1}$.
  7. Clamp Frobenius norm: $\|\Delta \mathbf{W}\|_F \le M_{\mathrm{step}} = 1.0$.
  8. Update synaptic weights: $\mathbf{W} \leftarrow \mathbf{W} - \frac{\Delta t}{\epsilon_1} \Delta \mathbf{W}_{\mathrm{clipped}}$.

---

### Algorithm 8: Criticality Regulation & SoftVQ Codebook Dynamics
* **Inputs:** Prediction error distribution $\tilde{\boldsymbol{\varepsilon}}_i$ across all network neurons.
* **Outputs:** Dynamic damping coefficient $\alpha_{\mathrm{damp}}(t+\Delta t)$, updated codebook $\mathcal{C}$.
* **Complexity:** $O(|V| + |\mathcal{C}| \cdot (d-1))$.
* **Execution Steps:**
  1. Identify avalanche vertices $\mathcal{S}_{\mathrm{tail}}$ where $\|\tilde{\boldsymbol{\varepsilon}}_i\|_{\tilde{\mathbf{\Pi}}} \ge \varepsilon_{\min}$.
  2. If $|\mathcal{S}_{\mathrm{tail}}| \ge N_{\min}$, smooth log-tail mean:

$$
\overline{L}_{\mathrm{tail}}(t) = (1 - \gamma_{\mathrm{soc}}) \overline{L}_{\mathrm{tail}}(t-\Delta t) + \gamma_{\mathrm{soc}} \mathrm{mean}_{i \in \mathcal{S}_{\mathrm{tail}}} \ln(\|\tilde{\boldsymbol{\varepsilon}}_i\| / \varepsilon_{\min})
$$

  3. Compute instantaneous Pareto exponent: $\hat{\alpha}_{\mathrm{measured}} = 1 + 1/\overline{L}_{\mathrm{tail}}$.
  4. Calculate damping adjustment: $\Delta \alpha = \mathrm{clip}(\Delta t \cdot 1.2 \eta_\alpha (1.5 - \hat{\alpha}_{\mathrm{measured}}), -\Delta \alpha_{\max}, +\Delta \alpha_{\max})$.
  5. Update damping with lower bound: $\alpha_{\mathrm{damp}} \leftarrow \max(\alpha_{\min}, \alpha_{\mathrm{damp}} + \Delta \alpha)$.
  6. Apply exponential decay to token usage: $u_k \leftarrow \gamma_{\mathrm{usage}} u_k$.
  7. Prune tokens below $\theta_{\mathrm{prune}} = 0.05$ outside $\mathcal{K}_{\mathrm{protected}}$, charging Landauer penalty $\mathcal{L}_{\mathrm{Landauer}}$.
  8. If prediction error exceeds $\theta_{\mathrm{spawn}}$, spawn a new concept centroid projected from the pole.

---

### Algorithm 9: Sleep Replay Consolidation & Anti-Livelock State Rollback
* **Inputs:** Episodic memory buffer $\mathcal{M}_{\mathrm{fast}}$, free energy derivative $\dot{\bar{\mathcal{J}}}(t)$, VRAM ring buffer.
* **Outputs:** Re-consolidated synaptic tensors or restored checkpoint state.
* **Complexity:** $O(K_{\mathrm{clusters}} \cdot |\mathcal{B}_{\mathrm{replay}}|)$ during sleep, $O(\mathrm{Size}(\text{Snapshot}))$ during rollback.
* **Execution Steps:**
  1. **Sleep Phase (Sleep Replay):**
     * Uniformly sample episodic memory clusters using Vitter's reservoir algorithm.
     * Execute state reconstruction through projection perceptrons.
     * Perform natural gradient descent with step-size $-\eta_{\mathrm{replay}}$.
  2. **Wake Phase (Anti-Livelock Watchdog):**
     * Evaluate running free energy derivative $\dot{\bar{\mathcal{J}}}(t)$.
     * If $\dot{\bar{\mathcal{J}}} > \theta_{\mathrm{impasse}}$ and cooldown timer $t - t_{\mathrm{last\_reset}} \ge \tau_{\mathrm{cooldown}}$:
       * Restore full neuron microstates and K-FAC factors from checkpoint $K_{\mathrm{check}} = 5$ in VRAM ring buffer.
       * Re-ground Dirichlet sensory layer with current environmental stream.
       * Anneal Langevin temperature: $\mathbf{T}_i \leftarrow T_{\max} \cdot (0.8)^N$.
       * Double cooldown timer: $\tau_{\mathrm{cooldown}} \leftarrow \min(\tau_{\max}, 2.0 \cdot \tau_{\mathrm{cooldown}})$.
       * Increment failure counter $N_{\mathrm{consecutive\_resets}}$.
     * If $\dot{\bar{\mathcal{J}}} \le 0$, reset failure counter to zero and restore base cooldown $\tau_{\mathrm{cooldown}, 0}$.

---

### Algorithm 10: System-1 Semantic Query Relaxation & Holographic Retrieval
* **Inputs:** Query probe $\mathbf{x}_{\mathrm{query}}$, holographic phase memory $\mathbf{H}_{\mathrm{hologram}}$.
* **Outputs:** Semantic target action response $\mathbf{a}_{\mathrm{target}}$.
* **Complexity:** $O(K_{\mathrm{relax}} \cdot |E| + M_{\mathrm{capacity}} \cdot (d-1))$.
* **Execution Steps:**
  1. Pin $\mathbf{x}_{\mathrm{query}}$ to Dirichlet sensory layer $\boldsymbol{\mu}_0^{(0)}$.
  2. Lock synaptic adaptation: $\Delta \mathbf{W} \equiv \mathbf{0}, \Delta E \equiv 0$.
  3. Execute zero-inertia System-1 relaxation: perform $K_{\mathrm{relax}} = 4$ consecutive geodesic steps driven by bottom-up forces $\tilde{\mathbf{v}}_{\mathrm{bottom-up}}$.
  4. Reach stationary state at top layer $\boldsymbol{\mu}^{(L)}$.
  5. Compute Harish-Chandra spherical convolution: evaluate phase interference between $\boldsymbol{\mu}^{(L)}$ and spectral modes $\tilde{A}_m$ weighted by Plancherel measure $|c(\lambda)|^{-2}$.
  6. Synthesize associative recall vector $\mathbf{H}_{\mathrm{recall}}$.
  7. Project stationary state and associative context through motor decoder to yield $\mathbf{a}_{\mathrm{target}}$.

---

### Algorithm 11: Anytime Polyhedral Levin Decider
* **Inputs:** Current bounds $Q_{\min}(a)$, $Q_{\max}(a)$, Kraft unresolved tail $\Omega_{\mathrm{unresolved}}(k)$.
* **Outputs:** Optimal action $a^*$ and safety radius $\varepsilon_{\mathrm{safe}}$.
* **Complexity:** $O(|\mathcal{A}|)$ per iteration with anytime interruptibility.
* **Execution Steps:**
  1. Update lower and upper bounds $Q_{\min}(a)$, $Q_{\max}(a)$ as candidate programs halt in Levin's interleaved search.
  2. Evaluate decision fixation boundary: $Q_{\min}(a^*) > \max_{a \neq a^*} Q_{\max}(a)$.
  3. If condition holds: fixate $a^*$ and compute polyhedral safety margin $\varepsilon_{\mathrm{safe}}$.
  4. Otherwise: allocate additional time slices to active programs in the search queue.
  5. Return $a^*$ and $\varepsilon_{\mathrm{safe}}$ to the motor synthesis controller.

---

### Algorithm 12: Holographic Memory Retrieval via Harish-Chandra Spherical Transforms
* **Inputs:** Query coordinate $\boldsymbol{\mu}_i^{(l)}$, memory buffer $\mathcal{M}_{\mathrm{fast}}$, spectral amplitudes $\tilde{A}_m(\lambda, \mathbf{b})$.
* **Outputs:** Associative context recall vector $\mathbf{H}_{\mathrm{recall}}(\boldsymbol{\mu}_i^{(l)})$.
* **Complexity:** $O(M_{\mathrm{capacity}} \cdot (d-1))$.
* **Execution Steps:**
  1. Evaluate Gelfand-Graev spherical kernels $\phi_{\lambda, \mathbf{b}}(\boldsymbol{\mu}_i^{(l)})$ across sampled frequency-angle pairs $(\lambda, \mathbf{b})$.
  2. Compute geodesic phase multipliers: $\exp(-i \lambda d_{\mathcal{H}^d}(\boldsymbol{\mu}_i^{(l)}, \boldsymbol{\mu}_m) / \hbar_{\mathrm{cog}})$.
  3. Integrate spherical convolution with amplitudes $\tilde{A}_m(\lambda, \mathbf{b})$ under Plancherel density $|c(\lambda)|^{-2}$.
  4. Return associative vector $\mathbf{H}_{\mathrm{recall}}$ to the inference loop.

---

### Algorithm 13: Asynchronous Zero-Downtime Gödel JIT Engine Hot-Swapping
* **Inputs:** Active engine $F_{\mathrm{old}}$, candidate engine $F_{\mathrm{new}}$, expected utility evaluators $\mathbb{E}[U(\cdot)]$.
* **Outputs:** Atomically swapped active execution engine.
* **Complexity:** $O(1)$ at swap point.
* **Execution Steps:**
  1. Verify formal Gödelian proof condition: $\mathcal{P} \vdash [\mathbb{E}[U(F_{\mathrm{new}})] > \mathbb{E}[U(F_{\mathrm{old}})]]$.
  2. If proof verifies: execute atomic pointer swap: `atomic_store(&ActiveEngine, Engine_new)`.
  3. Maintain double-buffering execution to ensure zero pipeline stalls ($\Delta t = 0$).

---

## Part III: Complete Mathematical & Symbol Index

---

### Geometric Manifolds, Bundles, and Sets
* $\mathcal{H}^d$: Lorentz hyperboloid manifold (upper sheet of the two-sheeted hyperboloid in Minkowski space $\mathbb{R}^{1, d-1}$):

$$
\mathcal{H}^d = \left\{ \mathbf{x} \in \mathbb{R}^d \;\middle|\; \langle \mathbf{x}, \mathbf{x} \rangle_{\mathcal{L}} = -x_0^2 + \sum_{k=1}^{d-1} x_k^2 = -1, \quad x_0 > 0 \right\}
$$

* $T_{\boldsymbol{\mu}}\mathcal{H}^d$: Tangent vector space orthogonal to $\boldsymbol{\mu}$ under the Minkowski bilinear form:

$$
T_{\boldsymbol{\mu}}\mathcal{H}^d = \left\{ \mathbf{v} \in \mathbb{R}^d \;\middle|\; \langle \boldsymbol{\mu}, \mathbf{v} \rangle_{\mathcal{L}} = 0 \right\}
$$

* $T_{\boldsymbol{\mu}}^*\mathcal{H}^d$: Cotangent space (space of canonical momenta 1-forms) dual to $T_{\boldsymbol{\mu}}\mathcal{H}^d$.
* $T^*\mathcal{H}^d$: Cotangent bundle representing canonical position-momentum phase space:

$$
T^*\mathcal{H}^d = \bigcup_{\boldsymbol{\mu} \in \mathcal{H}^d} \left( \{\boldsymbol{\mu}\} \times T_{\boldsymbol{\mu}}^*\mathcal{H}^d \right)
$$

* $T_{\mathbf{o}}\mathcal{H}^d$: Tangent space at the canonical origin $\mathbf{o} = (1, 0, \dots, 0)^\top$, isomorphic to $\mathbb{R}^{d-1}$.
* $\mathbb{S}^{d-2}$: Unit Euclidean sphere parameterizing directional vectors in the hyperboloid metric factorization.
* $\Delta^{|\mathcal{O}|-1}$: Probability simplex over the alphabet of environmental observations $\mathcal{O}$.
* $\mathcal{P}^*$: Polyhedral decision cell in the observation simplex inside which optimal actions remain invariant.
* $\mathcal{M}_i^{(l)}$: Riemannian configuration manifold of the full neuron microstate.
* $\mathcal{C}$: SoftVQ codebook of spatial centroids defined in the pole tangent space.
* $\mathcal{C}_{\mathrm{active}}$: Active subset of codebook centroids ($m_{\mathrm{pruned}} = 1$).
* $\mathcal{K}_{\mathrm{protected}}$: Protected semantic token indices exempt from pruning operations.
* $\mathcal{B}$: Mini-batch of environmental observation snapshots.
* $\mathcal{M}_{\mathrm{fast}}$: Rapid episodic memory buffer partitioned into stratified reservoirs $\mathcal{M}_k$.
* $\mathcal{S}_{\mathrm{tail}}$: Set of avalanche neurons whose precision-weighted error exceeds threshold $\varepsilon_{\min}$.
* $V^{(l)}$: Set of neurons comprising layer $l$ of the hierarchy.
* $\mathrm{parents}(i)$: Set of parent neurons in layer $l+1$ providing context to neuron $i$.
* $\mathrm{children}(i)$: Set of child neurons in layer $l-1$ providing bottom-up prediction errors to neuron $i$.

---

### Indices, Dimensions, and Architectural Constants
* $l \in \{0, 1, \dots, L\}$: Hierarchical layer index ($0$ is Dirichlet sensory layer, $L$ is ego-reflection layer).
* $i \in V^{(l)}$: Unique neuron index in layer $l$.
* $j \in \mathrm{parents}(i)$: Parent neuron index in layer $l+1$.
* $k \in \mathrm{children}(i)$: Child neuron index in layer $l-1$.
* $d = 16$: Ambient Minkowski dimension ($\mathbb{R}^{16}$).
* $r = 4$: Factorization rank of the DLR error covariance subspace ($r \ll d-1$).
* $M_{\mathrm{crossbar}} = 64$: Physical neuromorphic memristive crossbar lines.
* $M_{\mathrm{trace}} = 32$: Number of Rademacher vectors in Hutchinson's stochastic trace estimator.
* $h_{\mathrm{dim}} = 32$: Hidden dimension of projection perceptrons.
* $d_{\mathrm{act}} = 8$: Continuous motor action degrees of freedom.
* $d_{\mathrm{scm}} = 16$: Vertices in the structural causal model graph.
* $D_{\mathrm{hist}} = 128$: Historical macro-state feature dimension.
* $K_{\mathrm{relax}} = 4$: Fast System-1 relaxation steps for semantic inference queries.
* $K_{\mathrm{infer}} = 4$: Fast inference steps per single K-FAC natural gradient synaptic update.

---

### State Variables and Dynamic Vector Fields
* $\mathbf{\Xi}_i^{(l)}(t)$: Full microstate supervector of neuron $i$ in layer $l$ ($16\,768\text{ bytes}$).
* $\mathbf{\Gamma}_i^{(l)}(t) = (\boldsymbol{\mu}_i^{(l)}, \mathbf{p}_i^{(l)})$: Canonical phase point in $T^*\mathcal{H}^d$ ($256\text{ bytes}$ FP64).
* $\boldsymbol{\mu}_i^{(l)}(t) \in \mathcal{H}^d$: Coordinate vector on the Lorentz hyperboloid ($128\text{ bytes}$ FP64).
* $\mathbf{p}_i^{(l)}(t) \in T_{\boldsymbol{\mu}_i^{(l)}}^*\mathcal{H}^d$: Canonical momentum vector ($128\text{ bytes}$ FP64).
* $\overline{\boldsymbol{\mu}}^{(l)}(t) \in \mathcal{H}^d$: Running Fréchet barycenter of layer $l$ ($128\text{ bytes}$ FP64).
* $\boldsymbol{\mu}_{\mathrm{pred}, i}^{(l)}(t) \in \mathcal{H}^d$: Downward top-down prediction coordinate on the manifold.
* $\boldsymbol{\mu}_{\mathrm{self}}^*(t) \in \mathcal{H}^d$: Instantaneous ego-identity coordinate of the Gödelian operator.
* $\overline{\mathcal{T}}_{\mathrm{introspect}}(t) \in \mathcal{H}^d$: Running continuous introspection trajectory.
* $\boldsymbol{\mu}_0^{(0)}(t) \in \mathcal{H}^d$: Pinned Dirichlet sensory input vector.
* $\mathbf{v}_i^{(l)}(t) \in T_{\boldsymbol{\mu}_i}\mathcal{H}^d$: Physical tangent velocity vector on the hyperboloid.
* $\mathbf{w}_i^{(l)}(t) \in \mathbb{R}^{d-1}$: Inference velocity in the local orthonormal tetrad frame ($120\text{ bytes}$ FP64).
* $\mathbf{w}_{\mathrm{det}, i}^{(l)}(t), \mathbf{w}_{\mathrm{raw}, i}^{(l)}(t)$: Deterministic and raw unclipped stochastic velocity registers ($120\text{ bytes}$ FP64).
* $\mathbf{w}_{\mathrm{geodesic}, i}^{(l)}(t)$: Optical geodesic velocity vector derived from the Eikonal action gradient.
* $\mathbf{w}_{\mathrm{relax}, i}^{(l)}$: Inertia-free System-1 relaxation velocity during semantic queries.
* $\tilde{\mathbf{v}}_{\mathrm{forces}, i}^{(l)}(t)$: Total internal inference force vector in local frame.
* $\tilde{\mathbf{v}}_{\mathrm{drift\_Ito}, i}^{(l)}(t)$: Temperature-scaled Itô-to-Stratonovich drift correction vector.
* $\tilde{\mathbf{v}}_{\mathrm{conformal}, i}^{(l)}(t)$: Fictitious conformal drift force vector.
* $\tilde{\mathbf{v}}_{\mathrm{bottom-up}, i}^{(l)}(t)$: Bottom-up prediction error forces from child neurons in layer $l-1$.
* $v_{\mathrm{self}}^*(t) \in \mathbb{R}^{d-1}$: Fixed-point coordinates of the ego-operator in the top layer barycenter frame.
* $\boldsymbol{\varepsilon}_i^{(l)}(t) \in T_{\boldsymbol{\mu}_i}\mathcal{H}^d$: Tangent prediction error vector between state and prediction.
* $\tilde{\boldsymbol{\varepsilon}}_i^{(l)}(t) \in \mathbb{R}^{d-1}$: Local frame prediction error vector ($60\text{ bytes}$ FP32).
* $\mathbf{a}(t) \in [-1, 1]^{d_{\mathrm{act}}}$: Continuous motor action vector ($32\text{ bytes}$ FP32).
* $\mathbf{a}_{\mathrm{target}}(t)$: Target motor action synthesized by the wave and ToM modules ($32\text{ bytes}$ FP32).
* $\mathbf{a}_{\mathrm{counterfactual}}(t)$: Counterfactual motor action with regularized orthogonal perturbation.
* $\mathbf{c}_k \in \mathbb{R}^{d-1}$: SoftVQ concept centroids in the pole tangent space ($\mathbf{c}_0 = \mathbf{0}$).
* $m_{\mathrm{pruned}, k} \in \{0, 1\}$: Centroid activity boolean mask ($0$ is pruned, $1$ is active; $m_{\mathrm{pruned}, 0} \equiv 1$).
* $u_k(t) \in [0, \infty)$: Exponential token usage frequency counter (FP32).
* $\text{feat}_{\mathrm{hist}}(t)$: Historical macro-feature representation vector ($512\text{ bytes}$ FP32).
* $\tilde{\mathbf{d}}_i^{(l)} \in \mathbb{R}^{d-1}$: Diagonal residual error variance vector ($60\text{ bytes}$ FP32).
* $\mathbf{s}_i^{(l)} \in \mathbb{R}^r$: Principal singular values of error covariance ($16\text{ bytes}$ FP32).
* $\mathbf{s}_i(t) \in \{0, 1\}^{M_{\mathrm{crossbar}}}$: Neuromorphic discrete spike vector ($64\text{ bytes}$ UINT8).
* $\lambda_{\mathrm{spike}, i}^{(l)}(t)$: Neuromorphic spike generation firing rate ($256\text{ bytes}$ FP32).
* $\tilde{\mathbf{a}}_i^{(l)}$: Context vector assembled from parent neurons in the pole tangent space.
* $\mathbf{h}_i^{(l)}$: Activation vector of the perceptron hidden layer ($128\text{ bytes}$ FP32).
* $\mathbf{u}_{2, i}^{(l)}, \mathbf{u}_{2, \mathrm{safe}, i}^{(l)}$: Perceptron output vector before and after Euclidean retraction ($60\text{ bytes}$ FP32).
* $\mathbf{g}_i^{(1, l)}, \mathbf{g}_i^{(2, l)}$: Perceptron gradient sensitivity vectors in layer barycenter frame.
* $\mathbf{b}_{\mathrm{self}}(t) \in \mathbb{R}^{d-1}$: Dynamic bias vector of the Gödelian ego-reflection engine ($60\text{ bytes}$ FP32).
* $\mathbf{m}_{\mathrm{poly}}(a^*) \in \mathbb{R}^{d_{\mathrm{act}}}$: Polyhedral AIXI decision pressure vector ($32\text{ bytes}$ FP32).
* $\mathbf{T}, \mathbf{Z}$: Real Schur form and orthogonal transformation buffers for bounding $\rho(\mathbf{A}_{\mathrm{SCM}})$.
* $\kappa_l(t)$: Adaptive sectional curvature of layer $l$ (Section 36).

---

### Scalar Fields, Potentials, and Wavefunctions
* $S(\boldsymbol{\mu}) \in \mathbb{R}$: Geodesic action field satisfying the viscous Riemannian Eikonal PDE (FP64).
* $\psi(\boldsymbol{\mu}) \in \mathbb{R}^+$: Value diffusion wavefunction under the Cole-Hopf transformation (FP64).
* $\psi_{\mathrm{total}}(\boldsymbol{\mu})$: Linear superposition wavefunction across multiple conflicting goals (FP64).
* $n(\boldsymbol{\mu}) \in \mathbb{R}^+$: Optical refractive index of the mental phase space ($n = 1/\Omega$, FP64).
* $\Omega(\boldsymbol{\mu}) \in \mathbb{R}^+$: Conformal metric scaling factor derived from the free energy potential (FP64).
* $\mathcal{H}_{\mathrm{neuron}}(\boldsymbol{\mu}, \mathbf{p})$: Total microstate Hamiltonian energy functional (FP64).
* $\mathbf{H}_{\mathrm{hologram}}(\boldsymbol{\mu}) \in \mathbb{C}$: Complex-valued holographic memory interference pattern (Complex64).
* $\phi_{\lambda, \mathbf{b}}(\boldsymbol{\mu})$: Gelfand-Graev spherical functions of the Lorentz group $\mathrm{SO}^+(1, d-1)$.
* $|c(\lambda)|^{-2}$: Plancherel spectral density measure on the symmetric space $\mathcal{H}^d$.
* $\tilde{A}_m(\lambda, \mathbf{b})$: Spectral amplitude modes of episodic memories in the Harish-Chandra basis.

---

### Matrices, Tensors, and Differential Operators
* $E(\boldsymbol{\mu}_i^{(l)}) \in \mathbb{R}^{d \times (d-1)}$: Closed-form orthonormal Lorentz boost tetrad frame.
* $\boldsymbol{\eta}_{\mathcal{L}} = \mathrm{diag}(-1, 1, \dots, 1) \in \mathbb{R}^{d \times d}$: Minkowski metric tensor of signature $(-, +, \dots, +)$.
* $\tilde{g}_{AB}(\boldsymbol{\mu}) \in \mathbb{R}^{d \times d}$: Maupertuis-Jacobi conformal metric tensor.
* $\tilde{\Gamma}_{BC}^A$: Christoffel symbols of the second kind on the conformal manifold.
* $\omega \in \Lambda^2(T^*\mathcal{H}^d)$: Canonical symplectic 2-form of the cotangent phase space.
* $\Delta_{\mathcal{H}^d}$: Laplace-Beltrami differential operator on the Lorentz hyperboloid.
* $\nabla_{\mathcal{H}^d}$: Riemannian gradient operator: $\nabla_{\mathcal{H}^d} f = \boldsymbol{\eta}_{\mathcal{L}}^{-1} \mathrm{d}f + \langle \boldsymbol{\mu}, \boldsymbol{\eta}_{\mathcal{L}}^{-1} \mathrm{d}f \rangle_{\mathcal{L}} \boldsymbol{\mu}$.
* $\tilde{\mathbf{\Sigma}}_i^{(l)} \in \mathbb{R}^{(d-1) \times (d-1)}$: Fisher-Rao local prediction error covariance matrix ($900\text{ bytes}$ FP32).
* $\tilde{\mathbf{\Pi}}_i^{(l)} = (\tilde{\mathbf{\Sigma}}_i^{(l)})^{-1}$: Full local precision matrix inverted via Woodbury identity ($900\text{ bytes}$ FP32).
* $\tilde{\mathbf{V}}_i^{(l)} \in \mathbb{R}^{(d-1) \times r}$: Orthonormal basis of principal error covariance modes ($240\text{ bytes}$ FP32).
* $\tilde{\mathbf{A}}_i^{(1, l)}, \tilde{\mathbf{B}}_i^{(1, l)}$: Input and gradient Kronecker factors for K-FAC natural gradient update (FP32).
* $\mathbf{P}_{\mathbf{x} \to \mathbf{y}}$: Isometric parallel transport operator along geodesics.
* $\mathbf{P}_{\mathbf{a}^\perp}$: Tikhonov-regularized projection operator onto the subspace orthogonal to actions.
* $\mathbf{W}_i^{(1, l)}, \mathbf{W}_i^{(2, l)}$: Weight matrices of the two-layer projection perceptron (FP32).
* $\mathbf{b}_i^{(1, l)}, \mathbf{b}_i^{(2, l)}$: Bias vectors of the projection perceptron (FP32).
* $\mathbf{W}_{\mathrm{self}, v}, \mathbf{W}_{\mathrm{self}, \mathrm{hist}}$: Trainable weight matrices of the Gödelian ego-reflection engine (FP32).
* $\mathbf{A}_{\mathrm{SCM}}$: Differentiable adjacency matrix of the structural causal model ($1024\text{ bytes}$ FP32).
* $\mathbf{W}_{\mathrm{motor}}, \mathbf{b}_{\mathrm{motor}}$: Variational motor kinematic decoder parameters (FP32).
* $\mathbf{W}_{\mathrm{social}}$: Projection matrix coupling Theory of Mind vectors to physical motor action (FP32).
* $\mathbf{W}_{\mathrm{poly}}$: Projection matrix mapping polyhedral AIXI pressure to motor degrees of freedom (FP32).
* $\mathbf{W}_{\mathrm{key}}, \mathbf{W}_{\mathrm{val}}$: Key and value projection matrices of hyperbolic associative attention (FP32).
* $\mathbf{W}_{\mathrm{analog\_crossbar}} \in \mathbb{R}^{(d-1) \times M_{\mathrm{crossbar}}}$: Memristive crossbar conductance matrix ($3840\text{ bytes}$ FP32).
* $\mathbf{W}_{\mathrm{enc}}, \mathbf{b}_{\mathrm{enc}}$: Neuromorphic straight-through spike encoder parameters (FP32).
* $\mathbf{I}_{d-1}$: Identity matrix of dimension $(d-1) \times (d-1)$.
* $\mathbf{e}_k$: Standard Cartesian basis unit vector in Euclidean subspace.

---

### Hyperparameters, Coefficients, and Multipliers
* $t$: Continuous physical system time (s).
* $\Delta t = 0.01\text{ s}$: Numerical integration time step.
* $\epsilon_0, \epsilon_1, \epsilon_2$: Physical timescales ($\epsilon_0 \sim 10^{-3}\text{ s}$, $\epsilon_1 \sim 1\text{ s}$, $\epsilon_2 \sim 10^5\text{ s}$).
* $\lambda = 1.0$: Thermodynamic control regularization parameter in the Cole-Hopf scheme.
* $\sigma = 0.1$: Wavefunction stochastic diffusion coefficient.
* $\hbar_{\mathrm{cog}} = 1.0$: Cognitive quantum of action in the holographic phase interference operator.
* $\mathcal{F}_{\mathrm{threshold}} = 100.0$: Energy threshold mapping free energy to the conformal factor.
* $\lambda_{\mathrm{Eik}} = 0.5$: Weight of the Eikonal PDE residual penalty in the PHR functional.
* $\varepsilon_{\mathrm{visc}}$: Laplace-Beltrami viscosity coefficient regularizing the Eikonal equation.
* $\mathcal{L}_{\mathrm{Landauer}}$: Landauer thermal dissipation penalty for pruned vocabulary tokens.
* $k_B$: Boltzmann constant.
* $\varepsilon_{\mathrm{safe}}$: Analytical safety radius of the polyhedral AIXI decision cell.
* $\lambda_{\mathrm{mid}}, \mu_{\mathrm{mid}}$: Scalar Lagrange multipliers for the symplectic implicit midpoint integrator.
* $A_m, B_m, C_m$: Local quadratic coefficients for the Godunov upwind scheme.
* $R_{\mathrm{pred\_max}} = 3.0$: Maximum radius of tangent Euclidean retraction.
* $M_{\mathrm{step}} = 1.0$: Frobenius norm clipping limit for synaptic natural gradient tensors.
* $\gamma_{\mathrm{anneal}} \approx 0.95$: Epoch-wise Langevin noise annealing factor.
* $r_i^{(l)}(t)$: Chordal distance from neuron to its layer barycenter.
* $f_{\mathrm{drift\_safe}}(r)$: Kinematic Laplace-Beltrami drift smoothing function.
* $\pi_i^{(l)}(t)$: Scalar variational prediction precision (FP32).
* $\pi_{\min} = 10^{-4}, \pi_{\max} = 10^4$: Clamping bounds for scalar variational precision.
* $E_{ji}^{(l+1)}(t) \in [E_{\min}, 1.0]$: Synaptogenic inter-layer connection weight ($E_{\min} = 0.01$).
* $v_{\max} = 5.0$: Maximum velocity threshold on the hyperboloid.
* $\delta_{\mathrm{eps}} = 10^{-12}$: Machine numerical zero-division guard.
* $\delta_{\mathrm{reg}} = 10^{-6}$: Tikhonov regularizer for the Gödelian resolvent operator.
* $\delta_{\mathrm{proj}} = 10^{-5}$: Regularization constant for counterfactual action projection.
* $\delta_{\mathrm{soc}} = 10^{-6}$: Numerical guard for the SOC criticality regulator.
* $\delta_{\mathrm{scm}} = 10^{-3}$: Resolvent stabilization constant for the NOTEARS model.
* $\delta_{\mathrm{kfac}} = 0.2$: Tikhonov damping factor for inverting Kronecker covariance factors.
* $\delta_{\min} = 10^{-5}$: Regularization factor for DLR covariance tracking.
* $\epsilon_{\mathrm{gap}} = 0.05$: Spectral stability gap for $\mathbf{W}_{\mathrm{self}, v}$ and $\mathbf{A}_{\mathrm{SCM}}$.
* $\beta = 5.0$: Steepness coefficient of the sigmoidal spike mask.
* $\tau_i^{(l)}$: Local energy threshold for neuronal spike emission.
* $\hat{\alpha}_{\mathrm{measured}}(t)$: Measured Pareto tail exponent of the SOC criticality regulator.
* $\gamma_{\mathrm{soc}} = 0.05$: Exponential moving average smoothing factor for SOC tracking.
* $\alpha_{\mathrm{damp}}(t)$: Dynamic viscous damping coefficient.
* $\alpha_{\min} = 0.05$: Lower bound on damping ensuring asymptotic Lyapunov dissipativity.
* $\Delta \alpha_{\max} = 0.01$: Maximum allowable slew rate per step for the damping coefficient.
* $\mathbf{T}_i \in [0, T_{\max}]$: Thermodynamic Langevin temperature ($T_{\max} = 1.0$, FP32).
* $T_{\mathrm{vq}} = 0.1$: Softmax temperature for SoftVQ attention.
* $T_{\mathrm{attn}} = 0.5$: Attention temperature for associative memory retrieval.
* $R_{\mathcal{H}} = 1.0$: Radius of curvature of the hyperbolic manifold.
* $\eta_E = 0.01, \lambda_E = 0.001$: Hebbian growth rate and decay coefficient for graph topology.
* $\eta_m = 0.05, \eta_k = 0.01$: Exponential smoothing rates for DLR and K-FAC factors.
* $\eta_a = 0.05, \lambda_{\mathrm{policy}} = 0.1$: Learning rate and consistency weight for motor actuation.
* $\eta_\lambda = 0.01, \gamma_{\mathrm{prox}} = 0.1$: Adaptation rate and proximal smoothing for PHR multipliers.
* $\rho_k = 1.0$: Augmented Lagrangian quadratic penalty weight.
* $\gamma_{\mathrm{ema}} = 0.1$: Geodesic momentum factor for introspection trajectory tracking.
* $\gamma_{\mathrm{bary}} = 0.2$: Fréchet barycenter streaming filter step-size.
* $\tau_{\mathrm{dwell}} = 0.5\text{ s}$: Minimum continuous dwell time in Hespanha's stability theorem.
* $\mu_{\mathrm{jump}} = 0.1$: Maximum allowable Lyapunov jump factor under discrete switching events.
* $\sigma_{\max}(\mathbf{W})$: Maximum singular value of weight tensors.
* $\mu_{\mathrm{DAG}} = 1.0$: Penalty multiplier for NOTEARS acyclicity violations.
* $\theta_{\mathrm{impasse}} = 5.0$: Free energy derivative threshold triggering VRAM state rollback.
* $\tau_{\mathrm{cooldown}, 0} = 1.0\text{ s}$: Base cooldown timer for Anti-Livelock resets.
* $\tau_{\max} = 60.0\text{ s}$: Maximum allowable cooldown under exponential backoff.
* $N_{\mathrm{consecutive\_resets}}$: Counter of consecutive emergency VRAM rollbacks (INT32).
* $\theta_{\mathrm{spawn}} = 15.0$: Prediction error threshold triggering concept centroid spawning.
* $\theta_{\mathrm{prune}} = 0.05$: Usage threshold triggering codebook token pruning.
* $K_{\max} = 1024$: Maximum allowable codebook size for SoftVQ.
* $d_{\min} = 0.5$: Minimum spatial deduplication radius for codebook tokens.
* $K_{\mathrm{check}} = 5$: Depth of the VRAM checkpoint ring buffer.
* $f_{\max} = 1000.0\text{ Hz}$: Maximum clock frequency of the neuromorphic spike interface.
* $\kappa_v = 0.01$: Kinetic friction parameter in the PHR functional.

---

### Mathematical Operators and Notations
* $\langle \mathbf{x}, \mathbf{y} \rangle_{\mathcal{L}} = -x_0 y_0 + \sum_{k=1}^{d-1} x_k y_k$: Minkowski bilinear form of signature $(-, +, \dots, +)$.
* $\|\mathbf{v}\|_{\mathcal{L}} = \sqrt{\max(0.0, \langle \mathbf{v}, \mathbf{v} \rangle_{\mathcal{L}})}$: Minkowski seminorm for space-like vectors.
* $\|\mathbf{w}\|_2 = \sqrt{\sum_{k=1}^{d-1} w_k^2}$: Standard Euclidean norm in local orthonormal tetrad frame.
* $\|\mathbf{W}\|_F = \sqrt{\sum_{i, j} W_{ij}^2}$: Frobenius matrix norm.
* $d_{\mathcal{H}^d}(\mathbf{x}, \mathbf{y}) = 2 \mathrm{arcsinh}\left(0.5 \sqrt{\max(0, \langle \mathbf{x}-\mathbf{y}, \mathbf{x}-\mathbf{y} \rangle_{\mathcal{L}})}\right)$: Numerically stable chordal geodesic distance.
* $\mathrm{Exp}_{\boldsymbol{\mu}}^{\mathcal{H}^d}(\mathbf{v})$: Geodesic exponential mapping from $T_{\boldsymbol{\mu}}\mathcal{H}^d$ onto $\mathcal{H}^d$ with Taylor expansion for small velocities.
* $\mathrm{Log}_{\boldsymbol{\mu}}^{\mathcal{H}^d}(\mathbf{y})$: Riemannian logarithmic mapping projecting manifold coordinates into tangent vectors.
* $\mathbf{P}_{\mathbf{x} \to \mathbf{y}}(\mathbf{v})$: Isometric parallel transport operator along geodesics.
* $h_{\mathrm{log}}(\mathbf{A}) = \ln\left(\frac{1}{d} \mathrm{Tr}\left(\mathrm{expm}(\mathbf{A}\circ\mathbf{A})\right)\right)$: Smooth NOTEARS acyclicity constraint.
* $\mathrm{expm}(\mathbf{M}) = \sum_{k=0}^\infty \frac{\mathbf{M}^k}{k!}$: Matrix exponential.
* $\mathrm{Tr}(\mathbf{M})$: Trace of a square matrix.
* $\det(\mathbf{M})$: Determinant of a square matrix.
* $\rho(\mathbf{M})$: Spectral radius of a matrix (maximum eigenvalue modulus), evaluated via real Schur decomposition $\mathbf{M} = \mathbf{Z} \mathbf{T} \mathbf{Z}^\top$.
* $\mathbb{D}_{\mathrm{KL}}(Q \parallel P)$: Kullback-Leibler divergence between probability distributions.
* $\mathcal{F}_{\mathrm{pragmatic}}^{\mathrm{Sylvester}}$: Sylvester Pragmatic Free Energy functional summed from Dirichlet layer $l=0$ to layer $L$.
* $\mathcal{J}_{\mathrm{AGI\_Master}}$: Master Augmented Lagrangian (PHR) action functional with Landauer thermal dissipation.
* $V(\mathbf{\Xi})$: Unified Lyapunov-Hespanha dynamic stability candidate functional.
* $\odot$: Hadamard element-wise vector product.
* $\circ$: Hadamard element-wise matrix product.
* $\parallel$: Vector and tensor concatenation operator.
* $\wedge$: Grassmann exterior differential product.
* $\mathrm{sg}[\cdot]$: Gradient-stop (Stop-Gradient) operator.
* $\mathrm{Softclip}_{[a, b]}(x)$: Smooth saturation function bounding values within $[a, b]$.
* $\mathrm{clip}(x, a, b)$: Hard numerical clamping operator.
* $\sigma(x) = \frac{1}{1 + e^{-x}}$: Standard logistic sigmoid activation.
* $\sigma'(x) = \sigma(x)(1 - \sigma(x))$: First derivative of logistic sigmoid.
* $\mathrm{sech}(x) = \frac{2}{e^x + e^{-x}}$: Hyperbolic secant function.
* $\mathrm{sinhc}(x) = \frac{\sinh(x)}{x}$: Cardinal hyperbolic sine with Taylor series expansion around zero.
* $\delta(t - t_k)$: Dirac delta distribution.
* $\mathbb{I}(\text{cond})$: Indicator function ($1$ if condition is true, $0$ otherwise).
* $\mathrm{step}(x)$: Heaviside step function ($1$ if $x \ge 0$, $0$ otherwise).
* $\mathrm{QR\_Q}(\mathbf{M})$: Extraction of orthonormal basis via thin QR matrix decomposition.

---

## Conclusion & Architectural Parity
All equations in this specification are dimensionally consistent, Lorentz-invariant, and bounded. Continuous Weierstrass parameterization ensures that machine rounding errors cannot eject state trajectories from the hyperboloid manifold.

The complete dynamic kernel executes in $O(1)$ operations per moving entity per timestep, operating deterministically within real-time AAA game engine and robotic control budgets.
