# Runtime Implementation Status

This inventory describes the current Python runtime, not the validity of the mathematical proposals. `Implemented` means a corresponding runtime component exists; it does not mean every theoretical property has been independently proven. `Partial` means the runtime is a simplified subset. `Not implemented` means the named algorithm is not present as a runtime feature.

## Mathematical Sections

| Section | Status | Runtime scope |
|---|---|---|
| 1. Lorentz-bundle stochastic dynamics | Partial | Geodesic state updates exist; stochastic integration, layer barycenters, and the full precision/force loop are not integrated. |
| 2. Maupertuis-Jacobi conformal geometry | Partial | A conformal-drift helper exists; the conformal metric and connection are not integrated. |
| 3. Symplectic implicit midpoint | Partial | A projected kick-drift step exists; it is not an implicit midpoint solver and has no symplecticity guarantee. |
| 4. Cole-Hopf active inference | Partial | Cole-Hopf value transforms, goal superposition, and a positivity-limited explicit graph-diffusion step exist; there is no continuous HJB or Schrödinger-Beltrami solver. |
| 5. Viscous graph Eikonal | Partial | A finite-graph Godunov-style action solver uses hyperbolic edge distances and a graph viscosity term; it is not a general PDE solver. |
| 6. Solomonoff-AIXI decider | Partial | A finite, caller-supplied prefix-code model stream provides anytime lower/upper bounds; it does not execute universal programs or implement Levin search. |
| 7. Microstate and EFE motor loop | Partial | A small agent pipeline exists; the full microstate, expected-free-energy objective, and learned motor controller do not. |
| 8. Master augmented Lagrangian | Partial | A projected non-negative PHR inequality-multiplier update exists; the master objective and constraint gradients are not integrated. |
| 9. Sylvester pragmatic free energy | Partial | DLR covariance/precision tracking exists; the full free-energy functional is absent. |
| 10. Retraction perceptron and SoftVQ | Partial | A tangent-space nearest-centroid quantizer exists; the full soft assignment and perceptron training loop do not. |
| 11. Sigmoidal spike masking | Partial | Precision-weighted forward mask gates crossbar spikes; no straight-through backward estimator. |
| 12. DLR/Oja-QR and K-FAC | Partial | DLR and K-FAC primitives exist independently; no complete Riemannian training loop. |
| 13. Spectral Lipschitz regularization | Partial | A spectral-norm excess penalty is available as a standalone helper; it is not connected to weight training. |
| 14. Minkowski metric and geodesic operators | Partial | Core maps, metric operations, boost frame, and parallel transport have invariant tests; this does not implement every proposed geometric operation. |
| 15. Variational information bottleneck | Partial | Diagonal-Gaussian KL and a batch contrastive objective are implemented; learned encoders and a validated mutual-information estimator are absent. |
| 16. Causal NOTEARS | Partial | Acyclicity and a bounded adjacency update exist; counterfactual model training is absent. |
| 17. Active-inference target divergence | Partial | Categorical target KL is available as a standalone primitive; no predicted-state model or action-gradient integration is present. |
| 18. Multimodal sensory grounding | Partial | A single numeric vector can be grounded; modality encoders and contrastive grounding are absent. |
| 19. Gödelian ego-reflection | Partial | A regularized resolvent and manifold reflection exist; historical feature learning is absent. |
| 20. Sleep replay and reservoir sampling | Partial | A bounded checkpoint buffer exists; stratified Vitter replay and consolidation are absent. |
| 21. SoftVQ evolution and pruning | Partial | Nearest-centroid assignment and simple spawn/prune operations exist; full semantic evolution is absent. |
| 22. Thermodynamic Langevin noise | Partial | Formula-scaled noise can be enabled in the inference step; temperature estimation and annealing schedules are caller-controlled. |
| 23. Associative attention and crossbar currents | Partial | A threshold crossbar encoder and a simple phase-memory projection exist; associative retrieval is absent. |
| 24. Lyapunov-Hespanha stability | Partial | The non-negative Lyapunov candidate can be evaluated; no decay-rate proof or dwell-time enforcement is provided. |
| 25. SOC criticality regulator | Partial | Precision-weighted tail tracking adapts inference damping; it is not a validated avalanche/criticality detector. |
| 26. Harish-Chandra transforms | Partial | Radial phase features and capacity-bounded Plancherel-weighted associative retrieval exist; the full spherical transform is absent. |
| 27. Multi-agent theory of mind | Partial | A finite planner scores actions from explicit other-agent action beliefs and utilities; belief inference and policy learning are absent. |
| 28. Anti-livelock rollback and annealing | Partial | Checkpoint rollback primitives exist; the full watchdog, cooling schedule, and annealing are not integrated. |
| 29. Neuromorphic STE spike encoder | Partial | Threshold spikes and crossbar currents exist; no straight-through estimator. |
| 30. System-1 semantic relaxation | Partial | Inference-only repeated query relaxation exists; it does not include associative holographic retrieval. |
| 31. Conformal connection | Partial | Conformal drift is a standalone helper; Christoffel dynamics are not implemented. |
| 32. Non-separable conformal Hamiltonian | Partial | A constrained Hamiltonian and vector-field prototype are available; there is no integrated non-separable solver or energy-conservation guarantee. |
| 33. Generalized Cole-Hopf diffusion | Partial | A positivity-limited explicit graph diffusion step exists; the Riemannian Laplace-Beltrami operator is not implemented. |
| 34. Explicit AIXI polyhedral projection | Partial | Finite-model Q bounds, a separation check, and a pressure vector exist; these do not certify the universal AIXI optimum. |
| 35. Gödel JIT hot-swap | Partial | A versioned, thread-safe callable swap based on utility comparison exists; proof checking, JIT compilation, and zero-downtime guarantees are absent. |
| 36. Adaptive sectional curvature | Partial | Constant-curvature chart distances and a bounded scalar update exist; curvature learning is not integrated with free-energy gradients. |
| 37. RCU multi-timescale dispatcher | Partial | A copy-on-write engine map and versioned publication use a lock; this is not lock-free RCU and has no timing guarantee. |
| 38. Dynamic isometric dimension expansion | Partial | Zero-padding embeddings preserve tested Minkowski Gram matrices; automatic capacity-triggered expansion is absent. |

## Algorithm Specifications

| Algorithm | Status | Runtime scope |
|---|---|---|
| 1. Multimodal sensory grounding | Partial | Single-vector sensory projection only. |
| 2. System-1 fast inference | Partial | A simplified single-step inference update exists. |
| 3. Symplectic implicit midpoint | Partial | Projected kick-drift only; not implicit midpoint. |
| 4. Fast-sweeping Eikonal | Partial | Discrete graph action solver only. |
| 5. Cole-Hopf/AIXI motor synthesis | Partial | A discrete graph-diffusion helper, finite AIXI-like bounds, and bounded continuous-action synthesis exist; no continuous PDE or complete EFE gradient is integrated. |
| 6. Ego-reflection | Partial | Simplified regularized reflection. |
| 7. Riemannian K-FAC | Partial | K-FAC primitive exists, not the full training algorithm. |
| 8. Criticality and SoftVQ dynamics | Partial | Basic codebook spawn/prune exists; no SOC controller. |
| 9. Sleep replay and rollback | Partial | Checkpoint rollback only; no stratified replay. |
| 10. Semantic query and holographic retrieval | Partial | Inference-only query relaxation exists; no associative Harish-Chandra retrieval. |
| 11. Anytime Levin decider | Partial | Anytime bounds cover a finite caller-supplied prefix-code set; no Levin program scheduling, universal search, or certified action. |
| 12. Spherical-transform retrieval | Partial | Bounded radial-phase similarity retrieval uses a Plancherel weight; no angle-resolved spherical transform or full convolution. |
| 13. Gödel JIT hot-swap | Partial | Versioned callable publication and multi-timescale dispatch exist; no formal proof checker, JIT, lock-free RCU, or zero-stall guarantee. |

This inventory should be updated alongside runtime changes. Passing tests establish only the tested code invariants; they do not validate every equation or theorem in the specification.
