# Runtime Implementation Status

This inventory describes the current Python runtime, not the validity of the mathematical proposals. Every numbered section and algorithm currently has a bounded prototype and is marked `Partial`; the status does not assert that the theory is complete, mathematically valid, or fully integrated. Passing tests establish only the tested software behavior.

## Mathematical Sections

| Section | Status | Runtime scope |
|---|---|---|
| 1. Lorentz-bundle stochastic dynamics | Partial | Finite stochastic Lie-Euler-Maruyama step lifts noise and drift through Lorentz tangent frames; full layer precision/force integration is absent. |
| 2. Maupertuis-Jacobi conformal geometry | Partial | Conformal factor, metric, connection correction, and drift helpers are available; no end-to-end learned conformal dynamics. |
| 3. Symplectic implicit midpoint | Partial | Bounded constrained nonlinear midpoint solve is tested on cotangent constraints; exact symplecticity is not established. |
| 4. Cole-Hopf active inference | Partial | Log/value transforms, positive goal superposition, optimal velocity, and bounded graph HJB step; no continuous Schrödinger-Beltrami solver. |
| 5. Viscous graph Eikonal | Partial | Validated finite-graph Godunov/fast-sweeping approximation with hyperbolic edges and graph viscosity; not a general PDE solver. |
| 6. Solomonoff-AIXI decider | Partial | Finite caller-supplied prefix-code Q tables give anytime lower/upper bounds and decision regions; no universal search or Levin scheduling. |
| 7. Microstate and EFE motor loop | Partial | Shape-checked finite microstate and bounded motor target/update proxy; not a complete EFE objective or learned controller. |
| 8. Master augmented Lagrangian | Partial | Projected PHR inequality multiplier, shifted term, gradient, and bounded state update; one constraint at a time, not the full master objective. |
| 9. Sylvester pragmatic free energy | Partial | Finite local objective evaluates DLR/Woodbury precision, log-determinant, and probe-based trace term; not the complete free-energy theory. |
| 10. Retraction perceptron and SoftVQ | Partial | Active-mask tangent SoftVQ, context aggregation, introspection projection, and radius-bounded Lorentz prediction; no perceptron training. |
| 11. Sigmoidal spike masking | Partial | Numerically stable forward sigmoid mask and limiting behavior; backward STE is not specified/implemented in this section. |
| 12. DLR/Oja-QR and K-FAC | Partial | Composition wraps tested DLR/Oja-QR and K-FAC statistics; not a complete Riemannian training loop. |
| 13. Spectral Lipschitz regularization | Partial | Spectral norm estimate and squared excess penalty; no weight-update equation or training-wide Lipschitz guarantee. |
| 14. Minkowski metric and geodesic operators | Partial | Tested Lorentz maps, tangent projection, short-geodesic transport, and small-vector logarithm expansion; not every proposed operator. |
| 15. Variational information bottleneck | Partial | Diagonal-Gaussian KL and empirical contrastive objective; no learned encoder or validated mutual-information estimator. |
| 16. Causal NOTEARS | Partial | Log-acyclicity objective/gradient, bounded adjacency update, and real-Schur spectral cap; no causal identification guarantee. |
| 17. Active-inference target divergence | Partial | Normalized categorical KL with support-aware infinite divergence; not a complete predicted-state/action inference agent. |
| 18. Multimodal sensory grounding | Partial | Numeric modality evidence combines into Dirichlet posterior and Lorentz pole coordinate; no image/audio/text encoders or semantic alignment. |
| 19. Gödelian ego-reflection | Partial | Finite regularized adjoint resolvent, historical feature update, and Lorentz reflection; no formal Gödel self-proof. |
| 20. Sleep replay and reservoir sampling | Partial | Bounded per-stratum Algorithm R reservoir and balanced replay; not Vitter skip sampling or full consolidation. |
| 21. SoftVQ evolution and pruning | Partial | Bounded usage-driven codebook update, spawn gates, and pruning; no validated semantic evolution. |
| 22. Thermodynamic Langevin noise | Partial | Reproducible formula-scaled Gaussian noise lifted to a tangent frame; no autonomous temperature estimator. |
| 23. Associative attention and crossbar currents | Partial | Finite manifold-aware attention and transported crossbar-current calculation; no hardware emulation. |
| 24. Lyapunov-Hespanha stability | Partial | Candidate and sample-based finite-step decrease diagnostic; no stability theorem or dwell-time guarantee. |
| 25. SOC criticality regulator | Partial | Bounded precision-weighted tail statistic/regulator; not a validated avalanche/SOC detector. |
| 26. Harish-Chandra transforms | Partial | Bounded finite-grid Poincare waves, Plancherel-weighted angular/spectral quadrature, and radial phase; not continuous spherical inversion. |
| 27. Multi-agent theory of mind | Partial | Explicit normalized/shrunk action beliefs and finite expected-utility planner; no learned belief model or coverage guarantee. |
| 28. Anti-livelock rollback and annealing | Partial | Bounded checkpoint/watchdog rollback, sensory re-grounding, cooldown, and temperature policy; no hard livelock/VRAM guarantee. |
| 29. Neuromorphic STE spike encoder | Partial | Lorentz feature rate and Poisson probability, threshold sampling, and stated surrogate derivative; no hardware training system. |
| 30. System-1 semantic relaxation | Partial | Iterative inference-only Lorentz relaxation with speed cap; does not train weights or ensure semantic correctness. |
| 31. Conformal connection | Partial | Conformal Christoffel correction and base-connection composition; no general symbolic connection engine. |
| 32. Non-separable conformal Hamiltonian | Partial | Constrained vector field and projected explicit-midpoint step; no exact symplecticity or energy conservation guarantee. |
| 33. Generalized Cole-Hopf diffusion | Partial | Positivity-limited weighted graph approximation; not the continuous Laplace-Beltrami diffusion. |
| 34. Explicit AIXI polyhedral projection | Partial | Finite Kraft-weighted Q bounds, strict separation, and pressure; not a universal AIXI certificate. |
| 35. Gödel JIT hot-swap | Partial | Versioned utility-gated callable slot and snapshot dispatch; no proof checker, JIT, or zero-downtime guarantee. |
| 36. Adaptive sectional curvature | Partial | Flat/positive/negative chart distances and bounded scalar curvature update; no coupled curvature/free-energy training. |
| 37. RCU multi-timescale dispatcher | Partial | Thread-safe copy-on-write dispatcher with versioned publication; uses locks and is not lock-free RCU. |
| 38. Dynamic isometric dimension expansion | Partial | Zero-padded upper-hyperboloid embedding with Minkowski Gram preservation; no automatic capacity trigger. |

## Algorithm Specifications

| Algorithm | Status | Runtime scope |
|---|---|---|
| 1. Multimodal sensory grounding | Partial | Fixed numeric pooling/token bucketing, Section 18 Dirichlet evidence, and Lorentz grounding; no learned modality encoders or MI estimate. |
| 2. System-1 fast inference | Partial | One deterministic bounded manifold inference step with DLR precision update; no weight training. |
| 3. Symplectic implicit midpoint | Partial | Bounded nonlinear midpoint adapter enforces cotangent endpoint constraints; no exact symplecticity proof. |
| 4. Fast-sweeping Eikonal | Partial | Validated finite-graph Godunov sweeps with hyperbolic edge distances; not a continuous PDE solver. |
| 5. Cole-Hopf/AIXI motor synthesis | Partial | Composes graph wavefunction step, finite Q bounds, explicit social beliefs, and bounded motor target/update; not full PDE/AIXI/EFE. |
| 6. Ego-reflection | Partial | Composes Section 19 regularized resolvent with Lorentz reflection/introspection updates; no formal self-proof. |
| 7. Riemannian K-FAC | Partial | Scheduled finite K-FAC natural-gradient update with norm cap; not a convergence guarantee. |
| 8. Criticality and SoftVQ dynamics | Partial | Composes bounded SOC-like damping with usage decay, pruning, and gated spawn; not a criticality proof. |
| 9. Sleep replay and rollback | Partial | Composes stratified replay, bounded checkpoints, watchdog rollback, sensory re-grounding, and cooldown. |
| 10. Semantic query and holographic retrieval | Partial | Four-step weight-preserving relaxation plus bounded associative recall; default target is a feature proxy, not semantic decoding. |
| 11. Anytime Levin decider | Partial | Budgeted evaluation of supplied prefix-code models with finite Q-bound separation; no universal Levin search or certified action. |
| 12. Spherical-transform retrieval | Partial | Section 26 finite-grid angular/spectral quadrature adapter; not full continuous spherical transform. |
| 13. Gödel JIT hot-swap | Partial | Versioned utility-gated callable publication and timescale dispatch; no proof checking, JIT, lock-free RCU, or zero-stall guarantee. |

## Discoverability and Tests

Each formula section has a focused module at `clean_core/prototypes/sections/section_NN.py` and a test at `tests/prototype_sections/test_section_NN.py`. Algorithm prototypes and tests use the corresponding `algorithms/algorithm_NN.py` and `tests/prototype_algorithms/test_algorithm_NN.py` paths. `clean_core.prototypes.catalog` imports these by number. `clean_core.prototypes.runtime.sensory_to_inference_step` composes the initial numeric sensory-grounding and manifold-inference stages.

Run all tests with `python scripts/run_prototype_tests.py --all`, or select an individual block with `--section N` / `--algorithm N`. The catalog and integrated runtime have additional tests under `tests/prototype_integration/`. These checks cover finite test cases and invariants, not mathematical proof or completeness.
