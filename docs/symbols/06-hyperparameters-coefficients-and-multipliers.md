[Back to theory contents](../THEORY.md)

### Hyperparameters, Coefficients, and Multipliers
* $`t`$: Continuous physical system time (s).
* $`\Delta t = 0.01\text{ s}`$: Numerical integration time step.
* $`\epsilon_0, \epsilon_1, \epsilon_2`$: Physical timescales ($`\epsilon_0 \sim 10^{-3}\text{ s}`$, $`\epsilon_1 \sim 1\text{ s}`$, $`\epsilon_2 \sim 10^5\text{ s}`$).
* $`\lambda = 1.0`$: Thermodynamic control regularization parameter in the Cole-Hopf scheme.
* $`\sigma = 0.1`$: Wavefunction stochastic diffusion coefficient.
* $`\hbar_{\mathrm{cog}} = 1.0`$: Cognitive quantum of action in the holographic phase interference operator.
* $`\mathcal{F}_{\mathrm{threshold}} = 100.0`$: Energy threshold mapping free energy to the conformal factor.
* $`\lambda_{\mathrm{Eik}} = 0.5`$: Weight of the Eikonal PDE residual penalty in the PHR functional.
* $`\varepsilon_{\mathrm{visc}}`$: Laplace-Beltrami viscosity coefficient regularizing the Eikonal equation.
* $`\mathcal{L}_{\mathrm{Landauer}}`$: Landauer thermal dissipation penalty for pruned vocabulary tokens.
* $`k_B`$: Boltzmann constant.
* $`\varepsilon_{\mathrm{safe}}`$: Analytical safety radius of the polyhedral AIXI decision cell.
* $`\lambda_{\mathrm{mid}}, \mu_{\mathrm{mid}}`$: Scalar Lagrange multipliers for the symplectic implicit midpoint integrator.
* $`A_m, B_m, C_m`$: Local quadratic coefficients for the Godunov upwind scheme.
* $`R_{\mathrm{pred\_max}} = 3.0`$: Maximum radius of tangent Euclidean retraction.
* $`M_{\mathrm{step}} = 1.0`$: Frobenius norm clipping limit for synaptic natural gradient tensors.
* $`\gamma_{\mathrm{anneal}} \approx 0.95`$: Epoch-wise Langevin noise annealing factor.
* $`r_i^{(l)}(t)`$: Chordal distance from neuron to its layer barycenter.
* $`f_{\mathrm{drift\_safe}}(r)`$: Kinematic Laplace-Beltrami drift smoothing function.
* $`\pi_i^{(l)}(t)`$: Scalar variational prediction precision (FP32).
* $`\pi_{\min} = 10^{-4}, \pi_{\max} = 10^4`$: Clamping bounds for scalar variational precision.
* $`E_{ji}^{(l+1)}(t) \in [E_{\min}, 1.0]`$: Synaptogenic inter-layer connection weight ($`E_{\min} = 0.01`$).
* $`v_{\max} = 5.0`$: Maximum velocity threshold on the hyperboloid.
* $`\delta_{\mathrm{eps}} = 10^{-12}`$: Machine numerical zero-division guard.
* $`\delta_{\mathrm{reg}} = 10^{-6}`$: Tikhonov regularizer for the Gödelian resolvent operator.
* $`\delta_{\mathrm{proj}} = 10^{-5}`$: Regularization constant for counterfactual action projection.
* $`\delta_{\mathrm{soc}} = 10^{-6}`$: Numerical guard for the SOC criticality regulator.
* $`\delta_{\mathrm{scm}} = 10^{-3}`$: Resolvent stabilization constant for the NOTEARS model.
* $`\delta_{\mathrm{kfac}} = 0.2`$: Tikhonov damping factor for inverting Kronecker covariance factors.
* $`\delta_{\min} = 10^{-5}`$: Regularization factor for DLR covariance tracking.
* $`\epsilon_{\mathrm{gap}} = 0.05`$: Spectral stability gap for $`\mathbf{W}_{\mathrm{self}, v}`$ and $`\mathbf{A}_{\mathrm{SCM}}`$.
* $`\beta = 5.0`$: Steepness coefficient of the sigmoidal spike mask.
* $`\tau_i^{(l)}`$: Local energy threshold for neuronal spike emission.
* $`\hat{\alpha}_{\mathrm{measured}}(t)`$: Measured Pareto tail exponent of the SOC criticality regulator.
* $`\gamma_{\mathrm{soc}} = 0.05`$: Exponential moving average smoothing factor for SOC tracking.
* $`\alpha_{\mathrm{damp}}(t)`$: Dynamic viscous damping coefficient.
* $`\alpha_{\min} = 0.05`$: Lower bound on damping ensuring asymptotic Lyapunov dissipativity.
* $`\Delta \alpha_{\max} = 0.01`$: Maximum allowable slew rate per step for the damping coefficient.
* $`\mathbf{T}_i \in [0, T_{\max}]`$: Thermodynamic Langevin temperature ($`T_{\max} = 1.0`$, FP32).
* $`T_{\mathrm{vq}} = 0.1`$: Softmax temperature for SoftVQ attention.
* $`T_{\mathrm{attn}} = 0.5`$: Attention temperature for associative memory retrieval.
* $`R_{\mathcal{H}} = 1.0`$: Radius of curvature of the hyperbolic manifold.
* $`\eta_E = 0.01, \lambda_E = 0.001`$: Hebbian growth rate and decay coefficient for graph topology.
* $`\eta_m = 0.05, \eta_k = 0.01`$: Exponential smoothing rates for DLR and K-FAC factors.
* $`\eta_a = 0.05, \lambda_{\mathrm{policy}} = 0.1`$: Learning rate and consistency weight for motor actuation.
* $`\eta_\lambda = 0.01, \gamma_{\mathrm{prox}} = 0.1`$: Adaptation rate and proximal smoothing for PHR multipliers.
* $`\rho_k = 1.0`$: Augmented Lagrangian quadratic penalty weight.
* $`\gamma_{\mathrm{ema}} = 0.1`$: Geodesic momentum factor for introspection trajectory tracking.
* $`\gamma_{\mathrm{bary}} = 0.2`$: Fréchet barycenter streaming filter step-size.
* $`\tau_{\mathrm{dwell}} = 0.5\text{ s}`$: Minimum continuous dwell time in Hespanha's stability theorem.
* $`\mu_{\mathrm{jump}} = 0.1`$: Maximum allowable Lyapunov jump factor under discrete switching events.
* $`\sigma_{\max}(\mathbf{W})`$: Maximum singular value of weight tensors.
* $`\mu_{\mathrm{DAG}} = 1.0`$: Penalty multiplier for NOTEARS acyclicity violations.
* $`\theta_{\mathrm{impasse}} = 5.0`$: Free energy derivative threshold triggering VRAM state rollback.
* $`\tau_{\mathrm{cooldown}, 0} = 1.0\text{ s}`$: Base cooldown timer for Anti-Livelock resets.
* $`\tau_{\max} = 60.0\text{ s}`$: Maximum allowable cooldown under exponential backoff.
* $`N_{\mathrm{consecutive\_resets}}`$: Counter of consecutive emergency VRAM rollbacks (INT32).
* $`\theta_{\mathrm{spawn}} = 15.0`$: Prediction error threshold triggering concept centroid spawning.
* $`\theta_{\mathrm{prune}} = 0.05`$: Usage threshold triggering codebook token pruning.
* $`K_{\max} = 1024`$: Maximum allowable codebook size for SoftVQ.
* $`d_{\min} = 0.5`$: Minimum spatial deduplication radius for codebook tokens.
* $`K_{\mathrm{check}} = 5`$: Depth of the VRAM checkpoint ring buffer.
* $`f_{\max} = 1000.0\text{ Hz}`$: Maximum clock frequency of the neuromorphic spike interface.
* $`\kappa_v = 0.01`$: Kinetic friction parameter in the PHR functional.

---
