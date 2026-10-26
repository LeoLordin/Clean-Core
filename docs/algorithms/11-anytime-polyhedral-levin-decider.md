[Back to theory contents](../THEORY.md)

### Algorithm 11: Anytime Polyhedral Levin Decider
* **Inputs:** Current bounds $`Q_{\min}(a)`$, $`Q_{\max}(a)`$, Kraft unresolved tail $`\Omega_{\mathrm{unresolved}}(k)`$.
* **Outputs:** Optimal action $`a^{*}`$ and safety radius $`\varepsilon_{\mathrm{safe}}`$.
* **Complexity:** $`O(|\mathcal{A}|)`$ per iteration with anytime interruptibility.
* **Execution Steps:**
  1. Update lower and upper bounds $`Q_{\min}(a)`$, $`Q_{\max}(a)`$ as candidate programs halt in Levin's interleaved search.
  2. Evaluate decision fixation boundary: $`Q_{\min}(a^{*}) > \max_{a \neq a^{*}} Q_{\max}(a)`$.
  3. If condition holds: fixate $`a^{*}`$ and compute polyhedral safety margin $`\varepsilon_{\mathrm{safe}}`$.
  4. Otherwise: allocate additional time slices to active programs in the search queue.
  5. Return $`a^{*}`$ and $`\varepsilon_{\mathrm{safe}}`$ to the motor synthesis controller.

---
