[Back to theory contents](../THEORY.md)

### Algorithm 1: Multimodal Dirichlet Sensory Grounding ($`l=0`$)
* **Inputs:** Multimodal observation tensors $`\mathbf{x}_{\mathrm{img}} \in \mathbb{R}^{C \times H \times W}`$, $`\mathbf{x}_{\mathrm{aud}} \in \mathbb{R}^{T_{\mathrm{aud}}}`$, $`\mathbf{x}_{\mathrm{text}} \in \mathbb{Z}^{T_{\mathrm{txt}}}`$.
* **Outputs:** Grounded Dirichlet manifold coordinate $`\boldsymbol{\mu}_0^{(0)}(t+\Delta t) \in \mathcal{H}^d`$, mutual information bound $`\mathcal{I}_{\mathrm{grounding}}`$.
* **Complexity:** $`O(d \cdot \sum \mathrm{dim}(\text{inputs}))`$.
* **Execution Steps:**
  1. Extract multimodal latent embeddings: $`\mathbf{z}_{\mathrm{vis}} = \mathbf{W}_{\mathrm{vis}} \mathrm{Enc}_{\mathrm{vis}}(\mathbf{x}_{\mathrm{img}})`$, $`\mathbf{z}_{\mathrm{aud}} = \mathbf{W}_{\mathrm{aud}} \mathrm{Enc}_{\mathrm{aud}}(\mathbf{x}_{\mathrm{aud}})`$, $`\mathbf{z}_{\mathrm{txt}} = \mathbf{W}_{\mathrm{txt}} \mathrm{Enc}_{\mathrm{txt}}(\mathbf{x}_{\mathrm{text}})`$.
  2. Compute bundle projection: $`\mathbf{a}_{\mathrm{bundle}} = \pi_{\mathrm{bundle}}(\mathbf{z}_{\mathrm{vis}} \parallel \mathbf{z}_{\mathrm{aud}} \parallel \mathbf{z}_{\mathrm{txt}}) \in \mathbb{R}^{d-1}`$.
  3. Lift to tangent pole space: $`\mathbf{v}_{\mathrm{bundle}} = E(\mathbf{o})\mathbf{a}_{\mathrm{bundle}} \in T_{\mathbf{o}}\mathcal{H}^d`$, where $`E(\mathbf{o}) = [\mathbf{0}_{1 \times (d-1)}; \mathbf{I}_{d-1}]`$.
  4. Evaluate pole exponential map: $`\boldsymbol{\mu}_0^{(0)}(t+\Delta t) = \mathrm{Exp}_{\mathbf{o}}^{\mathcal{H}^d}(\mathbf{v}_{\mathrm{bundle}})`$.
  5. Enforce boundary conditions: $`\mathbf{w}_0^{(0)} \leftarrow \mathbf{0}`$, $`\mathbf{p}_0^{(0)} \leftarrow \mathbf{0}`$.
  6. Compute contrastive InfoNCE lower bound $`\mathcal{I}_{\mathrm{grounding}}`$ across batch $`\mathcal{B}`$.

---
