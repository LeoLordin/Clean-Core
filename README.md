# Clean Core 5.0 (`v0.1.0-alpha`)
> **A Lightweight Non-Euclidean Active Inference Kernel on the $\mathcal{H}^{15}$ Lorentz Hyperboloid**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

> ⚠️ **Project Status: Early-Stage Research Alpha (`v0.1.0-alpha`)**
> **Clean Core 5.0** is an open, exploratory research prototype. Errors, omissions, inconsistencies, and untested assumptions may exist in both the mathematical theory and the code. The theory is not a formal proof, and the runtime is not production software or a complete implementation of every theoretical section.
>
> **Help is welcome across the whole project:** mathematical derivations, assumptions and notation, geometry and numerical methods, algorithms, implementation, tests, benchmarks, documentation, and reproducibility. Please open an [Issue](../../issues) to discuss questions or report problems, or submit a Pull Request with focused changes and tests where practical.

---

## ⚡ Overview

**Clean Core 5.0** is an experimental Python research prototype exploring Lorentz-hyperboloid geometry and active-inference-inspired control. The code contains standalone components and demos; it does not yet integrate every module described in the theory documents into one validated cognitive architecture.

Instead of scaling parameter-heavy autoregressive models (Transformers) with high thermodynamic footprint, Clean Core operates as an active dynamical system moving along geodesics on a 16-dimensional Lorentz hyperboloid manifold $\mathcal{H}^{15}$.

```
               [ Sensory Packets (Dirichlet Layer) ]
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
       │       R^16 LORENTZ SPACE / H^15 SHEET        │
        │   Weierstrass Embedding: -x0^2 + ∑xi^2 = -1   │
        └──────────────────────┬───────────────────────┘
                               │
        ┌──────────────────────┼───────────────────────┐
        ▼                      ▼                       ▼
 [ Geodesic Core ]     [ Graph Action ]       [ NOTEARS Prototype ]
 Tested manifold      Godunov-style solver    Bounded adjacency update
 maps and transport   on hyperbolic edges     with log acyclicity
        │                      │                       │
        └──────────────────────┼───────────────────────┘
                               │
                               ▼
 [ Bounded score heuristic ] -> [ bounded motor vector ]
```

---

## Implemented Components

The current code provides experimental implementations of:

* Lorentz-hyperboloid exponential/logarithmic maps, orthonormal boost frames, and tested parallel transport.
* A graph-based Godunov-style action solver using hyperbolic edge distances.
* DLR covariance tracking with Oja-QR and Woodbury precision calculation.
* A bounded observation-score heuristic in the agent pipeline. It is **not** a Solomonoff/AIXI solver and does not provide a certified safety margin. A separate finite-model AIXI-like prototype exists, but it does not perform universal search.
* A small stateful agent and a separate hand-tuned CartPole-like control demo.
* Additional partial research prototypes for Cole-Hopf graph diffusion, VIB objectives, multi-agent action scoring, conformal Hamiltonian fields, curvature adaptation, isometric dimension expansion, radial-phase memory retrieval, motor synthesis, and versioned engine dispatch.

These additions implement limited, testable subsets only. They do not provide continuous manifold PDE solvers, certified AIXI decisions, full spherical transforms, formal proof checking, lock-free RCU, or complete integration of the theory. The [implementation status matrix](docs/IMPLEMENTATION_STATUS.md) records each section and algorithm's current scope; all are marked `Partial`.

---

## Quickstart and Tests

### Installation
The package requires NumPy and SciPy. Install the package and test dependencies with:

```bash
git clone https://github.com/LeoLordin/Clean-Core.git
cd Clean-Core
python -m pip install -e ".[test]"
python -m pytest
```

### Run Smoke Benchmark
Runs a short geometry check and agent loop. The printed timings are machine-dependent and are not performance guarantees:

```bash
python smoke_capabilities.py
```

---

## Standalone CartPole-Like Demo

This script contains its own small CartPole-like physics simulation and a hand-tuned linear state-feedback controller. It is not Gymnasium's `CartPole-v1` and does not validate the full agent or establish general active-inference performance:

```bash
python gym_cartpole_challenge/test_cartpole_active_inference.py
```

The initial state and controller gains are fixed in the script. Results are illustrative and should not be treated as a benchmark comparison.

---

## 📂 Repository Layout

```text
clean_core/
├── config.py              # Reference Profile (D=16, d=15, FP64/FP32 layout)
├── agent.py               # Experimental CleanCoreProductionAgent pipeline
├── geometry/              # Lorentz maps, projected steps, Hamiltonian, curvature, embeddings
├── estimation/            # Fisher-Rao DLR factorization, K-FAC Natural Gradients
├── core/                  # Python neuron states, SoftVQ prototype, fast inference
├── cognition/             # Inference, finite-model bounds, ToM, engine dispatch
├── memory/                # Bounded radial-phase associative retrieval and replay
└── interfaces/            # Sensory grounding, motor EFE and synthesis prototypes
```

---

## Roadmap and Contributions

Clean Core is an open research prototype. Contributions should distinguish measured behavior from theoretical proposals.

We actively welcome Pull Requests and contributions:
* [ ] **C++20 / CUDA Acceleration:** Porting `lorentz.py` to custom CUDA kernels via `pybind11`.
* [ ] **High-Order Symplectic Solvers:** Implementing full implicit fixed-point iterations on non-compact manifolds.
* [ ] **Gymnasium Multi-Task Suite:** Benchmarking non-Euclidean Active Inference across `Acrobot`, `MountainCarContinuous`, and custom non-Euclidean mazes.
* [ ] **Formal Verification:** Integrating automated theorem-proving hooks for the Gödel reflection module.

If you are a mathematician, systems programmer, or AI safety researcher — feel free to fork, open an issue, and submit a PR!

See the [contribution guide](CONTRIBUTING.md) and [runtime implementation status](docs/IMPLEMENTATION_STATUS.md) before proposing changes to theoretical modules.

---

## 📚 Mathematical Specification
For the research specification and notation (Sections 1–38), see the [theory index](docs/THEORY.md). Check the [implementation status matrix](docs/IMPLEMENTATION_STATUS.md) before assuming a section is implemented or independently validated.

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.

**Author:** [LeoLordin](https://github.com/LeoLordin)  
*Architectural Proposal & Open-Source Research Prototype.*
