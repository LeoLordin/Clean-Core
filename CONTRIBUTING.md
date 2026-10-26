# Contributing

Clean Core is an experimental research prototype. Contributions should keep measured behavior, implementation status, and theoretical proposals clearly distinguished.

## Before opening a pull request

1. Open or link an issue for substantial changes so the intended behavior can be discussed.
2. Keep changes focused and update the relevant documentation when public behavior changes.
3. Add tests for numerical invariants, edge cases, and regression behavior.
4. Run `python -m pytest`, `python smoke_capabilities.py`, and `python gym_cartpole_challenge/test_cartpole_active_inference.py`.
5. Do not describe heuristic outputs as certified bounds or theoretical claims as implemented results.

## Mathematical changes

Include the assumptions, dimensions, and source of any changed equation. If a derivation has not been independently reviewed, label it as a proposal rather than a validated result.
