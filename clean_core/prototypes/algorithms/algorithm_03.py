"""Bounded constrained midpoint implementation for Algorithm 3.

The solver uses a bounded nonlinear root solve for the midpoint equations and
both endpoint cotangent constraints. It is a numerical implicit solve, not a
proof that the canonical symplectic two-form is preserved.
"""

import operator

from clean_core.prototypes.sections.section_03 import (
    implicit_midpoint_step as _constrained_midpoint_step,
)

_MAX_FUNCTION_EVALUATIONS = 2000


def solve_algorithm_03(
    mu,
    p,
    grad_F,
    dt=0.01,
    *,
    tolerance=1e-10,
    max_iterations=200,
):
    """Advance one phase point with a bounded constrained midpoint solve.

    ``grad_F`` is evaluated at the implicit coordinate midpoint. The
    underlying nonlinear solver enforces the future unit-hyperboloid and
    cotangent endpoint constraints and raises ``RuntimeError`` if it fails to
    converge within its evaluation budget. The result is not a claim of exact
    symplecticity.

    ``max_iterations`` bounds the nonlinear solver's function evaluations; it
    must be an integer from 1 through 2000.
    """
    if isinstance(max_iterations, bool):
        raise ValueError("max_iterations must be a positive integer no greater than 2000")
    try:
        evaluation_budget = operator.index(max_iterations)
    except TypeError as error:
        raise ValueError(
            "max_iterations must be a positive integer no greater than 2000"
        ) from error
    if not 1 <= evaluation_budget <= _MAX_FUNCTION_EVALUATIONS:
        raise ValueError("max_iterations must be a positive integer no greater than 2000")

    return _constrained_midpoint_step(
        mu,
        p,
        grad_F,
        dt=dt,
        tolerance=tolerance,
        max_iterations=evaluation_budget,
    )


__all__ = ["solve_algorithm_03"]
