"""Finite-grid radial/spectral implementation of Algorithm 12.

The memory reuses Section 26's bounded point-mass episode store, sampled
Poincare waves, geodesic propagation phase, and Plancherel-weighted quadrature.
It is a finite-grid prototype, not a full angle-resolved spherical transform.
"""

from clean_core.prototypes.sections.section_26 import Section26HolographicMemory


class Algorithm12HolographicMemory(Section26HolographicMemory):
    """Bounded radial/spectral holographic memory for Algorithm 12.

    ``store`` accepts hyperboloid episode coordinates and vector amplitudes;
    ``recall`` evaluates their finite spectral/angular quadrature at a query.
    The sampled boundary grid is only a numerical approximation and does not
    implement the full angle-resolved spherical transform.
    """


def solve_algorithm_12(memory, query_mu, hbar_cog=1.0):
    """Retrieve an associative context vector from an Algorithm 12 memory."""
    if not isinstance(memory, Section26HolographicMemory):
        raise ValueError("memory must be a Section26HolographicMemory")
    return memory.recall(query_mu, hbar_cog=hbar_cog)


__all__ = ["Algorithm12HolographicMemory", "solve_algorithm_12"]
