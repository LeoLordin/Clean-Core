"""Finite-graph approximation for Section 5 viscous eikonal wavefronts.

This is not a solver for the continuous PDE.
"""

import operator

import numpy as np

from clean_core.config import D_LORENTZ, FP64
from clean_core.geometry.eikonal import godunov_fast_sweeping_solver


def solve_section_05_wavefront(
    graph_adj,
    node_mus,
    omega,
    eps_visc=0.01,
    max_sweeps=100,
    tolerance=1e-6,
    source_index=0,
):
    """Compute graph wavefront action, interpreting ``omega`` as local Omega.

    The existing Godunov solver accepts the refractive index ``n``; this
    finite-graph approximation converts the Section 5 conformal factor using
    ``n = 1 / Omega``. It does not solve the continuous viscous PDE.
    """
    try:
        adjacency = np.asarray(graph_adj, dtype=FP64)
        coordinates = np.asarray(node_mus, dtype=FP64)
        omega_values = np.asarray(omega, dtype=FP64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("graph_adj, node_mus, and omega must be numeric") from error

    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("graph_adj must be a square matrix")
    if not np.all(np.isfinite(adjacency)) or np.any(adjacency < 0.0):
        raise ValueError("graph_adj must contain finite non-negative edge weights")
    if coordinates.ndim == 2 and adjacency.shape[0] != coordinates.shape[0]:
        raise ValueError("graph_adj must be a square matrix matching node_mus")
    if coordinates.shape != (adjacency.shape[0], D_LORENTZ):
        raise ValueError(f"node_mus must have shape ({adjacency.shape[0]}, {D_LORENTZ})")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("node_mus must contain only finite coordinates")
    if omega_values.shape != (adjacency.shape[0],):
        raise ValueError("omega must contain one value per graph node")
    if not np.all(np.isfinite(omega_values)) or np.any(omega_values <= 0.0):
        raise ValueError("omega must contain finite positive values")

    try:
        eps_visc = float(eps_visc)
        tolerance = float(tolerance)
        max_sweeps = operator.index(max_sweeps)
        source_index = operator.index(source_index)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("solver parameters must be finite scalars and integer indices") from error
    if not np.isfinite(eps_visc) or eps_visc < 0.0:
        raise ValueError("eps_visc must be finite and non-negative")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    if max_sweeps < 1:
        raise ValueError("max_sweeps must be a positive integer")
    if not 0 <= source_index < adjacency.shape[0]:
        raise ValueError("source_index is outside the graph")

    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        refractive_index = np.reciprocal(omega_values)
    if not np.all(np.isfinite(refractive_index)) or np.any(refractive_index <= 0.0):
        raise ValueError("omega must have finite positive reciprocals")

    return godunov_fast_sweeping_solver(
        adjacency,
        coordinates,
        refractive_index,
        eps_visc=eps_visc,
        max_sweeps=max_sweeps,
        tolerance=tolerance,
        source_index=source_index,
    )
