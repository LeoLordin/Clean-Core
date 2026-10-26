"""Finite-graph Godunov fast-sweeping approximation for Algorithm 4.

This computes a discrete action field on a finite graph; it is not a general
solver for the continuous Eikonal PDE.
"""

import operator

import numpy as np

from clean_core.config import D_LORENTZ, ETA_L, FP64
from clean_core.geometry.lorentz import chordal_distance


def solve_algorithm_04(
    graph_adj,
    node_mus,
    refractive_index,
    eps_visc=0.01,
    max_sweeps=100,
    tolerance=1e-6,
    source_index=0,
):
    """Return graph action values from one source using chordal edge lengths.

    ``graph_adj`` is a symmetric, non-negative connectivity matrix: every
    positive off-diagonal entry denotes an edge, and its magnitude is ignored.
    ``node_mus`` contains one future-unit Lorentz-hyperboloid coordinate per
    node. ``refractive_index`` contains one finite positive index per node.
    Nodes unreachable from the source have action ``inf``.

    The viscosity term is an explicit weighted graph-Laplacian correction in
    the local Godunov update. The result is a finite-graph approximation, not
    a general continuous-PDE solution.
    """
    try:
        adjacency = np.asarray(graph_adj, dtype=FP64)
        coordinates = np.asarray(node_mus, dtype=FP64)
        indices = np.asarray(refractive_index, dtype=FP64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("graph_adj, node_mus, and refractive_index must be numeric") from error

    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("graph_adj must be a square matrix")
    node_count = adjacency.shape[0]
    if node_count == 0:
        raise ValueError("graph_adj must contain at least one node")
    if not np.all(np.isfinite(adjacency)) or np.any(adjacency < 0.0):
        raise ValueError("graph_adj must contain finite non-negative entries")
    if not np.array_equal(adjacency, adjacency.T):
        raise ValueError("graph_adj must describe an undirected graph")
    if np.any(np.diag(adjacency) != 0.0):
        raise ValueError("graph_adj must not contain self-edges")

    if coordinates.shape != (node_count, D_LORENTZ):
        raise ValueError(f"node_mus must have shape ({node_count}, {D_LORENTZ})")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("node_mus must contain only finite coordinates")
    minkowski_norms = np.sum(coordinates * ETA_L * coordinates, axis=1)
    if (
        np.any(coordinates[:, 0] <= 0.0)
        or not np.all(np.isfinite(minkowski_norms))
        or not np.allclose(minkowski_norms, -1.0, rtol=1e-7, atol=1e-7)
    ):
        raise ValueError("node_mus must lie on the future unit Lorentz hyperboloid")

    if indices.shape != (node_count,):
        raise ValueError("refractive_index must contain one value per graph node")
    if not np.all(np.isfinite(indices)) or np.any(indices <= 0.0):
        raise ValueError("refractive_index must contain finite positive values")
    with np.errstate(over="ignore"):
        index_squares = np.square(indices)
    if not np.all(np.isfinite(index_squares)):
        raise ValueError("refractive_index values must have finite squares")

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
    if not 0 <= source_index < node_count:
        raise ValueError("source_index is outside the graph")

    neighbors_by_node = []
    distances_by_node = []
    for node in range(node_count):
        neighbors = np.flatnonzero(adjacency[node] > 0.0)
        distances = np.asarray(
            [chordal_distance(coordinates[node], coordinates[neighbor]) for neighbor in neighbors],
            dtype=FP64,
        )
        if not np.all(np.isfinite(distances)) or np.any(distances <= 0.0):
            raise ValueError("each graph edge must join distinct coordinates at finite distance")
        neighbors_by_node.append(neighbors)
        distances_by_node.append(distances)

    action = np.full(node_count, np.inf, dtype=FP64)
    action[source_index] = 0.0

    for _ in range(max_sweeps):
        changed = False
        largest_change = 0.0
        for order in (range(node_count), range(node_count - 1, -1, -1)):
            for node in order:
                if node == source_index:
                    continue
                neighbors = neighbors_by_node[node]
                if neighbors.size == 0:
                    continue
                distances = distances_by_node[node]
                values = action[neighbors]
                finite = np.isfinite(values)
                if not np.any(finite):
                    continue
                values = values[finite]
                weights = 1.0 / np.square(distances[finite])
                sort_order = np.argsort(values)
                values = values[sort_order]
                weights = weights[sort_order]

                rhs = index_squares[node]
                old_value = action[node]
                if eps_visc and np.isfinite(old_value):
                    laplacian = np.sum(weights * (values - old_value)) / np.sum(weights)
                    rhs = max(np.finfo(FP64).eps, rhs + eps_visc * laplacian)

                candidate = None
                for active_count in range(1, len(values) + 1):
                    active_weights = weights[:active_count]
                    active_values = values[:active_count]
                    coefficient_a = np.sum(active_weights)
                    coefficient_b = np.sum(active_weights * active_values)
                    coefficient_c = np.sum(active_weights * np.square(active_values))
                    discriminant = coefficient_b**2 - coefficient_a * (coefficient_c - rhs)
                    if discriminant < 0.0:
                        continue
                    root = (coefficient_b + np.sqrt(discriminant)) / coefficient_a
                    lower = active_values[-1]
                    upper = values[active_count] if active_count < len(values) else np.inf
                    if root + tolerance >= lower and root <= upper + tolerance:
                        candidate = root
                        break

                if candidate is None:
                    continue
                new_value = min(old_value, candidate)
                if new_value < old_value:
                    action[node] = new_value
                    changed = True
                    if np.isfinite(old_value):
                        largest_change = max(largest_change, old_value - new_value)
                    else:
                        largest_change = np.inf

        if not changed or largest_change <= tolerance:
            break

    return action
