import numpy as np

from clean_core.config import FP64
from clean_core.geometry.lorentz import chordal_distance


def godunov_fast_sweeping_solver(
    graph_adj,
    node_mus,
    refractive_index,
    eps_visc=0.01,
    max_sweeps=100,
    tolerance=1e-6,
    source_index=0,
):
    coordinates = np.asarray(node_mus, dtype=FP64)
    adjacency = np.asarray(graph_adj)
    refractive_index = np.asarray(refractive_index, dtype=FP64)
    node_count = len(coordinates)

    if adjacency.shape != (node_count, node_count):
        raise ValueError("graph_adj must be a square matrix matching node_mus")
    if refractive_index.shape != (node_count,):
        raise ValueError("refractive_index must contain one value per node")
    if not 0 <= source_index < node_count:
        raise ValueError("source_index is outside the graph")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("node_mus must contain only finite coordinates")
    if not np.all(np.isfinite(refractive_index)) or np.any(refractive_index <= 0.0):
        raise ValueError("refractive_index must contain finite positive values")
    if eps_visc < 0.0 or max_sweeps < 1 or tolerance < 0.0:
        raise ValueError("eps_visc, max_sweeps, and tolerance must be non-negative")

    neighbors_by_node = []
    distances_by_node = []
    for node in range(node_count):
        neighbors = np.flatnonzero(adjacency[node] > 0)
        distances = np.array(
            [chordal_distance(coordinates[node], coordinates[other]) for other in neighbors],
            dtype=FP64,
        )
        valid = distances > np.finfo(FP64).eps
        neighbors_by_node.append(neighbors[valid])
        distances_by_node.append(distances[valid])

    action = np.full(node_count, np.inf, dtype=FP64)
    action[source_index] = 0.0
    fixed_sources = {source_index}

    for _ in range(max_sweeps):
        largest_change = 0.0
        changed = False
        for order in (range(node_count), range(node_count - 1, -1, -1)):
            for node in order:
                if node in fixed_sources:
                    continue
                neighbors = neighbors_by_node[node]
                distances = distances_by_node[node]
                if neighbors.size == 0:
                    continue

                values = action[neighbors]
                finite = np.isfinite(values)
                if not np.any(finite):
                    continue
                values = values[finite]
                edge_lengths = distances[finite]
                sorted_order = np.argsort(values)
                values = values[sorted_order]
                edge_lengths = edge_lengths[sorted_order]
                weights = 1.0 / np.square(edge_lengths)

                rhs = refractive_index[node] ** 2
                if eps_visc and np.isfinite(action[node]):
                    laplacian = np.sum(weights * (values - action[node])) / np.sum(weights)
                    rhs = max(np.finfo(FP64).eps, rhs + eps_visc * laplacian)

                candidate = None
                for active_count in range(1, len(values) + 1):
                    active_weights = weights[:active_count]
                    active_values = values[:active_count]
                    A_m = np.sum(active_weights)
                    B_m = np.sum(active_weights * active_values)
                    C_m = np.sum(active_weights * np.square(active_values))
                    discriminant = B_m**2 - A_m * (C_m - rhs)
                    if discriminant < 0.0:
                        continue

                    root = (B_m + np.sqrt(discriminant)) / A_m
                    lower = active_values[-1]
                    upper = values[active_count] if active_count < len(values) else np.inf
                    if root + tolerance >= lower and root <= upper + tolerance:
                        candidate = root
                        break

                if candidate is None:
                    continue
                old_value = action[node]
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