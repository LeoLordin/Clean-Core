import numpy as np
from clean_core.config import D_SPATIAL, FP64

def godunov_fast_sweeping_solver(graph_adj, node_mus, refractive_index, eps_visc=0.01):
    N = len(node_mus)
    S_grid = np.full(N, 1e6, dtype=FP64)
    S_grid[0] = 0.0
    for sweep in range(4):
        for i in range(N):
            neighbors = np.where(graph_adj[i] > 0)[0]
            if len(neighbors) == 0:
                continue
            s_vals = S_grid[neighbors]
            sorted_idx = np.argsort(s_vals)
            s_sorted = s_vals[sorted_idx]
            A_m = len(s_sorted)
            B_m = np.sum(s_sorted)
            C_m = np.sum(s_sorted**2)
            n_i_sq = refractive_index[i]**2
            disc = B_m**2 - A_m * (C_m - n_i_sq)
            if disc >= 0:
                S_candidate = (B_m + np.sqrt(disc)) / max(1e-4, A_m)
                S_grid[i] = min(S_grid[i], S_candidate)
    return S_grid
