import numpy as np
import pytest

from clean_core.config import D_LORENTZ, POLE
from clean_core.geometry.lorentz import exp_map
from clean_core.prototypes.algorithms.algorithm_04 import solve_algorithm_04


def _line_graph_nodes(distances):
    tangent = np.zeros(D_LORENTZ)
    tangent[1] = 1.0
    return np.stack([exp_map(POLE.copy(), distance * tangent) for distance in distances])


def test_algorithm_04_matches_known_homogeneous_shortest_paths():
    adjacency = np.array(
        [[0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 0]],
        dtype=float,
    )
    # The branch has the known path costs 0 -> 1 -> 2 and 0 -> 1 -> 3.
    nodes = _line_graph_nodes([0.0, 1.0, 2.0, 3.0])

    action = solve_algorithm_04(
        adjacency, nodes, np.ones(4), eps_visc=0.0, source_index=0
    )

    np.testing.assert_allclose(action, [0.0, 1.0, 2.0, 3.0], atol=1e-10)


def test_algorithm_04_uses_refractive_index_and_nonzero_source():
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    action = solve_algorithm_04(
        adjacency,
        _line_graph_nodes([0.0, 1.0, 2.0]),
        np.full(3, 2.0),
        eps_visc=0.0,
        source_index=1,
    )

    np.testing.assert_allclose(action, [2.0, 0.0, 2.0], atol=1e-10)


def test_algorithm_04_viscosity_regularizes_finite_graph_action():
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    nodes = _line_graph_nodes([0.0, 1.0, 2.0])
    indices = np.ones(3)
    inviscid = solve_algorithm_04(adjacency, nodes, indices, eps_visc=0.0)

    viscous = solve_algorithm_04(adjacency, nodes, indices, eps_visc=0.01)

    assert np.all(np.isfinite(viscous))
    assert viscous[0] == 0.0
    assert np.all(viscous >= 0.0)
    assert np.all(viscous[1:] < inviscid[1:])


def test_algorithm_04_leaves_disconnected_nodes_unreachable():
    adjacency = np.zeros((3, 3))
    adjacency[0, 1] = adjacency[1, 0] = 1.0

    action = solve_algorithm_04(
        adjacency, _line_graph_nodes([0.0, 1.0, 2.0]), np.ones(3), eps_visc=0.0
    )

    np.testing.assert_allclose(action[:2], [0.0, 1.0], atol=1e-10)
    assert np.isinf(action[2])


@pytest.mark.parametrize(
    ("graph_adj", "node_mus", "refractive_index", "message"),
    [
        (np.zeros((2, 3)), _line_graph_nodes([0.0, 1.0]), np.ones(2), "square matrix"),
        (np.array([[0, 1], [0, 0]]), _line_graph_nodes([0.0, 1.0]), np.ones(2), "undirected"),
        (np.array([[0, -1], [-1, 0]]), _line_graph_nodes([0.0, 1.0]), np.ones(2), "non-negative"),
        (np.zeros((2, 2)), _line_graph_nodes([0.0]), np.ones(2), "node_mus"),
        (np.zeros((2, 2)), _line_graph_nodes([0.0, 1.0]), [1.0], "refractive_index"),
    ],
)
def test_algorithm_04_rejects_invalid_graph_inputs(
    graph_adj, node_mus, refractive_index, message
):
    with pytest.raises(ValueError, match=message):
        solve_algorithm_04(graph_adj, node_mus, refractive_index)


@pytest.mark.parametrize(
    ("index_values", "message"),
    [
        ([1.0, 0.0], "refractive_index"),
        ([1.0, np.nan], "refractive_index"),
        ([1.0, np.inf], "refractive_index"),
        ([1.0, 1e200], "finite squares"),
    ],
)
def test_algorithm_04_rejects_invalid_refractive_indices(index_values, message):
    with pytest.raises(ValueError, match=message):
        solve_algorithm_04(
            np.zeros((2, 2)), _line_graph_nodes([0.0, 1.0]), index_values
        )


def test_algorithm_04_rejects_non_hyperboloid_coordinates():
    nodes = _line_graph_nodes([0.0, 1.0])
    nodes[1, 0] += 1.0

    with pytest.raises(ValueError, match="Lorentz hyperboloid"):
        solve_algorithm_04(np.zeros((2, 2)), nodes, np.ones(2))


@pytest.mark.parametrize(
    "options",
    [
        {"eps_visc": -0.1},
        {"eps_visc": np.inf},
        {"max_sweeps": 0},
        {"max_sweeps": 1.5},
        {"tolerance": np.nan},
        {"tolerance": -1.0},
        {"source_index": 2},
    ],
)
def test_algorithm_04_rejects_invalid_solver_options(options):
    with pytest.raises(ValueError):
        solve_algorithm_04(
            np.zeros((2, 2)), _line_graph_nodes([0.0, 1.0]), np.ones(2), **options
        )


def test_algorithm_04_rejects_zero_length_edges():
    nodes = _line_graph_nodes([0.0, 0.0])
    adjacency = np.array([[0, 1], [1, 0]])

    with pytest.raises(ValueError, match="distinct coordinates"):
        solve_algorithm_04(adjacency, nodes, np.ones(2))
