import numpy as np
import pytest

from clean_core.config import POLE
from clean_core.geometry.lorentz import exp_map
from clean_core.prototypes.sections.section_05 import solve_section_05_wavefront


def _line_graph_nodes():
    tangent = np.zeros_like(POLE)
    tangent[1] = 1.0
    return np.stack([exp_map(POLE.copy(), distance * tangent) for distance in (0, 1, 2)])


def test_section_05_wavefront_scales_path_action_by_inverse_omega():
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    action = solve_section_05_wavefront(
        adjacency, _line_graph_nodes(), np.full(3, 0.5), eps_visc=0.0
    )

    np.testing.assert_allclose(action, [0.0, 2.0, 4.0], atol=1e-10)
    assert np.all(action >= 0.0)
    assert action[0] == 0.0


def test_section_05_viscosity_preserves_finite_nonnegative_wavefront():
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    nodes = _line_graph_nodes()
    omega = np.ones(3)
    inviscid = solve_section_05_wavefront(adjacency, nodes, omega, eps_visc=0.0)

    viscous = solve_section_05_wavefront(adjacency, nodes, omega, eps_visc=0.01)

    assert np.all(np.isfinite(viscous))
    assert np.all(viscous >= 0.0)
    assert viscous[0] == 0.0
    assert np.all(np.diff(viscous) >= 0.0)
    assert np.any(viscous[1:] < inviscid[1:])


def test_section_05_wavefront_supports_nonzero_source_and_disconnected_nodes():
    nodes = _line_graph_nodes()
    nodes = np.concatenate([nodes, nodes[:1]], axis=0)
    adjacency = np.zeros((4, 4), dtype=int)
    adjacency[:3, :3] = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]

    action = solve_section_05_wavefront(
        adjacency, nodes, np.ones(4), eps_visc=0.0, source_index=1
    )

    np.testing.assert_allclose(action[:3], [1.0, 0.0, 1.0], atol=1e-10)
    assert np.isinf(action[3])


@pytest.mark.parametrize(
    "omega",
    [
        [1.0, 0.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, np.nan, 1.0],
        [1.0, np.inf, 1.0],
        [[1.0, 1.0, 1.0]],
        [1e-320, 1.0, 1.0],
    ],
)
def test_section_05_wavefront_rejects_invalid_omega(omega):
    adjacency = np.zeros((3, 3), dtype=int)

    with pytest.raises(ValueError, match="omega"):
        solve_section_05_wavefront(adjacency, _line_graph_nodes(), omega)


def test_section_05_wavefront_delegates_graph_shape_validation():
    with pytest.raises(ValueError, match="square matrix"):
        solve_section_05_wavefront(
            np.zeros((2, 2)), _line_graph_nodes(), np.ones(3), eps_visc=0.0
        )


@pytest.mark.parametrize(
    "options",
    [
        {"eps_visc": np.nan},
        {"eps_visc": np.inf},
        {"eps_visc": -0.1},
        {"max_sweeps": 0},
        {"max_sweeps": 1.5},
        {"tolerance": np.inf},
        {"tolerance": -1.0},
        {"source_index": 3},
    ],
)
def test_section_05_wavefront_rejects_invalid_solver_parameters(options):
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    with pytest.raises(ValueError):
        solve_section_05_wavefront(
            adjacency, _line_graph_nodes(), np.ones(3), **options
        )


def test_section_05_wavefront_rejects_invalid_graph_and_coordinate_values():
    adjacency = np.array([[0.0, np.nan, 0.0], [np.nan, 0.0, 0.0], [0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="graph_adj"):
        solve_section_05_wavefront(adjacency, _line_graph_nodes(), np.ones(3))

    coordinates = _line_graph_nodes()
    coordinates[1, 0] = np.inf
    with pytest.raises(ValueError, match="node_mus"):
        solve_section_05_wavefront(np.zeros((3, 3)), coordinates, np.ones(3))
