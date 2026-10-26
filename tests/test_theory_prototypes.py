import numpy as np
import pytest

from clean_core.cognition.cole_hopf import (
    cole_hopf_graph_diffusion_step,
    superpose_goal_wavefunctions,
    value_to_wavefunction,
    wavefunction_to_value,
)
from clean_core.cognition.engine_runtime import AtomicEngineSlot, MultiTimescaleDispatcher
from clean_core.cognition.polyhedral_aixi import PolyhedralAIXIPrototype
from clean_core.cognition.theory_of_mind import TheoryOfMindPlanner
from clean_core.config import POLE
from clean_core.estimation.vib import (
    contrastive_information_lower_bound,
    diagonal_gaussian_kl_to_standard_normal,
    variational_information_bottleneck_loss,
)
from clean_core.geometry.curvature import SectionalCurvatureAdapter, constant_curvature_distance
from clean_core.geometry.dimension_expansion import (
    expand_hyperboloid_embedding,
    minkowski_gram_matrix,
)
from clean_core.geometry.hamiltonian import conformal_hamiltonian_vector_field
from clean_core.geometry.lorentz import exp_map
from clean_core.interfaces.motor_synthesis import polyhedral_cole_hopf_motor_step
from clean_core.memory.harish_chandra import HarishChandraHolographicMemory


def test_cole_hopf_transform_superposition_and_graph_diffusion():
    values = np.array([0.1, 0.3, 0.8])
    wavefunction = value_to_wavefunction(values, 1.2, 0.5)
    np.testing.assert_allclose(wavefunction_to_value(wavefunction, 1.2, 0.5), values)
    np.testing.assert_allclose(superpose_goal_wavefunctions(np.stack([wavefunction])), wavefunction)

    edges = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    np.testing.assert_allclose(
        cole_hopf_graph_diffusion_step(np.ones(3), edges, np.zeros(3), 0.1, 1.0, 1.0),
        np.ones(3),
    )
    with pytest.raises(ValueError, match="positivity bound"):
        cole_hopf_graph_diffusion_step(np.ones(3), edges, np.zeros(3), 3.0, 1.0, 1.0)


def test_polyhedral_aixi_bounds_only_fix_action_after_separation():
    decider = PolyhedralAIXIPrototype(2, 1, r_max=1.0, discount=0.5)
    decider.add_candidate("0", [[1.0], [0.0]])
    decider.add_candidate("10", [[0.0], [2.0]])
    decider.add_candidate("11", [[0.0], [2.0]])

    first = decider.advance()
    assert not first["bounds_separate"]
    assert first["unresolved_mass"] == 0.5

    result = decider.advance(2)
    assert result["bounds_separate"]
    assert result["action"] == 1
    assert result["unresolved_mass"] == 0.0
    assert np.linalg.norm(result["motor_pressure"]) > 0.0

    with pytest.raises(ValueError, match="prefix-free"):
        decider.add_candidate("1", [[0.0], [0.0]])


def test_vib_primitives_match_the_written_objective():
    assert diagonal_gaussian_kl_to_standard_normal(np.zeros(2), np.ones(2)) == 0.0
    scores = contrastive_information_lower_bound(
        np.array([2.0, 2.0]), np.array([[2.0, 0.0], [2.0, 0.0]])
    )
    np.testing.assert_allclose(scores, -np.log1p(np.exp(-2.0)))
    assert np.isfinite(
        variational_information_bottleneck_loss(
            np.zeros(2),
            np.ones(2),
            np.array([2.0, 2.0]),
            np.array([[2.0, 0.0], [2.0, 0.0]]),
            0.1,
        )
    )


def test_theory_of_mind_planner_uses_normalized_action_beliefs():
    planner = TheoryOfMindPlanner()
    utilities = np.array([[[1.0, 0.0]], [[0.0, 3.0]]])
    action, scores = planner.choose_action([0.2, 0.1], [np.array([1.0, 3.0])], utilities)

    np.testing.assert_allclose(scores, [0.45, 2.35])
    assert action == 1


def test_conformal_hamiltonian_field_preserves_cotangent_constraint():
    metric = np.diag([-1.0] + [1.0] * (POLE.size - 1))
    momentum = np.zeros_like(POLE)
    momentum[1] = 1.0
    zero = np.zeros_like(POLE)

    x_dot, p_dot, _ = conformal_hamiltonian_vector_field(
        POLE, momentum, 2.0, zero, zero, kappa_v=0.5
    )
    np.testing.assert_allclose(x_dot[1], 0.375)
    np.testing.assert_allclose(POLE @ metric @ x_dot, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        POLE @ metric @ p_dot + x_dot @ metric @ momentum, 0.0, atol=1e-12
    )


def test_curvature_distances_and_bounded_adapter_update():
    x = np.array([0.1, 0.0])
    y = np.array([0.0, 0.0])
    assert constant_curvature_distance(-1.0, x, y) > 0.0
    assert constant_curvature_distance(1.0, x, y) > 0.0
    np.testing.assert_allclose(constant_curvature_distance(0.0, x, y), 0.2)

    adapter = SectionalCurvatureAdapter(0.5, 0.1)
    assert adapter.update(2.0, bounds=(-1.0, 1.0)) == 0.3


def test_hyperboloid_dimension_expansion_preserves_minkowski_gram_matrix():
    tangent = np.zeros_like(POLE)
    tangent[1] = 0.4
    points = np.stack([POLE, exp_map(POLE.copy(), tangent)])

    expanded = expand_hyperboloid_embedding(points, 4)
    assert expanded.shape == (2, POLE.size + 4)
    np.testing.assert_allclose(
        minkowski_gram_matrix(points), minkowski_gram_matrix(expanded), atol=1e-12
    )


def test_holographic_memory_retrieval_is_bounded_and_validates_empty_queries():
    memory = HarishChandraHolographicMemory(capacity=2)
    first = POLE.copy()
    second = POLE.copy()
    second[1] = 0.5
    second[0] = np.sqrt(1.25)

    memory.store(first, [1.0, 0.0])
    memory.store(second, [0.0, 1.0])
    np.testing.assert_allclose(memory.retrieve(first, temperature=0.02), [1.0, 0.0], atol=1e-6)
    memory.store(POLE, [2.0, 2.0])
    assert len(memory._values) == 2

    empty_memory = HarishChandraHolographicMemory()
    with pytest.raises(ValueError, match="temperature"):
        empty_memory.retrieve(POLE, temperature=0.0)
    with pytest.raises(ValueError, match="mu_point"):
        empty_memory.retrieve(np.zeros(2))


def test_motor_synthesis_combines_inputs_and_clips_action():
    action, target = polyhedral_cole_hopf_motor_step(
        [0.0], [1.0], [0.5], [-0.2], [[1.0]], [[2.0]], [[1.0]], [0.0],
        dt=0.1, learning_rate=1.0,
    )

    assert 0.0 < action[0] <= 1.0
    assert -1.0 < target[0] < 1.0


def test_engine_publication_and_timescale_dispatch():
    slot = AtomicEngineSlot(lambda value: value + 1)
    assert slot.dispatch(2) == 3
    changed, version = slot.swap_if_better(
        lambda value: value + 3,
        lambda engine: engine(0),
    )
    assert changed and version == 1
    assert slot.dispatch(2) == 5

    dispatcher = MultiTimescaleDispatcher(
        {"fast": lambda value: value + 1, "slow": lambda value: value - 1}
    )
    assert dispatcher.dispatch("fast", 2) == 3
    assert dispatcher.publish("fast", lambda value: value * 4) == 1
    assert dispatcher.dispatch("fast", 2) == 8
