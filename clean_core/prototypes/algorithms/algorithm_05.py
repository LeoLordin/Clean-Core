"""Bounded finite-graph motor synthesis for Algorithm 5.

This prototype combines the Sections 4, 6, and 27 APIs on finite supplied
inputs. Its graph differences are not a continuous PDE solver, its decider is
not classical AIXI, and its bounded motor update is not a full EFE gradient.
"""

from dataclasses import dataclass
import operator

import numpy as np

from clean_core.config import FP64
from clean_core.prototypes.sections.section_04 import (
    linearized_hjb_graph_step,
    optimal_geodesic_velocity,
    superpose_goal_wavefunctions,
)
from clean_core.prototypes.sections.section_06 import Section06PolyhedralDecider
from clean_core.prototypes.sections.section_27 import ConformalTheoryOfMindPlanner


@dataclass(frozen=True)
class Algorithm05Result:
    """Finite intermediate signals and bounded motor actions."""

    goal_wavefunction: np.ndarray
    propagated_wavefunction: np.ndarray
    graph_log_gradient: np.ndarray
    geodesic_signal: np.ndarray
    discrete_action: int
    polyhedral_pressure: np.ndarray
    polyhedral_safe_radius: float
    polyhedral_separation: float
    social_action: np.ndarray
    social_scores: np.ndarray
    target_action: np.ndarray
    motor_velocity: np.ndarray
    updated_action: np.ndarray


def _finite_array(value, name):
    try:
        array = np.asarray(value, dtype=FP64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite numeric array") from error
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _finite_positive_scalar(value, name):
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite positive scalar") from error
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return scalar


def _graph_log_gradient(wavefunction, edge_weights, coordinates, state_index):
    """Estimate a local log-wavefunction gradient from finite graph edges."""
    neighbors = np.flatnonzero(edge_weights[state_index] > 0.0)
    if neighbors.size == 0:
        return np.zeros(coordinates.shape[1], dtype=FP64)

    with np.errstate(over="ignore", invalid="ignore"):
        displacements = coordinates[neighbors] - coordinates[state_index]
    if not np.all(np.isfinite(displacements)):
        raise ValueError("finite graph edge displacements must be finite")
    log_differences = np.log(wavefunction[neighbors]) - np.log(
        wavefunction[state_index]
    )
    try:
        gradient, *_ = np.linalg.lstsq(displacements, log_differences, rcond=None)
    except np.linalg.LinAlgError as error:
        raise ValueError("finite graph gradient could not be estimated") from error
    if not np.all(np.isfinite(gradient)):
        raise ValueError("finite graph gradient must be finite")
    return gradient


def synthesize_motor_action(
    goal_wavefunctions,
    edge_weights,
    pragmatic_potential,
    node_coordinates,
    top_layer_coordinate,
    frame,
    lorentz_metric,
    *,
    lambda_value,
    sigma,
    graph_dt,
    polyhedral_decider,
    optimal_action,
    social_planner,
    social_curiosity,
    other_action_beliefs,
    social_utility_tensor,
    motor_weights,
    polyhedral_weights,
    social_weights,
    motor_bias,
    current_action,
    motor_dt=0.01,
    motor_learning_rate=1.0,
    goal_weights=None,
):
    """Synthesize a target and one bounded Euler motor update.

    ``goal_wavefunctions`` has shape ``(goals, nodes)``; graph signals and
    ``pragmatic_potential`` have one entry per node. ``node_coordinates`` has
    shape ``(nodes, coordinate_dim)`` and the top-layer coordinate is mapped
    to its nearest supplied graph sample in ordinary Euclidean coordinates.
    The local log-gradient is a least-squares finite-edge estimate.

    ``motor_weights``, ``polyhedral_weights``, and ``social_weights`` have
    shapes ``(motor_dim, geodesic_dim)``, ``(motor_dim, discrete_actions)``,
    and ``(motor_dim, social_actions)`` respectively. The Section 6 action
    must match ``optimal_action``; Section 27's selected social action is
    encoded as a one-hot vector. The update is gradient descent on a quadratic
    target-tracking proxy, not the complete expected-free-energy gradient.
    """
    goals = _finite_array(goal_wavefunctions, "goal_wavefunctions")
    coordinates = _finite_array(node_coordinates, "node_coordinates")
    top_coordinate = _finite_array(top_layer_coordinate, "top_layer_coordinate")
    adjacency = _finite_array(edge_weights, "edge_weights")
    potential = _finite_array(pragmatic_potential, "pragmatic_potential")

    if goals.ndim != 2 or goals.shape[0] == 0 or goals.shape[1] == 0:
        raise ValueError("goal_wavefunctions must have shape (goals, nodes)")
    node_count = goals.shape[1]
    if coordinates.ndim != 2 or coordinates.shape[0] != node_count or coordinates.shape[1] == 0:
        raise ValueError("node_coordinates must have shape (nodes, coordinate_dim)")
    if top_coordinate.shape != (coordinates.shape[1],):
        raise ValueError("top_layer_coordinate must match the coordinate dimension")
    if adjacency.shape != (node_count, node_count):
        raise ValueError("edge_weights must be square and match the goal wavefunctions")
    if potential.shape != (node_count,):
        raise ValueError("pragmatic_potential must have one value per graph node")

    with np.errstate(over="ignore", invalid="ignore"):
        squared_distances = np.sum(np.square(coordinates - top_coordinate), axis=1)
    if not np.all(np.isfinite(squared_distances)):
        raise ValueError("top-layer coordinate distances must be finite")
    state_index = int(np.argmin(squared_distances))

    if not isinstance(polyhedral_decider, Section06PolyhedralDecider):
        raise ValueError("polyhedral_decider must be a Section06PolyhedralDecider")
    try:
        selected_action = operator.index(optimal_action)
    except TypeError as error:
        raise ValueError("optimal_action must be an integer action index") from error
    if isinstance(optimal_action, (bool, np.bool_)):
        raise ValueError("optimal_action must be an integer action index")
    if not 0 <= selected_action < polyhedral_decider.num_actions:
        raise ValueError("optimal_action is outside the polyhedral action range")

    if not isinstance(social_planner, ConformalTheoryOfMindPlanner):
        raise ValueError("social_planner must be a ConformalTheoryOfMindPlanner")
    graph_dt = _finite_positive_scalar(graph_dt, "graph_dt")
    motor_dt = _finite_positive_scalar(motor_dt, "motor_dt")
    motor_learning_rate = _finite_positive_scalar(
        motor_learning_rate, "motor_learning_rate"
    )

    goal_wavefunction = superpose_goal_wavefunctions(goals, goal_weights)
    propagated_wavefunction = linearized_hjb_graph_step(
        goal_wavefunction,
        adjacency,
        potential,
        graph_dt,
        lambda_value,
        sigma,
    )
    graph_gradient = _graph_log_gradient(
        propagated_wavefunction, adjacency, coordinates, state_index
    )
    geodesic_signal = optimal_geodesic_velocity(
        graph_gradient, frame, lorentz_metric, lambda_value, sigma
    )

    polyhedral_result = polyhedral_decider.decision()
    if polyhedral_result["action"] != selected_action:
        raise ValueError(
            "optimal_action must match the Section 6 decider's selected action"
        )
    polyhedral_pressure = np.asarray(
        polyhedral_result["motor_pressure"], dtype=FP64
    )

    social_index, social_scores = social_planner.choose_action(
        social_curiosity, other_action_beliefs, social_utility_tensor
    )
    social_action = np.zeros(social_scores.size, dtype=FP64)
    social_action[social_index] = 1.0

    motor_matrix = _finite_array(motor_weights, "motor_weights")
    polyhedral_matrix = _finite_array(polyhedral_weights, "polyhedral_weights")
    social_matrix = _finite_array(social_weights, "social_weights")
    bias = _finite_array(motor_bias, "motor_bias")
    current = _finite_array(current_action, "current_action")
    if motor_matrix.ndim != 2 or motor_matrix.shape[1] != geodesic_signal.size:
        raise ValueError("motor_weights must have shape (motor_dim, geodesic_dim)")
    motor_dim = motor_matrix.shape[0]
    if motor_dim == 0 or bias.shape != (motor_dim,):
        raise ValueError("motor_bias must have one entry per motor dimension")
    if current.shape != (motor_dim,) or np.any(current < -1.0) or np.any(current > 1.0):
        raise ValueError("current_action must lie in [-1, 1] with shape (motor_dim,)")
    if polyhedral_matrix.shape != (motor_dim, polyhedral_decider.num_actions):
        raise ValueError("polyhedral_weights must map discrete actions to motor dimensions")
    if social_matrix.shape != (motor_dim, social_action.size):
        raise ValueError("social_weights must map social actions to motor dimensions")
    if polyhedral_pressure.shape != (polyhedral_decider.num_actions,):
        raise ValueError("Section 6 motor pressure has an invalid shape")

    with np.errstate(over="ignore", invalid="ignore"):
        motor_drive = (
            motor_matrix @ geodesic_signal
            + polyhedral_matrix @ polyhedral_pressure
            + social_matrix @ social_action
            + bias
        )
    if not np.all(np.isfinite(motor_drive)):
        raise ValueError("combined motor drive must be finite")
    target_action = np.tanh(motor_drive)
    with np.errstate(over="ignore", invalid="ignore"):
        motor_velocity = motor_learning_rate * (target_action - current)
        proposed_action = current + motor_dt * motor_velocity
    if not np.all(np.isfinite(motor_velocity)) or not np.all(np.isfinite(proposed_action)):
        raise ValueError("motor update must remain finite")
    updated_action = np.clip(proposed_action, -1.0, 1.0)

    return Algorithm05Result(
        goal_wavefunction=goal_wavefunction.copy(),
        propagated_wavefunction=propagated_wavefunction.copy(),
        graph_log_gradient=graph_gradient.copy(),
        geodesic_signal=geodesic_signal.copy(),
        discrete_action=selected_action,
        polyhedral_pressure=polyhedral_pressure.copy(),
        polyhedral_safe_radius=float(polyhedral_result["safe_radius"]),
        polyhedral_separation=float(polyhedral_result["separation"]),
        social_action=social_action,
        social_scores=social_scores.copy(),
        target_action=target_action,
        motor_velocity=motor_velocity,
        updated_action=updated_action,
    )


__all__ = ["Algorithm05Result", "synthesize_motor_action"]
