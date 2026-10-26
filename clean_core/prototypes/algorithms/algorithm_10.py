"""Inference-only query relaxation with bounded holographic associative recall.

This is a finite prototype of Algorithm 10. The repository does not provide
the learned motor decoder referenced by the theory, so the default target
response is a deterministic feature-space proxy rather than a semantic action.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from clean_core.config import D_LORENTZ, ETA_L, FP64, POLE
from clean_core.core.inference import system1_query_relaxation
from clean_core.core.neuron import NeuronMicrostate
from clean_core.prototypes.sections.section_26 import Section26HolographicMemory

RELAXATION_STEPS = 4


@dataclass(frozen=True)
class Algorithm10Result:
    """Finite inference and recall outputs for one query."""

    relaxed_states: np.ndarray
    errors: tuple[np.ndarray, ...]
    stationary_state: np.ndarray
    associative_recall: np.ndarray
    target_action: np.ndarray


def _hyperboloid_point(value, name):
    point = np.asarray(value, dtype=FP64)
    if point.shape != (D_LORENTZ,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be a finite vector with shape ({D_LORENTZ},)")
    norm_sq = float(np.sum(point * ETA_L * point))
    if (
        point[0] <= 0.0
        or not np.isfinite(norm_sq)
        or not np.isclose(norm_sq, -1.0, rtol=1e-9, atol=1e-9)
    ):
        raise ValueError(f"{name} must lie on the future unit Lorentz hyperboloid")
    return point.copy()


def _finite_action(value):
    action = np.asarray(value, dtype=FP64)
    if action.ndim != 1 or action.size == 0 or not np.all(np.isfinite(action)):
        raise ValueError("motor_decoder must return a non-empty finite vector")
    return action.copy()


def run_algorithm_10(
    neurons: Sequence[NeuronMicrostate],
    query_mu,
    barycenter_mu,
    memory: Section26HolographicMemory,
    *,
    hbar_cog=1.0,
    motor_decoder: Callable[[np.ndarray], np.ndarray] | None = None,
):
    """Relax a pinned query for four steps, then retrieve bounded memory.

    ``neurons`` are updated only through :func:`system1_query_relaxation`:
    their inference state may change, but model weight matrices are not
    adapted. ``memory`` must be a bounded Section 26 memory instance.

    If supplied, ``motor_decoder`` receives the concatenation of the
    stationary spatial coordinates and the real and imaginary recall parts.
    Without a decoder this same feature vector is returned as a finite target
    response proxy; it makes no semantic correctness claim.
    """
    try:
        neuron_list = tuple(neurons)
    except TypeError as error:
        raise ValueError("neurons must be a non-empty sequence") from error
    if not neuron_list or any(not isinstance(n, NeuronMicrostate) for n in neuron_list):
        raise ValueError("neurons must contain at least one NeuronMicrostate")
    if not isinstance(memory, Section26HolographicMemory):
        raise ValueError("memory must be a Section26HolographicMemory")

    query = _hyperboloid_point(query_mu, "query_mu")
    barycenter = _hyperboloid_point(barycenter_mu, "barycenter_mu")

    states, errors = system1_query_relaxation(
        neuron_list,
        query,
        barycenter,
        steps=RELAXATION_STEPS,
    )
    if states.shape != (len(neuron_list), D_LORENTZ) or not np.all(np.isfinite(states)):
        raise ValueError("query relaxation produced invalid states")
    stationary_state = states[-1].copy()
    recall = memory.recall(stationary_state, hbar_cog=hbar_cog)
    if recall.ndim != 1 or not np.all(np.isfinite(recall)):
        raise ValueError("associative recall must be a finite vector")

    features = np.concatenate(
        (stationary_state[1:], np.real(recall), np.imag(recall))
    ).astype(FP64, copy=False)
    if not np.all(np.isfinite(features)):
        raise ValueError("target response features must be finite")
    target_action = (
        features.copy()
        if motor_decoder is None
        else _finite_action(motor_decoder(features.copy()))
    )

    return Algorithm10Result(
        relaxed_states=states.copy(),
        errors=tuple(np.asarray(error).copy() for error in errors),
        stationary_state=stationary_state,
        associative_recall=recall.copy(),
        target_action=target_action,
    )


__all__ = ["Algorithm10Result", "RELAXATION_STEPS", "run_algorithm_10"]
