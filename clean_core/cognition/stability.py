import numpy as np

from clean_core.config import D_LORENTZ, FP64
from clean_core.geometry.lorentz import chordal_distance


def lyapunov_candidate(states, barycenter, actions=None, target_actions=None):
    """Evaluate the non-negative Lyapunov candidate; this does not certify stability."""
    states = np.asarray(states, dtype=FP64)
    barycenter = np.asarray(barycenter, dtype=FP64)
    if states.ndim != 2 or states.shape[1] != D_LORENTZ or states.shape[0] == 0:
        raise ValueError(f"states must have shape (n > 0, {D_LORENTZ})")
    if barycenter.shape != (D_LORENTZ,):
        raise ValueError(f"barycenter must have shape ({D_LORENTZ},)")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(barycenter)):
        raise ValueError("states and barycenter must contain only finite values")

    energy = 0.5 * sum(chordal_distance(state, barycenter) ** 2 for state in states)
    if (actions is None) != (target_actions is None):
        raise ValueError("actions and target_actions must be provided together")
    if actions is not None:
        actions = np.asarray(actions, dtype=FP64)
        target_actions = np.asarray(target_actions, dtype=FP64)
        if actions.shape != target_actions.shape or not np.all(np.isfinite(actions)) or not np.all(np.isfinite(target_actions)):
            raise ValueError("actions and target_actions must be matching finite arrays")
        energy += 0.5 * float(np.sum(np.square(actions - target_actions)))
    return float(energy)