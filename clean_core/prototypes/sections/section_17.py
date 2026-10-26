"""Standalone categorical target-divergence primitive for Section 17.

This computes the KL divergence between caller-supplied distributions; it does
not predict future states or implement an active-inference agent.
"""

import numpy as np

from clean_core.config import FP64


def target_divergence(predicted_state_distribution, target_distribution):
    """Compute KL(predicted || target), normalizing each positive-mass vector.

    A predicted zero contributes nothing; predicted mass on a zero-probability
    target state yields infinite divergence.
    """
    try:
        predicted_input = np.asarray(predicted_state_distribution)
        target_input = np.asarray(target_distribution)
        if np.iscomplexobj(predicted_input) or np.iscomplexobj(target_input):
            raise ValueError("distributions must be real-valued")
        predicted = np.asarray(predicted_input, dtype=FP64)
        target = np.asarray(target_input, dtype=FP64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("distributions must be real numeric vectors") from error
    if (
        predicted.ndim != 1
        or target.shape != predicted.shape
        or predicted.size == 0
    ):
        raise ValueError(
            "predicted_state_distribution and target_distribution must be "
            "non-empty vectors of equal shape"
        )
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(target)):
        raise ValueError("distributions must contain only finite values")
    if np.any(predicted < 0.0) or np.any(target < 0.0):
        raise ValueError("probabilities must be non-negative")

    predicted_scale = np.max(predicted)
    target_scale = np.max(target)
    if predicted_scale == 0.0 or target_scale == 0.0:
        raise ValueError("distributions must have positive total mass")

    predicted_scaled = predicted / predicted_scale
    target_scaled = target / target_scale
    q = predicted_scaled / np.sum(predicted_scaled)
    target_scaled_total = np.sum(target_scaled)

    if np.any((predicted > 0.0) & (target == 0.0)):
        return float("inf")
    nonzero = q > 0.0
    log_q = (
        np.log(predicted[nonzero])
        - np.log(predicted_scale)
        - np.log(np.sum(predicted_scaled))
    )
    log_p = (
        np.log(target[nonzero])
        - np.log(target_scale)
        - np.log(target_scaled_total)
    )
    return float(np.sum(q[nonzero] * (log_q - log_p)))


__all__ = ["target_divergence"]
