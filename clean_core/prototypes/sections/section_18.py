"""Bounded multimodal Dirichlet grounding for Section 18.

Inputs are numeric class-evidence scores supplied per modality; this prototype
does not implement image, audio, or text encoders. Each modality contributes a
categorical evidence distribution, whose total concentration is bounded
before applying a Dirichlet prior.
"""

from collections.abc import Mapping

import numpy as np

from clean_core.config import D_SPATIAL, FP64, POLE
from clean_core.geometry.lorentz import exp_map


def _finite_vector(value, name):
    vector = np.asarray(value, dtype=FP64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def multimodal_dirichlet_posterior(
    modality_scores,
    alpha_params=None,
    modality_weights=None,
    max_evidence=1.0,
):
    """Combine numeric modality scores into Dirichlet posterior mean probabilities.

    Each modality's scores are independently softmax-normalized to form its
    evidence distribution. Reliability weights are normalized to sum to one,
    so combined evidence has concentration ``max_evidence`` regardless of
    modality count.
    """
    if not isinstance(modality_scores, Mapping) or not modality_scores:
        raise ValueError("modality_scores must be a non-empty mapping")

    names = list(modality_scores)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("modality names must be non-empty strings")

    vectors = [
        _finite_vector(modality_scores[name], f"{name} scores") for name in names
    ]
    class_count = vectors[0].size
    if any(vector.size != class_count for vector in vectors):
        raise ValueError("all modality scores must have the same number of classes")

    if alpha_params is None:
        alpha = np.ones(class_count, dtype=FP64)
    else:
        alpha = _finite_vector(alpha_params, "alpha_params")
        if alpha.size != class_count:
            raise ValueError("alpha_params must match the modality class count")
        if np.any(alpha <= 0.0):
            raise ValueError("alpha_params must contain only positive values")
    alpha_total = float(np.sum(alpha))
    if not np.isfinite(alpha_total):
        raise ValueError("alpha_params must have a finite sum")

    try:
        evidence_cap = float(max_evidence)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("max_evidence must be finite and in (0, 1]") from error
    if not np.isfinite(evidence_cap) or not 0.0 < evidence_cap <= 1.0:
        raise ValueError("max_evidence must be finite and in (0, 1]")

    if modality_weights is None:
        weights = np.ones(len(names), dtype=FP64)
    else:
        if not isinstance(modality_weights, Mapping) or set(modality_weights) != set(
            names
        ):
            raise ValueError(
                "modality_weights must provide one weight for each modality"
            )
        try:
            weights = np.asarray([modality_weights[name] for name in names], dtype=FP64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "modality_weights must be finite non-negative values"
            ) from error
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("modality_weights must be finite non-negative values")
    weight_total = float(np.sum(weights))
    if not np.isfinite(weight_total) or weight_total <= 0.0:
        raise ValueError("modality_weights must have a positive finite sum")
    weights /= weight_total

    evidence = np.zeros(class_count, dtype=FP64)
    for weight, vector in zip(weights, vectors):
        probabilities = np.exp(vector - np.max(vector))
        probabilities /= np.sum(probabilities)
        evidence += weight * probabilities

    posterior_alpha = alpha + evidence_cap * evidence
    posterior_total = float(np.sum(posterior_alpha))
    if not np.isfinite(posterior_total):
        raise ValueError("combined posterior concentration must be finite")
    return posterior_alpha / posterior_total


def multimodal_dirichlet_grounding(
    modality_scores,
    alpha_params=None,
    modality_weights=None,
    max_evidence=1.0,
):
    """Map the multimodal posterior mean to the configured sensory hyperboloid."""
    posterior = multimodal_dirichlet_posterior(
        modality_scores,
        alpha_params=alpha_params,
        modality_weights=modality_weights,
        max_evidence=max_evidence,
    )
    if posterior.size != D_SPATIAL:
        raise ValueError(f"grounding requires exactly {D_SPATIAL} classes")
    tangent = np.zeros_like(POLE, dtype=FP64)
    tangent[1:] = posterior
    return exp_map(POLE, tangent)


__all__ = [
    "multimodal_dirichlet_grounding",
    "multimodal_dirichlet_posterior",
]
