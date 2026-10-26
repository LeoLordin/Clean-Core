"""Deterministic numeric-tensor prototype for multimodal sensory grounding.

The fixed pooling and token bucketing below are input adapters, not learned
encoders. The result is a geometric prototype; it does not establish semantic
alignment or estimate the contrastive mutual-information expression in
Section 18.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from clean_core.config import D_SPATIAL, FP64, POLE
from clean_core.geometry.lorentz import exp_map
from clean_core.prototypes.sections.section_18 import (
    multimodal_dirichlet_posterior,
)


@dataclass(frozen=True)
class Algorithm01Output:
    """Posterior summary and grounded Lorentz coordinate."""

    grounded_coordinate: NDArray[np.float64]
    posterior_mean: NDArray[np.float64]


def _numeric_tensor(value, name: str, ndim: int) -> NDArray[np.float64]:
    raw = np.asarray(value)
    if raw.ndim != ndim or raw.size == 0:
        raise ValueError(f"{name} must be a non-empty {ndim}-dimensional tensor")
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(
        raw.dtype, np.complexfloating
    ):
        raise ValueError(f"{name} must contain real numeric values")
    try:
        tensor = np.asarray(raw, dtype=FP64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain real numeric values") from error
    if not np.all(np.isfinite(tensor)):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _fixed_mean_pool(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Scale safely and mean-pool flattened values into the spatial dimension."""
    flattened = values.reshape(-1)
    scale = float(np.max(np.abs(flattened)))
    if scale > 0.0:
        flattened = flattened / scale
    pooled = np.zeros(D_SPATIAL, dtype=FP64)
    for index, chunk in enumerate(np.array_split(flattened, D_SPATIAL)):
        if chunk.size:
            pooled[index] = np.mean(chunk)
    return pooled


def _text_bucket_logits(token_ids) -> NDArray[np.float64]:
    tokens = np.asarray(token_ids)
    if (
        tokens.ndim != 1
        or tokens.size == 0
        or not np.issubdtype(tokens.dtype, np.integer)
        or np.issubdtype(tokens.dtype, np.bool_)
    ):
        raise ValueError("text must be a non-empty one-dimensional integer token array")
    if np.issubdtype(tokens.dtype, np.signedinteger) and np.any(tokens < 0):
        raise ValueError("text token IDs must be non-negative")
    buckets = np.remainder(tokens, D_SPATIAL).astype(np.intp, copy=False)
    counts = np.bincount(buckets, minlength=D_SPATIAL)
    return np.log(counts.astype(FP64) + 1e-12)


def algorithm_01_multimodal_dirichlet_sensory_grounding(
    image,
    audio,
    text,
    *,
    alpha_params=None,
    modality_weights=None,
    max_evidence: float = 1.0,
) -> Algorithm01Output:
    """Ground numeric image, audio, and token tensors on the Lorentz hyperboloid.

    Image tensors must have shape ``(channels, height, width)``; audio and text
    are one-dimensional. Image/audio values use deterministic fixed mean
    pooling, while integer token IDs use modulo buckets. These intentionally
    simple adapters provide equal-width numeric evidence without claiming
    trainable or semantic modality encoders. Combined evidence is capped by
    ``max_evidence`` through the Section 18 Dirichlet posterior API.

    No InfoNCE or mutual-information bound is returned: that calculation
    requires a batch of paired grounded samples and a specified score model.
    """
    image_tensor = _numeric_tensor(image, "image", ndim=3)
    audio_tensor = _numeric_tensor(audio, "audio", ndim=1)
    image_logits = _fixed_mean_pool(image_tensor)
    audio_logits = _fixed_mean_pool(audio_tensor)
    text_logits = _text_bucket_logits(text)

    modality_logits = {
        "visual": image_logits,
        "audio": audio_logits,
        "text": text_logits,
    }
    posterior = multimodal_dirichlet_posterior(
        modality_logits,
        alpha_params=alpha_params,
        modality_weights=modality_weights,
        max_evidence=max_evidence,
    )
    tangent = np.zeros_like(POLE, dtype=FP64)
    tangent[1:] = posterior
    grounded = exp_map(POLE, tangent)
    return Algorithm01Output(
        grounded_coordinate=grounded,
        posterior_mean=posterior,
    )


__all__ = ["Algorithm01Output", "algorithm_01_multimodal_dirichlet_sensory_grounding"]
