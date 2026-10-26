"""Small end-to-end runtime joining sensory grounding to geodesic inference.

The theory's algorithms cover distinct processes and timescales; this runtime
connects the two compatible online stages without pretending to execute the
entire specification as one validated cognitive architecture.
"""

from dataclasses import dataclass

from clean_core.core.neuron import NeuronMicrostate
from clean_core.prototypes.algorithms.algorithm_01 import (
    Algorithm01Output,
    algorithm_01_multimodal_dirichlet_sensory_grounding,
)
from clean_core.prototypes.algorithms.algorithm_02 import (
    Algorithm02Result,
    run_algorithm_02,
)


@dataclass(frozen=True)
class SensoryInferenceResult:
    """Outputs from one composed grounding and inference step."""

    grounding: Algorithm01Output
    inference: Algorithm02Result


def sensory_to_inference_step(
    neuron: NeuronMicrostate,
    image,
    audio,
    text,
    *,
    dt=0.01,
    alpha_damp=0.05,
    alpha_params=None,
    modality_weights=None,
    max_evidence=1.0,
) -> SensoryInferenceResult:
    """Ground one numeric multimodal observation and advance one neuron.

    The Section 18 Dirichlet posterior from Algorithm 1 supplies the
    ``D_SPATIAL`` context, and its Lorentz-grounded coordinate supplies the
    barycenter for Algorithm 2. Image/audio/text encoders remain deterministic
    fixed numeric adapters; this composition is a prototype, not a learned
    multimodal model.
    """
    grounding = algorithm_01_multimodal_dirichlet_sensory_grounding(
        image,
        audio,
        text,
        alpha_params=alpha_params,
        modality_weights=modality_weights,
        max_evidence=max_evidence,
    )
    inference = run_algorithm_02(
        neuron,
        grounding.posterior_mean,
        grounding.grounded_coordinate,
        dt=dt,
        alpha_damp=alpha_damp,
    )
    return SensoryInferenceResult(grounding=grounding, inference=inference)


__all__ = ["SensoryInferenceResult", "sensory_to_inference_step"]
