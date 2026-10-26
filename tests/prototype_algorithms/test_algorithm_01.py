import numpy as np
import pytest

from clean_core.config import D_SPATIAL, POLE
from clean_core.geometry.lorentz import minkowski_norm_sq
from clean_core.prototypes.algorithms.algorithm_01 import (
    algorithm_01_multimodal_dirichlet_sensory_grounding,
)


def _sample_inputs():
    image = np.linspace(0.0, 1.0, 2 * 4 * 5).reshape(2, 4, 5)
    audio = np.linspace(-1.0, 1.0, 32)
    text = np.array([2, 17, 32, 4, 19], dtype=np.int64)
    return image, audio, text


def test_algorithm_01_combines_tensor_evidence_and_grounds_on_hyperboloid():
    image, audio, text = _sample_inputs()
    result = algorithm_01_multimodal_dirichlet_sensory_grounding(
        image, audio, text
    )

    assert result.posterior_mean.shape == (D_SPATIAL,)
    assert np.all(result.posterior_mean > 0.0)
    assert np.sum(result.posterior_mean) == pytest.approx(1.0)
    assert result.grounded_coordinate.shape == POLE.shape
    assert result.grounded_coordinate[0] > 0.0
    assert minkowski_norm_sq(result.grounded_coordinate) == pytest.approx(
        -1.0, abs=1e-10
    )


def test_algorithm_01_evidence_is_bounded_and_modality_weights_are_used():
    image, audio, text = _sample_inputs()
    weights = {"visual": 1.0, "audio": 0.0, "text": 0.0}
    result = algorithm_01_multimodal_dirichlet_sensory_grounding(
        image,
        audio,
        text,
        modality_weights=weights,
        max_evidence=0.25,
    )

    prior_mean = np.full(D_SPATIAL, 1.0 / D_SPATIAL)
    maximum_l1_shift = 2.0 * 0.25 / (D_SPATIAL + 0.25)
    assert np.linalg.norm(result.posterior_mean - prior_mean, ord=1) <= (
        maximum_l1_shift + 1e-12
    )


@pytest.mark.parametrize(
    ("image", "audio", "text", "message"),
    [
        (np.zeros((2, 3)), [0.0], [1], "image"),
        (np.zeros((1, 2, 2)), [[0.0]], [1], "audio"),
        (np.zeros((1, 2, 2)), [0.0], [], "text"),
        (np.zeros((1, 2, 2)), [np.nan], [1], "finite"),
        (np.zeros((1, 2, 2)), [0.0], [1.5], "integer token"),
        (np.zeros((1, 2, 2)), [0.0], [-1], "non-negative"),
    ],
)
def test_algorithm_01_rejects_invalid_tensor_inputs(image, audio, text, message):
    with pytest.raises(ValueError, match=message):
        algorithm_01_multimodal_dirichlet_sensory_grounding(image, audio, text)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"max_evidence": 1.1}, "max_evidence"),
        ({"alpha_params": [1.0]}, "match the modality class count"),
        ({"modality_weights": {"visual": 1.0}}, "one weight for each modality"),
    ],
)
def test_algorithm_01_validates_dirichlet_configuration(options, message):
    with pytest.raises(ValueError, match=message):
        algorithm_01_multimodal_dirichlet_sensory_grounding(
            *_sample_inputs(), **options
        )
