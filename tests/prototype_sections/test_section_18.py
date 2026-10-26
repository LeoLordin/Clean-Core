import numpy as np
import pytest

from clean_core.config import D_SPATIAL, POLE
from clean_core.geometry.lorentz import minkowski_norm_sq
from clean_core.prototypes.sections.section_18 import (
    multimodal_dirichlet_grounding,
    multimodal_dirichlet_posterior,
)


def test_multimodal_dirichlet_posterior_accumulates_weighted_modality_evidence():
    scores = {
        "visual": [3.0, 0.0, -1.0],
        "text": [-1.0, 2.0, 0.0],
        "audio": [0.0, -2.0, 1.0],
    }
    alpha = np.array([2.0, 1.0, 3.0])
    modality_weights = {"visual": 2.0, "text": 1.0, "audio": 1.0}
    evidence_cap = 0.5
    posterior = multimodal_dirichlet_posterior(
        scores,
        alpha_params=alpha,
        modality_weights=modality_weights,
        max_evidence=evidence_cap,
    )

    assert posterior.shape == (3,)
    assert np.all(posterior > 0.0)
    assert np.sum(posterior) == pytest.approx(1.0)
    weights = np.array([2.0, 1.0, 1.0]) / 4.0
    evidence = sum(
        weight * np.exp(values - np.max(values))
        / np.sum(np.exp(values - np.max(values)))
        for weight, values in zip(weights, scores.values())
    )
    expected = alpha + evidence_cap * evidence
    expected /= np.sum(expected)
    np.testing.assert_allclose(posterior, expected)
    prior_mean = alpha / np.sum(alpha)
    assert np.linalg.norm(posterior - prior_mean, ord=1) <= 0.5


def test_equal_modalities_have_bounded_combined_evidence():
    modalities = {
        "visual": [1000.0, -1000.0],
        "text": [-1000.0, 1000.0],
    }
    posterior = multimodal_dirichlet_posterior(
        modalities, alpha_params=[1.0, 1.0], max_evidence=1.0
    )

    np.testing.assert_allclose(posterior, [0.5, 0.5])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"modality_scores": {}}, "non-empty mapping"),
        ({"modality_scores": {"visual": [0.0], "text": [1.0, 2.0]}}, "same number"),
        ({"modality_scores": {"visual": [np.nan]}}, "finite"),
        (
            {"modality_scores": {"visual": [0.0]}, "alpha_params": [0.0]},
            "positive",
        ),
        (
            {"modality_scores": {"visual": [0.0]}, "alpha_params": [-1.0]},
            "positive",
        ),
        (
            {"modality_scores": {"visual": [0.0]}, "max_evidence": 1.1},
            "max_evidence",
        ),
        (
            {
                "modality_scores": {"visual": [0.0], "text": [1.0]},
                "modality_weights": {"visual": 1.0},
            },
            "one weight",
        ),
        (
            {
                "modality_scores": {"visual": [0.0]},
                "modality_weights": {"visual": 0.0},
            },
            "positive finite sum",
        ),
    ],
)
def test_multimodal_dirichlet_posterior_rejects_invalid_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        multimodal_dirichlet_posterior(**kwargs)


def test_multimodal_grounding_maps_posterior_mean_to_unit_hyperboloid():
    posterior = multimodal_dirichlet_posterior(
        {"visual": np.linspace(-1.0, 1.0, D_SPATIAL)}
    )
    grounded = multimodal_dirichlet_grounding(
        {"visual": np.linspace(-1.0, 1.0, D_SPATIAL)}
    )

    assert grounded.shape == POLE.shape
    assert grounded[0] > 0.0
    assert minkowski_norm_sq(grounded) == pytest.approx(-1.0, abs=1e-10)
    np.testing.assert_allclose(
        grounded[1:] / np.linalg.norm(grounded[1:]),
        posterior / np.linalg.norm(posterior),
    )


def test_multimodal_grounding_requires_configured_spatial_dimension():
    with pytest.raises(ValueError, match="exactly"):
        multimodal_dirichlet_grounding({"visual": [0.0, 1.0]})
