import numpy as np
import pytest

from clean_core.prototypes.sections.section_15 import (
    contrastive_information_lower_bound,
    diagonal_gaussian_kl_to_standard_normal,
    variational_information_bottleneck_loss,
)


def test_standard_normal_posterior_has_zero_kl_in_arbitrary_latent_shape():
    assert diagonal_gaussian_kl_to_standard_normal(
        np.zeros((2, 3, 4)), np.ones((2, 3, 4))
    ) == 0.0


def test_diagonal_gaussian_kl_sums_all_latent_dimensions():
    result = diagonal_gaussian_kl_to_standard_normal(
        np.array([[1.0, 0.0]]), np.array([[1.0, 2.0]])
    )

    np.testing.assert_allclose(result, 2.0 - np.log(2.0))


def test_contrastive_term_uses_stable_log_sum_exp():
    result = contrastive_information_lower_bound(
        positive_scores=np.array([1000.0, 1001.0]),
        candidate_scores=np.array([[1000.0, 999.0], [1001.0, 1000.0]]),
    )

    np.testing.assert_allclose(result, -np.log1p(np.exp(-1.0)))


def test_vib_loss_composes_kl_and_contrastive_bound():
    mean = np.array([1.0, 0.0])
    standard_deviation = np.array([1.0, 1.0])
    positive_scores = np.array([2.0, 1.0])
    candidate_scores = np.array([[2.0, 0.0], [1.0, -1.0]])
    beta_info = 0.5

    complexity = 0.5
    contrastive_term = -np.log1p(np.exp(-2.0))
    result = variational_information_bottleneck_loss(
        mean,
        standard_deviation,
        positive_scores,
        candidate_scores,
        beta_info,
    )

    np.testing.assert_allclose(result, complexity - beta_info * contrastive_term)


@pytest.mark.parametrize(
    ("mean", "standard_deviation", "message"),
    [
        ([], [], "non-empty"),
        ([0.0, 1.0], [1.0], "same non-empty shape"),
        ([0.0, np.nan], [1.0, 1.0], "finite"),
        ([0.0], [0.0], "positive"),
    ],
)
def test_kl_rejects_invalid_gaussian_parameters(mean, standard_deviation, message):
    with pytest.raises(ValueError, match=message):
        diagonal_gaussian_kl_to_standard_normal(mean, standard_deviation)


@pytest.mark.parametrize(
    ("positive_scores", "candidate_scores", "message"),
    [
        ([], [[]], "non-empty vector"),
        ([1.0], [1.0], "matrix"),
        ([1.0], [[],], "one non-empty row"),
        ([1.0], [[1.0], [2.0]], "one non-empty row"),
        ([np.inf], [[1.0]], "finite"),
    ],
)
def test_contrastive_bound_rejects_invalid_scores(
    positive_scores, candidate_scores, message
):
    with pytest.raises(ValueError, match=message):
        contrastive_information_lower_bound(positive_scores, candidate_scores)


@pytest.mark.parametrize("beta_info", [-1.0, np.nan, np.inf])
def test_vib_loss_rejects_invalid_information_weight(beta_info):
    with pytest.raises(ValueError, match="beta_info"):
        variational_information_bottleneck_loss(
            mean=[0.0],
            standard_deviation=[1.0],
            positive_scores=[1.0],
            candidate_scores=[[1.0]],
            beta_info=beta_info,
        )
