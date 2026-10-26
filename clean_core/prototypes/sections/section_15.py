"""Encoder-agnostic variational information bottleneck objective prototype.

The caller supplies diagonal-Gaussian posterior parameters and scores from
their own encoder and contrastive scorer. The module implements only the
Section 15 objective and its numerical input validation. The empirical
contrastive term is not, by itself, a true mutual-information estimator;
interpreting it as an information lower bound requires assumptions not
specified by the formula.
"""

import numpy as np


def diagonal_gaussian_kl_to_standard_normal(mean, standard_deviation):
    """Sum diagonal-Gaussian KL terms over all supplied latent dimensions."""
    mean = np.asarray(mean, dtype=np.float64)
    standard_deviation = np.asarray(standard_deviation, dtype=np.float64)
    if mean.ndim == 0 or mean.size == 0 or standard_deviation.shape != mean.shape:
        raise ValueError("mean and standard_deviation must have the same non-empty shape")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(standard_deviation)):
        raise ValueError("Gaussian parameters must be finite")
    if np.any(standard_deviation <= 0.0):
        raise ValueError("standard_deviation must be positive")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        terms = mean**2 + standard_deviation**2 - 1.0 - 2.0 * np.log(standard_deviation)
        kl = 0.5 * np.sum(terms)
    if not np.isfinite(kl):
        raise ValueError("Gaussian KL is not finite for the supplied parameters")
    return float(kl)


def contrastive_information_lower_bound(positive_scores, candidate_scores):
    """Compute Section 15's empirical contrastive term via stable log-sum-exp.

    ``candidate_scores`` contains one score row per positive example and one
    column per candidate, including the positive candidate when using the
    usual InfoNCE construction. This calculation alone does not establish a
    true mutual-information estimate or the assumptions needed for a lower
    bound interpretation.
    """
    positive_scores = np.asarray(positive_scores, dtype=np.float64)
    candidate_scores = np.asarray(candidate_scores, dtype=np.float64)
    if positive_scores.ndim != 1 or positive_scores.size == 0:
        raise ValueError("positive_scores must be a non-empty vector")
    if candidate_scores.ndim != 2:
        raise ValueError("candidate_scores must be a matrix")
    if candidate_scores.shape[0] != positive_scores.size or candidate_scores.shape[1] == 0:
        raise ValueError("candidate_scores must have one non-empty row per positive score")
    if not np.all(np.isfinite(positive_scores)) or not np.all(np.isfinite(candidate_scores)):
        raise ValueError("scores must be finite")

    row_max = np.max(candidate_scores, axis=1)
    with np.errstate(over="ignore", invalid="ignore"):
        log_partition = row_max + np.log(
            np.sum(np.exp(candidate_scores - row_max[:, None]), axis=1)
        )
        bound = np.mean(positive_scores - log_partition)
    if not np.isfinite(bound):
        raise ValueError("contrastive information bound is not finite for the supplied scores")
    return float(bound)


def variational_information_bottleneck_loss(
    mean,
    standard_deviation,
    positive_scores,
    candidate_scores,
    beta_info,
):
    """Return KL complexity minus ``beta_info`` times the contrastive bound."""
    if not np.isscalar(beta_info) or not np.isfinite(beta_info) or beta_info < 0.0:
        raise ValueError("beta_info must be finite and non-negative")
    complexity = diagonal_gaussian_kl_to_standard_normal(mean, standard_deviation)
    information_bound = contrastive_information_lower_bound(
        positive_scores, candidate_scores
    )
    loss = complexity - beta_info * information_bound
    if not np.isfinite(loss):
        raise ValueError("VIB loss is not finite for the supplied inputs")
    return float(loss)


__all__ = [
    "contrastive_information_lower_bound",
    "diagonal_gaussian_kl_to_standard_normal",
    "variational_information_bottleneck_loss",
]
