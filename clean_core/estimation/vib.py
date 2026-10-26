import numpy as np

from clean_core.config import FP64


def diagonal_gaussian_kl_to_standard_normal(mean, standard_deviation):
    mean = np.asarray(mean, dtype=FP64)
    standard_deviation = np.asarray(standard_deviation, dtype=FP64)
    if mean.ndim != 1 or standard_deviation.shape != mean.shape or mean.size == 0:
        raise ValueError("mean and standard_deviation must be non-empty equal-sized vectors")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(standard_deviation)):
        raise ValueError("Gaussian parameters must be finite")
    if np.any(standard_deviation <= 0.0):
        raise ValueError("standard_deviation must be positive")
    variance = standard_deviation**2
    return float(0.5 * np.sum(mean**2 + variance - 1.0 - np.log(variance)))


def contrastive_information_lower_bound(positive_scores, candidate_scores):
    positive_scores = np.asarray(positive_scores, dtype=FP64)
    candidate_scores = np.asarray(candidate_scores, dtype=FP64)
    if positive_scores.ndim != 1 or candidate_scores.ndim != 2:
        raise ValueError("positive_scores must be a vector and candidate_scores a matrix")
    if candidate_scores.shape[0] != positive_scores.size or candidate_scores.shape[1] == 0:
        raise ValueError("candidate_scores must have one non-empty row per positive score")
    if not np.all(np.isfinite(positive_scores)) or not np.all(np.isfinite(candidate_scores)):
        raise ValueError("scores must be finite")

    row_max = np.max(candidate_scores, axis=1)
    log_partition = row_max + np.log(np.sum(np.exp(candidate_scores - row_max[:, None]), axis=1))
    return float(np.mean(positive_scores - log_partition))


def variational_information_bottleneck_loss(
    mean,
    standard_deviation,
    positive_scores,
    candidate_scores,
    beta_info,
):
    if not np.isfinite(beta_info) or beta_info < 0.0:
        raise ValueError("beta_info must be finite and non-negative")
    complexity = diagonal_gaussian_kl_to_standard_normal(mean, standard_deviation)
    information_bound = contrastive_information_lower_bound(positive_scores, candidate_scores)
    return complexity - beta_info * information_bound