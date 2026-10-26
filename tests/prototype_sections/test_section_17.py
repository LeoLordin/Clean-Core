import numpy as np
import pytest

from clean_core.prototypes.sections.section_17 import target_divergence


def test_target_divergence_matches_categorical_kl_and_is_scale_invariant():
    predicted = np.array([0.75, 0.25])
    target = np.array([0.5, 0.5])
    expected = 0.75 * np.log(1.5) + 0.25 * np.log(0.5)

    assert target_divergence(predicted, target) == pytest.approx(expected)
    assert target_divergence(3.0 * predicted, 7.0 * target) == pytest.approx(expected)


def test_target_divergence_handles_zeros_and_support_mismatch():
    assert target_divergence([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)
    assert target_divergence([0.0, 1.0], [0.25, 0.75]) == pytest.approx(
        np.log(1.0 / 0.75)
    )
    assert target_divergence([1.0, 0.0], [0.5, 0.5]) == pytest.approx(np.log(2.0))
    assert target_divergence([1.0, 0.0], [0.0, 1.0]) == float("inf")


def test_target_divergence_handles_wide_dynamic_range_without_false_zero_support():
    result = target_divergence([0.0, 1.0], [1e308, 1e-308])

    assert result == pytest.approx(np.log(1e308) - np.log(1e-308), rel=1e-12)


@pytest.mark.parametrize(
    ("predicted", "target"),
    [
        ([], []),
        ([1.0], [1.0, 0.0]),
        ([[0.5, 0.5]], [0.5, 0.5]),
        ([float("nan"), 1.0], [0.5, 0.5]),
        ([0.5, 0.5], [0.0, float("inf")]),
        ([-0.1, 1.1], [0.5, 0.5]),
        ([0.5, 0.5], [0.5, -0.1]),
        ([0.0, 0.0], [0.5, 0.5]),
        ([0.5, 0.5], [0.0, 0.0]),
        ([0.5 + 1.0j, 0.5], [0.5, 0.5]),
        (["not a probability", 0.5], [0.5, 0.5]),
        ([0.5, 0.5], ["not a probability", 0.5]),
    ],
)
def test_target_divergence_rejects_invalid_distributions(predicted, target):
    with pytest.raises(ValueError):
        target_divergence(predicted, target)
