import numpy as np
import pytest

from clean_core.prototypes.sections.section_31 import (
    conformal_christoffel_correction,
    conformal_christoffel_symbols,
)


def test_conformal_christoffel_correction_matches_equation():
    correction = conformal_christoffel_correction(
        gradient_log_omega=np.array([2.0, 3.0]),
        lorentz_metric=np.array([-1.0, 1.0]),
    )

    assert correction[0, 0, 0] == 2.0
    assert correction[0, 0, 1] == 3.0
    assert correction[1, 0, 0] == 3.0


def test_conformal_christoffel_correction_is_symmetric_in_lower_indices():
    correction = conformal_christoffel_correction(
        gradient_log_omega=np.array([0.5, -2.0, 1.25]),
        lorentz_metric=np.array([-1.0, 1.0, 1.0]),
    )

    np.testing.assert_allclose(correction, correction.swapaxes(1, 2))


def test_zero_log_conformal_gradient_gives_zero_correction():
    correction = conformal_christoffel_correction(
        np.zeros(3),
        lorentz_metric=np.array([-1.0, 1.0, 1.0]),
    )

    np.testing.assert_array_equal(correction, np.zeros((3, 3, 3)))


def test_zero_log_conformal_gradient_preserves_supplied_base_connection():
    base = np.arange(27, dtype=float).reshape(3, 3, 3)

    symbols = conformal_christoffel_symbols(
        base,
        np.zeros(3),
        lorentz_metric=np.array([-1.0, 1.0, 1.0]),
    )

    np.testing.assert_array_equal(symbols, base)
    assert symbols is not base


@pytest.mark.parametrize(
    ("gradient", "metric", "message"),
    [
        ([1.0, 2.0], [-1.0], "lorentz_metric"),
        ([[1.0, 2.0]], [-1.0, 1.0], "gradient_log_omega"),
        ([1.0, np.nan], [-1.0, 1.0], "gradient_log_omega"),
        ([1.0, 2.0], [-1.0, np.inf], "lorentz_metric"),
        ([1.0, 2.0], [-1.0, 0.0], "lorentz_metric"),
    ],
)
def test_correction_rejects_invalid_dimensions_and_values(gradient, metric, message):
    with pytest.raises(ValueError, match=message):
        conformal_christoffel_correction(gradient, metric)


def test_connection_addition_rejects_invalid_dimensions_and_non_finite_values():
    with pytest.raises(ValueError, match="base_connection"):
        conformal_christoffel_symbols(np.zeros((2, 3, 2)), np.zeros(2), [-1.0, 1.0])
    with pytest.raises(ValueError, match="dimensions"):
        conformal_christoffel_symbols(np.zeros((3, 3, 3)), np.zeros(2), [-1.0, 1.0])
    with pytest.raises(ValueError, match="base_connection"):
        conformal_christoffel_symbols(
            np.full((2, 2, 2), np.inf), np.zeros(2), [-1.0, 1.0]
        )


def test_rejects_non_finite_results_from_overflow():
    with pytest.raises(ValueError, match="too large"):
        conformal_christoffel_correction(
            [np.finfo(float).max, np.finfo(float).max],
            [-1.0, 1.0],
        )

    base = np.full((2, 2, 2), np.finfo(float).max)
    with pytest.raises(ValueError, match="too large"):
        conformal_christoffel_symbols(base, [0.0, 1e293], [-1.0, 1.0])
