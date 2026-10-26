import numpy as np
import pytest

from clean_core.prototypes.sections.section_09 import (
    section_09_pragmatic_free_energy,
)


def test_section_09_objective_matches_diagonal_hand_calculation():
    # The covariance is diag(5, 4), so precision is diag(1/5, 1/4).
    # The determinant term is 1/2 log(20), and each probe gives trace -1.55.
    result = section_09_pragmatic_free_energy(
        prediction_error=[1.0, 2.0],
        precision_scalar=2.0,
        d_diag=[2.0, 4.0],
        low_rank_basis=[[1.0], [0.0]],
        low_rank_values=[3.0],
        trace_probes=[[1.0, 1.0], [1.0, -1.0]],
    )

    expected = 1.2 - 0.5 * np.log(2.0) + 0.5 * np.log(20.0) - 0.775
    np.testing.assert_allclose(result, expected)


def test_section_09_rank_one_objective_matches_scalar_calculation():
    # Covariance 2 + 3 = 5, precision 1/5, and the one-dimensional
    # stochastic trace term is exactly (1/5 - 1)/2.
    result = section_09_pragmatic_free_energy(
        prediction_error=[2.0],
        precision_scalar=2.0,
        d_diag=[2.0],
        low_rank_basis=[[1.0]],
        low_rank_values=[3.0],
        trace_probes=[[1.0], [-1.0]],
    )

    np.testing.assert_allclose(result, 0.4 + 0.5 * np.log(5.0))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("prediction_error", [[1.0, 2.0]], "prediction_error must be a vector"),
        ("d_diag", [1.0], "d_diag must have shape"),
        ("low_rank_basis", [[1.0]], "low_rank_basis must have 2 rows"),
        ("low_rank_values", [1.0, 2.0], "must match a non-empty basis rank"),
        ("trace_probes", [[1.0]], "trace_probes must have shape"),
        ("trace_probes", [[0.0, 1.0]], r"only -1 or \+1"),
        ("prediction_error", [1.0, np.nan], "finite"),
        ("d_diag", [1.0, 0.0], "must be positive"),
        ("low_rank_values", [np.inf], "finite"),
    ],
)
def test_section_09_rejects_invalid_array_shapes_and_values(field, value, message):
    arguments = {
        "prediction_error": [1.0, 2.0],
        "precision_scalar": 1.0,
        "d_diag": [1.0, 1.0],
        "low_rank_basis": [[1.0], [0.0]],
        "low_rank_values": [1.0],
        "trace_probes": [[1.0, -1.0]],
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=message):
        section_09_pragmatic_free_energy(**arguments)


@pytest.mark.parametrize("precision_scalar", [0.0, -1.0, np.inf, np.nan, [1.0]])
def test_section_09_requires_finite_positive_scalar_precision(precision_scalar):
    with pytest.raises(ValueError, match="precision_scalar"):
        section_09_pragmatic_free_energy(
            prediction_error=[1.0],
            precision_scalar=precision_scalar,
            d_diag=[1.0],
            low_rank_basis=[[1.0]],
            low_rank_values=[1.0],
            trace_probes=[[1.0]],
        )
