import numpy as np
import pytest

from clean_core.config import ETA_L, FP32
from clean_core.prototypes.algorithms.algorithm_07 import apply_algorithm_07
from clean_core.prototypes.sections.section_12 import Section12Prototype


def _prototype():
    return Section12Prototype(input_dim=3, hidden_dim=2, rank=2, seed=23)


def _values():
    return (
        np.arange(6, dtype=FP32).reshape(2, 3) / 10.0,
        np.arange(2 * (ETA_L.size - 1), dtype=FP32).reshape(ETA_L.size - 1, 2)
        / 10.0,
        np.full((2, 3), 1e4, dtype=FP32),
        np.full((ETA_L.size - 1, 2), -1e4, dtype=FP32),
    )


def test_algorithm_07_applies_finite_natural_gradient_updates_with_a_bound():
    prototype = _prototype()
    weights_1, weights_2, grad_w1, grad_w2 = _values()
    original_weights_1 = weights_1.copy()
    original_weights_2 = weights_2.copy()
    factors = tuple(
        factor.copy()
        for estimator in (prototype.kfac_layer1, prototype.kfac_layer2)
        for factor in (estimator.A, estimator.B)
    )
    dlr_state = (
        prototype.dlr.V_basis.copy(),
        prototype.dlr.s_values.copy(),
        prototype.dlr.d_diag.copy(),
    )

    result = apply_algorithm_07(
        prototype,
        weights_1,
        weights_2,
        grad_w1,
        grad_w2,
        inference_tick=4,
        dt=0.2,
        epsilon_1=0.5,
    )

    assert result["updated"]
    for index, (key, original) in enumerate(
        (("weights_1", original_weights_1), ("weights_2", original_weights_2))
    ):
        natural_gradient = result[f"natural_gradient_{index + 1}"]
        updated = result[key]
        assert np.isfinite(natural_gradient).all()
        assert np.linalg.norm(natural_gradient) <= 1.0 + 1e-6
        assert np.isfinite(updated).all()
        assert np.linalg.norm(updated - original) <= 0.4 + 1e-6

    np.testing.assert_array_equal(weights_1, original_weights_1)
    np.testing.assert_array_equal(weights_2, original_weights_2)
    for old, estimator, factor_name in zip(
        factors,
        (
            prototype.kfac_layer1,
            prototype.kfac_layer1,
            prototype.kfac_layer2,
            prototype.kfac_layer2,
        ),
        ("A", "B", "A", "B"),
    ):
        np.testing.assert_array_equal(old, getattr(estimator, factor_name))
    for old, current in zip(
        dlr_state,
        (prototype.dlr.V_basis, prototype.dlr.s_values, prototype.dlr.d_diag),
    ):
        np.testing.assert_array_equal(old, current)


@pytest.mark.parametrize("tick", [1, 2, 3, 5, 6, 7])
def test_algorithm_07_skips_non_update_ticks_without_changing_state(tick):
    prototype = _prototype()
    weights_1, weights_2, grad_w1, grad_w2 = _values()
    initial_factors = [
        factor.copy()
        for estimator in (prototype.kfac_layer1, prototype.kfac_layer2)
        for factor in (estimator.A, estimator.B)
    ]
    initial_dlr_state = (
        prototype.dlr.V_basis.copy(),
        prototype.dlr.s_values.copy(),
        prototype.dlr.d_diag.copy(),
    )

    result = apply_algorithm_07(
        prototype,
        weights_1,
        weights_2,
        grad_w1,
        grad_w2,
        inference_tick=tick,
        dt=0.1,
    )

    assert not result["updated"]
    assert result["natural_gradient_1"] is None
    assert result["natural_gradient_2"] is None
    np.testing.assert_array_equal(result["weights_1"], weights_1)
    np.testing.assert_array_equal(result["weights_2"], weights_2)
    current_factors = [
        factor
        for estimator in (prototype.kfac_layer1, prototype.kfac_layer2)
        for factor in (estimator.A, estimator.B)
    ]
    for initial, current in zip(initial_factors, current_factors):
        np.testing.assert_array_equal(initial, current)
    for initial, current in zip(
        initial_dlr_state,
        (prototype.dlr.V_basis, prototype.dlr.s_values, prototype.dlr.d_diag),
    ):
        np.testing.assert_array_equal(initial, current)


@pytest.mark.parametrize(
    ("argument", "replacement", "message"),
    [
        ("weights_1", np.ones((3, 2)), "weights_1 must have shape"),
        ("grad_w2", np.ones((2, 2)), "grad_w2 must have shape"),
        ("grad_w1", np.full((2, 3), np.nan), "grad_w1 must contain only finite"),
        ("dt", 0.0, "dt must be positive"),
        ("epsilon_1", np.inf, "epsilon_1 must be positive"),
        ("inference_tick", 4.5, "inference_tick must be a positive integer"),
    ],
)
def test_algorithm_07_rejects_invalid_inputs(argument, replacement, message):
    prototype = _prototype()
    values = dict(zip(("weights_1", "weights_2", "grad_w1", "grad_w2"), _values()))
    options = {"inference_tick": 4, "dt": 0.1}
    if argument in options:
        options[argument] = replacement
    else:
        values[argument] = replacement

    with pytest.raises(ValueError, match=message):
        apply_algorithm_07(prototype, **values, **options)
