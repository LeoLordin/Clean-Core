import numpy as np
import pytest

from clean_core.interfaces.motor_efe import MotorEFEController, categorical_target_kl


def test_categorical_target_kl_matches_manual_value_and_is_scale_invariant():
    posterior = np.array([0.75, 0.25])
    target = np.array([0.5, 0.5])
    expected = 0.75 * np.log(1.5) + 0.25 * np.log(0.5)

    assert categorical_target_kl(posterior, target) == pytest.approx(expected)
    assert categorical_target_kl(3.0 * posterior, 7.0 * target) == pytest.approx(expected)
    assert MotorEFEController().target_divergence(posterior, target) == pytest.approx(expected)


def test_categorical_target_kl_handles_zero_mass_and_support_mismatch():
    assert categorical_target_kl([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)
    assert categorical_target_kl([1.0, 0.0], [0.0, 1.0]) == float("inf")