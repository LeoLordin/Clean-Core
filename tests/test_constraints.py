import pytest

from clean_core.core.constraints import projected_phr_multiplier_update


def test_inactive_constraint_keeps_zero_multiplier_nonnegative():
    assert projected_phr_multiplier_update(0.0, 0.5, 1.0) == 0.0


def test_active_constraint_increases_multiplier():
    updated = projected_phr_multiplier_update(0.2, 1.5, 1.0, dt=0.1, eta_lambda=0.5)
    assert updated > 0.2


def test_multiplier_remains_nonnegative_when_constraint_is_strongly_inactive():
    updated = projected_phr_multiplier_update(0.01, -100.0, 1.0, dt=1.0, eta_lambda=10.0)
    assert updated == 0.0


def test_multiplier_update_rejects_nonpositive_rho():
    with pytest.raises(ValueError):
        projected_phr_multiplier_update(0.0, 1.0, 0.0, rho=0.0)