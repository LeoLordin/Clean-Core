import numpy as np
import pytest

from clean_core.prototypes.sections.section_25 import Section25SOCRegulator


def _observations(norms):
    errors = [np.array([norm]) for norm in norms]
    precisions = [np.eye(1) for _ in norms]
    return errors, precisions


def test_regulator_updates_tail_log_mean_and_alpha_from_equations():
    regulator = Section25SOCRegulator(
        epsilon_min=1.0,
        gamma_soc=0.5,
        n_min=2,
        eta_alpha=1.0,
        alpha_damp=0.5,
        delta_alpha_max=0.1,
    )
    errors, precisions = _observations([np.e**2, np.e**4])

    updated = regulator.update(errors, precisions, dt=0.1)

    expected_log_mean = 0.5 * ((2.0 + 4.0) / 2.0)
    expected_alpha = 1.0 + 1.0 / expected_log_mean
    expected_delta = np.clip(0.1 * 1.2 * (1.5 - expected_alpha), -0.1, 0.1)
    assert regulator.tail_count == 2
    assert regulator.log_tail_mean == pytest.approx(expected_log_mean)
    assert regulator.alpha_measured == pytest.approx(expected_alpha)
    assert regulator.delta_alpha == pytest.approx(expected_delta)
    assert updated == pytest.approx(0.5 + expected_delta)


def test_regulator_holds_statistic_and_damping_with_too_few_tail_samples():
    regulator = Section25SOCRegulator(
        epsilon_min=1.0, gamma_soc=1.0, n_min=2, alpha_damp=0.2
    )
    errors, precisions = _observations([np.e**2, 0.5])
    regulator.log_tail_mean = 0.75

    updated = regulator.update(errors, precisions, dt=0.1)

    assert regulator.tail_count == 1
    assert regulator.log_tail_mean == 0.75
    assert regulator.alpha_measured == pytest.approx(1.0 + 1.0 / 0.75)
    assert regulator.delta_alpha == 0.0
    assert updated == 0.2


@pytest.mark.parametrize(
    ("norm", "initial_alpha", "expected_delta"),
    [
        (np.e, 0.5, -0.01),
        (np.e**4, 0.5, 0.01),
        (np.e, 0.05, -0.01),
    ],
)
def test_regulator_clips_update_and_respects_alpha_floor(
    norm, initial_alpha, expected_delta
):
    regulator = Section25SOCRegulator(
        epsilon_min=1.0,
        gamma_soc=1.0,
        n_min=1,
        eta_alpha=10.0,
        alpha_damp=initial_alpha,
        delta_alpha_max=0.01,
    )
    errors, precisions = _observations([norm])

    updated = regulator.update(errors, precisions, dt=1.0)

    assert regulator.delta_alpha == pytest.approx(expected_delta)
    assert updated == pytest.approx(max(0.05, initial_alpha + expected_delta))


@pytest.mark.parametrize(
    ("errors", "precisions", "dt", "message"),
    [
        ([np.array([1.0])], [], 0.1, "equal lengths"),
        ([np.array([np.inf])], [np.eye(1)], 0.1, "must be finite"),
        ([np.array([1.0])], [np.eye(2)], 0.1, "match its error"),
        ([np.array([1.0])], [np.eye(1)], np.nan, "dt must be a finite scalar"),
        ([np.array([1.0])], [np.eye(1)], 0.0, "dt must be positive"),
    ],
)
def test_regulator_rejects_invalid_observations(errors, precisions, dt, message):
    regulator = Section25SOCRegulator(n_min=1)

    with pytest.raises(ValueError, match=message):
        regulator.update(errors, precisions, dt=dt)


def test_regulator_requires_finite_precision_weighted_norm():
    regulator = Section25SOCRegulator(n_min=1)
    errors = [np.array([1e200])]
    precisions = [np.eye(1)]

    with pytest.raises(ValueError, match="squared error must be finite"):
        regulator.update(errors, precisions, dt=0.1)
