import numpy as np

from clean_core.cognition.criticality import SOCCriticalityRegulator


def test_criticality_regulator_ignores_tails_below_minimum_count():
    regulator = SOCCriticalityRegulator(n_min=3, alpha_damp=0.2)
    errors = [np.array([2.0]), np.array([2.0])]
    precisions = [np.eye(1), np.eye(1)]

    assert regulator.update(errors, precisions, dt=0.01) == 0.2
    assert regulator.alpha_measured is None


def test_criticality_regulator_updates_from_precision_weighted_tail():
    regulator = SOCCriticalityRegulator(
        epsilon_min=0.5,
        gamma_soc=1.0,
        n_min=2,
        eta_alpha=1.0,
        alpha_damp=0.5,
        delta_alpha_max=0.1,
    )
    errors = [np.array([1.0]), np.array([2.0])]
    precisions = [np.eye(1), np.eye(1)]

    updated = regulator.update(errors, precisions, dt=0.1)

    assert regulator.alpha_measured is not None
    assert updated >= regulator.alpha_min
    assert updated != 0.5