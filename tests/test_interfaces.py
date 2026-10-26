import numpy as np
import pytest

from clean_core.core.inference import precision_weighted_spike_mask
from clean_core.interfaces.neuromorphic_crossbar import NeuromorphicCrossbarEncoder


def test_precision_weighted_spike_mask_matches_sigmoid_energy_equation():
    error = np.array([0.5, -0.25])
    precision = np.diag([2.0, 4.0])
    energy = error @ precision @ error
    expected = 1.0 / (1.0 + np.exp(-3.0 * (energy - 0.2)))

    actual = precision_weighted_spike_mask(error, precision, threshold=0.2, beta=3.0)

    assert actual == pytest.approx(expected)


def test_crossbar_mask_suppresses_spikes_below_error_threshold():
    encoder = NeuromorphicCrossbarEncoder(m_size=4)
    encoder.G_conductance.fill(1.0)
    tangent = np.full(4, 4.0, dtype=np.float32)
    error = np.zeros(2)
    precision = np.eye(2)

    spikes, current = encoder.encode_spikes_ste(
        tangent,
        prediction_error=error,
        precision=precision,
        threshold=1.0,
    )

    np.testing.assert_array_equal(spikes, np.zeros(4, dtype=np.uint8))
    np.testing.assert_array_equal(current, np.zeros(4, dtype=np.float32))