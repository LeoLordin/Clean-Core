import numpy as np
import pytest

from clean_core.prototypes.sections.section_36 import (
    Section36CurvatureAdapter,
    section_36_distance,
)


def test_signed_curvature_chart_distances_and_flat_limit():
    x = np.array([0.1, 0.0])
    y = np.array([0.0, 0.0])
    flat_distance = section_36_distance(0.0, x, y)

    assert flat_distance == pytest.approx(0.2)
    assert section_36_distance(1.0, x, y) > 0.0
    assert section_36_distance(-1.0, x, y) > 0.0
    for curvature in (-1e-7, 1e-7):
        assert section_36_distance(curvature, x, y) == pytest.approx(
            flat_distance, rel=1e-6
        )


def test_negative_curvature_rejects_coordinates_outside_chart_domain():
    with pytest.raises(ValueError, match="Poincare ball"):
        section_36_distance(-1.0, [1.0, 0.0], [0.0, 0.0])


def test_curvature_update_is_clipped_to_bounds():
    adapter = Section36CurvatureAdapter(curvature=0.5, learning_rate=0.1)

    assert adapter.update(2.0, bounds=(-1.0, 1.0)) == pytest.approx(0.3)
    assert adapter.update(-100.0, bounds=(-1.0, 1.0)) == 1.0
