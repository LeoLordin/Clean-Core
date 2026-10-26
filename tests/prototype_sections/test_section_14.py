import numpy as np
import pytest

from clean_core.config import D_LORENTZ, POLE
from clean_core.prototypes.sections.section_14 import (
    geodesic_distance,
    geodesic_exp,
    geodesic_log,
    minkowski_inner_product,
    minkowski_metric,
    minkowski_norm,
    parallel_transport,
)


def test_minkowski_metric_and_inner_product_use_lorentz_signature():
    metric = minkowski_metric()
    x = np.linspace(-0.4, 0.6, D_LORENTZ)
    y = np.linspace(0.3, -0.2, D_LORENTZ)

    np.testing.assert_array_equal(
        np.diag(metric), np.array([-1.0] + [1.0] * (D_LORENTZ - 1))
    )
    np.testing.assert_allclose(
        minkowski_inner_product(x, y), x @ metric @ y, atol=1e-14
    )
    np.testing.assert_allclose(
        minkowski_norm(x) ** 2,
        max(0.0, minkowski_inner_product(x, x)),
        atol=1e-14,
    )


def test_geodesic_exp_log_and_distance_match_hyperbolic_geometry():
    tangent = np.zeros(D_LORENTZ)
    tangent[1] = 0.8
    point = geodesic_exp(POLE, tangent)

    np.testing.assert_allclose(minkowski_inner_product(point, point), -1.0, atol=1e-12)
    logarithm = geodesic_log(POLE, point)
    np.testing.assert_allclose(logarithm, tangent, atol=1e-12)
    np.testing.assert_allclose(minkowski_inner_product(POLE, logarithm), 0.0, atol=1e-12)
    np.testing.assert_allclose(geodesic_distance(POLE, point), 0.8, atol=1e-12)


def test_parallel_transport_preserves_tangency_and_minkowski_norm():
    displacement = np.zeros(D_LORENTZ)
    displacement[1] = 0.5
    target = geodesic_exp(POLE, displacement)
    tangent = np.zeros(D_LORENTZ)
    tangent[2] = 1.0
    transported = parallel_transport(POLE, target, tangent)

    np.testing.assert_allclose(
        minkowski_inner_product(target, transported), 0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        minkowski_inner_product(transported, transported),
        minkowski_inner_product(tangent, tangent),
        atol=1e-12,
    )


def test_parallel_transport_preserves_tangency_for_short_geodesics():
    displacement = np.zeros(D_LORENTZ)
    displacement[1] = 5e-7
    target = geodesic_exp(POLE, displacement)
    tangent = np.zeros(D_LORENTZ)
    tangent[1] = 1.0

    transported = parallel_transport(POLE, target, tangent)

    np.testing.assert_allclose(
        minkowski_inner_product(target, transported), 0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        minkowski_inner_product(transported, transported),
        minkowski_inner_product(tangent, tangent),
        atol=1e-12,
    )


def test_maps_reject_nonmanifold_points_and_nontangent_vectors():
    with pytest.raises(ValueError, match="future unit hyperboloid"):
        geodesic_log(POLE, np.zeros(D_LORENTZ))

    invalid_tangent = POLE.copy()
    with pytest.raises(ValueError, match="tangent"):
        geodesic_exp(POLE, invalid_tangent)
