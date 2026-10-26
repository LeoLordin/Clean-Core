import numpy as np
import pytest

from clean_core.prototypes.sections.section_38 import (
    expand_hyperboloid_embedding,
    minkowski_gram_matrix,
)


def test_zero_padding_embedding_preserves_batch_gram_matrix():
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [np.cosh(0.4), np.sinh(0.4), 0.0],
            [np.cosh(0.7), 0.0, np.sinh(0.7)],
        ]
    )

    expanded = expand_hyperboloid_embedding(points, 2)

    assert expanded.shape == (3, 5)
    np.testing.assert_array_equal(expanded[:, 3:], np.zeros((3, 2)))
    np.testing.assert_allclose(
        minkowski_gram_matrix(expanded),
        minkowski_gram_matrix(points),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        -expanded[:, 0] ** 2 + np.sum(expanded[:, 1:] ** 2, axis=1),
        -1.0,
        rtol=0.0,
        atol=1e-12,
    )


def test_embedding_accepts_a_single_point_and_increases_ambient_dimension():
    point = np.array([np.cosh(0.5), np.sinh(0.5)])

    expanded = expand_hyperboloid_embedding(point, np.int64(3))

    assert expanded.shape == (5,)
    np.testing.assert_array_equal(expanded[2:], np.zeros(3))
    np.testing.assert_allclose(-expanded[0] ** 2 + expanded[1:] @ expanded[1:], -1.0)


@pytest.mark.parametrize(
    "points",
    [
        np.empty((0, 2)),
        np.ones((2, 1)),
        np.ones((2, 2, 2)),
        [[1.0, 0.0], [np.nan, 0.0]],
        [[1.0, 0.0], [-1.0, 0.0]],
        [[1.0, 0.5]],
    ],
)
def test_embedding_rejects_invalid_batch_and_ambient_points(points):
    with pytest.raises(ValueError):
        expand_hyperboloid_embedding(points, 1)


@pytest.mark.parametrize("extra_dimensions", [0, -1, 1.5, True, np.bool_(False)])
def test_embedding_requires_positive_integer_ambient_expansion(extra_dimensions):
    with pytest.raises(ValueError, match="positive integer"):
        expand_hyperboloid_embedding([1.0, 0.0], extra_dimensions)


@pytest.mark.parametrize("tolerance", [0.0, -1.0, np.inf, np.nan])
def test_embedding_requires_finite_positive_tolerance(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        expand_hyperboloid_embedding([1.0, 0.0], 1, tolerance=tolerance)


@pytest.mark.parametrize(
    "points",
    [
        np.empty((0, 2)),
        np.ones(2),
        np.ones((2, 1)),
        np.ones((2, 2, 2)),
        [[1.0, np.inf]],
    ],
)
def test_minkowski_gram_matrix_requires_a_nonempty_finite_ambient_batch(points):
    with pytest.raises(ValueError):
        minkowski_gram_matrix(points)
