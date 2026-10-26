"""Finite-quadrature radial prototype for Section 26 holographic memory.

This implements sampled Poincare plane waves, the stated Plancherel density,
and a bounded discrete episode sum. Each episode is represented by a point
mass and both integrals are finite quadrature sums. This is not the continuous
Harish-Chandra transform or its full spherical inversion.
"""

from collections import deque
from math import gamma, pi

import numpy as np
from scipy.special import loggamma

from clean_core.config import D_LORENTZ, FP64, POLE
from clean_core.geometry.lorentz import log_map, minkowski_dot, minkowski_norm_sq


def _finite_scalar(value, name):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _validate_dimension(dimension):
    if isinstance(dimension, (bool, np.bool_)) or int(dimension) != dimension:
        raise ValueError("dimension must be an integer of at least 3")
    dimension = int(dimension)
    if dimension < 3:
        raise ValueError("dimension must be an integer of at least 3")
    return dimension


def _validate_hyperboloid_point(point, name="mu_point"):
    point = np.asarray(point, dtype=FP64)
    if point.shape != POLE.shape or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be a finite vector with shape {POLE.shape}")
    if point[0] <= 0.0 or not np.isclose(
        minkowski_norm_sq(point), -1.0, rtol=1e-9, atol=1e-9
    ):
        raise ValueError(f"{name} must lie on the future unit hyperboloid")
    return point.copy()


def _validate_frequencies(lambdas):
    lambdas = np.asarray(lambdas, dtype=FP64)
    if lambdas.ndim != 1 or lambdas.size == 0:
        raise ValueError("lambdas must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(lambdas)) or np.any(lambdas < 0.0):
        raise ValueError("lambdas must contain finite non-negative values")
    return lambdas


def _unit_boundary_directions(boundary_directions):
    directions = np.asarray(boundary_directions, dtype=FP64)
    if (
        directions.ndim != 2
        or directions.shape[0] == 0
        or directions.shape[1] != D_LORENTZ - 1
        or not np.all(np.isfinite(directions))
    ):
        raise ValueError(
            f"boundary_directions must be a non-empty finite array with "
            f"{D_LORENTZ - 1} columns"
        )
    norms = np.linalg.norm(directions, axis=1)
    if not np.all(np.isfinite(norms)) or not np.allclose(norms, 1.0, atol=1e-9):
        raise ValueError("each boundary direction must have unit Euclidean norm")
    return directions.copy()


def poincare_plane_waves(mu_point, lambdas, boundary_directions):
    """Evaluate ``phi_(lambda,b)(mu)`` on finite frequency and boundary grids.

    With the repository's ``(-,+,...,+)`` Lorentz convention, the positive
    horospherical coordinate is ``-<mu, (1,b)>_L``. Output shape is
    ``(len(lambdas), number_of_boundary_directions)``.
    """
    point = _validate_hyperboloid_point(mu_point)
    frequencies = _validate_frequencies(lambdas)
    directions = _unit_boundary_directions(boundary_directions)
    null_vectors = np.column_stack((np.ones(len(directions)), directions))
    with np.errstate(over="ignore", invalid="ignore"):
        metric_point = point * np.array([-1.0] + [1.0] * (D_LORENTZ - 1))
        horospherical = -np.einsum("i,bi->b", metric_point, null_vectors)
    if not np.all(np.isfinite(horospherical)) or np.any(horospherical <= 0.0):
        raise ValueError("Poincare plane-wave coordinates must be finite and positive")

    exponent = -(D_LORENTZ - 2.0) / 2.0 + 1j * frequencies[:, None]
    with np.errstate(over="ignore", invalid="ignore"):
        waves = np.exp(exponent * np.log(horospherical)[None, :])
    if not np.all(np.isfinite(waves)):
        raise ValueError("Poincare plane waves are too large to represent")
    return waves


def plancherel_density(lambdas, dimension=D_LORENTZ):
    """Evaluate the Section 26 ``|c(lambda)|^-2`` density stably."""
    frequencies = _validate_frequencies(lambdas)
    dimension = _validate_dimension(dimension)
    rho = (dimension - 1.0) / 2.0
    density = np.zeros(frequencies.shape, dtype=FP64)
    positive = frequencies > 0.0
    if np.any(positive):
        selected = frequencies[positive]
        log_c_magnitude = (
            np.log(gamma(rho))
            + np.real(loggamma(1j * selected))
            - np.real(loggamma(rho + 1j * selected))
        )
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            density[positive] = np.exp(-np.log(2.0 * pi) - 2.0 * log_c_magnitude)
    if not np.all(np.isfinite(density)):
        raise ValueError("Plancherel density is too large to represent")
    return density


def _trapezoid_weights(nodes):
    if nodes.size < 2:
        return np.ones(1, dtype=FP64)
    if np.any(np.diff(nodes) <= 0.0):
        raise ValueError("frequency_nodes must be strictly increasing")
    weights = np.empty_like(nodes)
    weights[0] = 0.5 * (nodes[1] - nodes[0])
    weights[-1] = 0.5 * (nodes[-1] - nodes[-2])
    if nodes.size > 2:
        weights[1:-1] = 0.5 * (nodes[2:] - nodes[:-2])
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("frequency quadrature weights must be finite and positive")
    return weights


class Section26HolographicMemory:
    """Bounded discrete radial/spectral memory using finite quadrature grids.

    Each stored point is a point-mass proxy for an episode. Its default vector
    amplitude is ``Log_POLE(mu)``; callers may provide a different vector
    amplitude. Retrieval sums the episode coefficient
    ``amplitude * phi_(-lambda,b)(mu_episode)``, the stated propagation phase,
    ``phi_(lambda,b)(mu_query)``, and the Plancherel density. Angular weights
    are normalized and then scaled by the area of ``S^(d-2)``.

    This is a bounded numerical prototype, not a full spherical transform.
    """

    def __init__(
        self,
        capacity=256,
        frequency_nodes=None,
        frequency_weights=None,
        boundary_directions=None,
        angular_weights=None,
    ):
        if (
            isinstance(capacity, (bool, np.bool_))
            or int(capacity) != capacity
            or capacity < 1
        ):
            raise ValueError("capacity must be a positive integer")
        self.capacity = int(capacity)

        if frequency_nodes is None:
            frequencies = np.linspace(0.1, 5.0, 32, dtype=FP64)
        else:
            frequencies = _validate_frequencies(frequency_nodes)
        if frequencies.size > 1 and np.any(np.diff(frequencies) <= 0.0):
            raise ValueError("frequency_nodes must be strictly increasing")
        if frequencies[0] <= 0.0:
            raise ValueError("frequency_nodes must be positive")
        self.frequency_nodes = frequencies.copy()

        if frequency_weights is None:
            weights = _trapezoid_weights(frequencies)
        else:
            weights = np.asarray(frequency_weights, dtype=FP64)
            if (
                weights.shape != frequencies.shape
                or not np.all(np.isfinite(weights))
                or np.any(weights <= 0.0)
            ):
                raise ValueError(
                    "frequency_weights must be finite, positive, and match frequency_nodes"
                )
        self.frequency_weights = weights.copy()

        if boundary_directions is None:
            identity = np.eye(D_LORENTZ - 1, dtype=FP64)
            directions = np.concatenate((identity, -identity), axis=0)
        else:
            directions = _unit_boundary_directions(boundary_directions)
        self.boundary_directions = directions

        if angular_weights is None:
            angular = np.full(len(directions), 1.0 / len(directions), dtype=FP64)
        else:
            angular = np.asarray(angular_weights, dtype=FP64)
            if (
                angular.shape != (len(directions),)
                or not np.all(np.isfinite(angular))
                or np.any(angular <= 0.0)
            ):
                raise ValueError(
                    "angular_weights must be finite, positive, and match boundary_directions"
                )
            angular = angular / np.max(angular)
            angular = angular / np.sum(angular)
        self.angular_weights = angular.copy()
        rho = (D_LORENTZ - 1.0) / 2.0
        self._sphere_area = 2.0 * pi**rho / gamma(rho)
        self._density = plancherel_density(self.frequency_nodes)
        self._episodes = deque()
        self._value_shape = None

    def __len__(self):
        return len(self._episodes)

    def store(self, mu_point, value=None):
        """Store one finite episode, evicting its oldest predecessor if full."""
        point = _validate_hyperboloid_point(mu_point)
        if value is None:
            amplitude = log_map(POLE, point)[1:].astype(FP64, copy=True)
        else:
            amplitude = np.asarray(value, dtype=FP64)
            if amplitude.ndim != 1 or amplitude.size == 0 or not np.all(np.isfinite(amplitude)):
                raise ValueError("value must be a non-empty finite vector")
            amplitude = amplitude.copy()
        if self._value_shape is not None and amplitude.shape != self._value_shape:
            raise ValueError(f"all episode values must have shape {self._value_shape}")
        if not np.all(np.isfinite(amplitude)):
            raise ValueError("episode amplitude must contain only finite values")

        if self._value_shape is None:
            self._value_shape = amplitude.shape
        if len(self._episodes) == self.capacity:
            self._episodes.popleft()
        self._episodes.append((point, amplitude))

    def recall(self, query_mu, hbar_cog=1.0):
        """Return the finite-grid interference sum at a hyperbolic query point."""
        query = _validate_hyperboloid_point(query_mu, "query_mu")
        hbar = _finite_scalar(hbar_cog, "hbar_cog")
        if hbar <= 0.0:
            raise ValueError("hbar_cog must be positive")
        if not self._episodes:
            return np.zeros(0, dtype=np.complex128)

        query_waves = poincare_plane_waves(
            query, self.frequency_nodes, self.boundary_directions
        )
        weighted_measure = self.frequency_weights * self._density
        result = np.zeros(self._episodes[0][1].shape, dtype=np.complex128)
        for point, amplitude in self._episodes:
            episode_waves = poincare_plane_waves(
                point, self.frequency_nodes, self.boundary_directions
            )
            distance_argument = -minkowski_dot(query, point)
            if not np.isfinite(distance_argument):
                raise ValueError("hyperbolic episode distance is too large to represent")
            distance = float(np.arccosh(max(1.0, distance_argument)))
            phase = np.exp(-1j * self.frequency_nodes * distance / hbar)
            angular_integral = np.sum(
                episode_waves.conj() * query_waves * self.angular_weights[None, :],
                axis=1,
            )
            spectral_integral = self._sphere_area * np.sum(
                weighted_measure * phase * angular_integral
            )
            result += amplitude * spectral_integral
        if not np.all(np.isfinite(result)):
            raise ValueError("recall result is too large to represent")
        return result


__all__ = [
    "Section26HolographicMemory",
    "poincare_plane_waves",
    "plancherel_density",
]
