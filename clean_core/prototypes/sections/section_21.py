"""Bounded usage/error-driven SoftVQ codebook prototype for Section 21.

This tracks numeric code usage and gates code creation on prediction error and
geometric novelty; it does not infer or validate semantic concepts.
"""

import numpy as np

from clean_core.config import D_SPATIAL, K_MAX, POLE, FP64
from clean_core.geometry.lorentz import chordal_distance, exp_map, log_map


def _finite_scalar(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite scalar")
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite scalar") from error
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


class SoftVQCodebookEvolution:
    """Track code usage and deterministically evolve a bounded tangent codebook."""

    def __init__(
        self,
        centroids,
        usage=None,
        *,
        max_size=K_MAX,
        protected_indices=(0,),
        active=None,
        seed=None,
    ):
        self.centroids = np.asarray(centroids, dtype=FP64).copy()
        if self.centroids.ndim != 2 or self.centroids.shape[1] != D_SPATIAL:
            raise ValueError(f"centroids must have shape (n > 0, {D_SPATIAL})")
        if self.centroids.shape[0] == 0:
            raise ValueError("centroids must contain at least one code")
        if (
            isinstance(max_size, (bool, np.bool_))
            or not isinstance(max_size, (int, np.integer))
            or max_size < self.centroids.shape[0]
        ):
            raise ValueError("max_size must be an integer at least the initial codebook size")
        self.max_size = int(max_size)

        if usage is None:
            self.usage = np.ones(self.centroids.shape[0], dtype=FP64)
        else:
            self.usage = np.asarray(usage, dtype=FP64).copy()
        if self.usage.shape != (self.centroids.shape[0],):
            raise ValueError("usage must have one value per centroid")

        if active is None:
            self.active = np.ones(self.centroids.shape[0], dtype=bool)
        else:
            self.active = np.asarray(active, dtype=bool).copy()
        if self.active.shape != (self.centroids.shape[0],):
            raise ValueError("active must have one value per centroid")

        if active is not None:
            raw_active = np.asarray(active)
            if raw_active.dtype.kind not in "biuf" or not np.all(
                np.isin(raw_active, (0, 1))
            ):
                raise ValueError("active must contain only boolean or binary values")

        try:
            raw_protected_indices = tuple(protected_indices)
        except TypeError as error:
            raise ValueError("protected_indices must be integer indices") from error
        if any(
            isinstance(index, (bool, np.bool_))
            or not isinstance(index, (int, np.integer))
            for index in raw_protected_indices
        ):
            raise ValueError("protected_indices must be integer indices")
        self.protected_indices = frozenset(int(index) for index in raw_protected_indices)
        if 0 not in self.protected_indices:
            raise ValueError("code 0 must be protected")
        if any(index < 0 or index >= len(self.centroids) for index in self.protected_indices):
            raise ValueError("protected indices must refer to existing codes")
        if any(not self.active[index] for index in self.protected_indices):
            raise ValueError("protected codes must be active")
        self._rng = np.random.default_rng(seed)
        self._validate_state()

    def _validate_state(self):
        if self.centroids.ndim != 2 or self.centroids.shape[1] != D_SPATIAL:
            raise ValueError(f"centroids must have shape (n, {D_SPATIAL})")
        count = self.centroids.shape[0]
        if count == 0 or count > self.max_size:
            raise ValueError("codebook size must be between one and max_size")
        if self.usage.shape != (count,) or self.active.shape != (count,):
            raise ValueError("codebook state arrays must have one value per centroid")
        if not np.all(np.isfinite(self.centroids)):
            raise ValueError("centroids must be finite")
        if not np.all(np.isfinite(self.usage)) or np.any(self.usage < 0.0):
            raise ValueError("usage must be finite and non-negative")
        if not np.any(self.active):
            raise ValueError("at least one code must be active")
        if any(index >= count or not self.active[index] for index in self.protected_indices):
            raise ValueError("protected codes must exist and remain active")
        for centroid in self.centroids[self.active]:
            with np.errstate(over="ignore", invalid="ignore"):
                point = exp_map(POLE.copy(), np.concatenate(([0.0], centroid)))
            if not np.all(np.isfinite(point)) or point[0] <= 0.0:
                raise ValueError("centroid does not define finite hyperbolic geometry")

    def update_usage(self, assignments, *, gamma_usage=1.0):
        """Apply usage decay and assignment counts, returning the updated usage."""
        self._validate_state()
        gamma_usage = _finite_scalar(gamma_usage, "gamma_usage")
        if not 0.0 <= gamma_usage <= 1.0:
            raise ValueError("gamma_usage must be finite and between zero and one")
        assignments = np.asarray(assignments)
        if assignments.ndim != 1 or (
            assignments.size and not np.issubdtype(assignments.dtype, np.integer)
        ):
            raise ValueError("assignments must be a one-dimensional integer array")
        assignments = assignments.astype(np.intp, copy=False)
        if assignments.size and (
            np.any(assignments < 0) or np.any(assignments >= len(self.centroids))
        ):
            raise ValueError("assignments must refer to existing codes")
        if assignments.size and np.any(~self.active[assignments]):
            raise ValueError("assignments cannot refer to pruned codes")

        counts = np.bincount(assignments.astype(np.intp), minlength=len(self.centroids))
        with np.errstate(over="ignore", invalid="ignore"):
            updated_usage = gamma_usage * self.usage + counts
        if not np.all(np.isfinite(updated_usage)):
            raise ValueError("updated usage must remain finite")
        self.usage = updated_usage
        self._validate_state()
        return self.usage.copy()

    def prune(self, *, theta_prune=0.05):
        """Deactivate low-usage unprotected codes and return their indices."""
        self._validate_state()
        theta_prune = _finite_scalar(theta_prune, "theta_prune")
        if theta_prune < 0.0:
            raise ValueError("theta_prune must be finite and non-negative")
        remove = [
            index
            for index in range(len(self.centroids))
            if self.active[index]
            and index not in self.protected_indices
            and self.usage[index] < theta_prune
        ]
        self.active[remove] = False
        self._validate_state()
        return tuple(remove)

    def maybe_spawn(
        self,
        mu_point,
        prediction_error,
        precision,
        *,
        theta_spawn=15.0,
        d_min=0.1,
        noise_scale=0.0,
    ):
        """Spawn a code if divergence, novelty, and active-capacity gates pass."""
        self._validate_state()
        mu_point = np.asarray(mu_point, dtype=FP64)
        if mu_point.shape != (D_SPATIAL + 1,) or not np.all(np.isfinite(mu_point)):
            raise ValueError(f"mu_point must be a finite vector of length {D_SPATIAL + 1}")
        with np.errstate(over="ignore", invalid="ignore"):
            point_norm_sq = float(-mu_point[0] ** 2 + np.dot(mu_point[1:], mu_point[1:]))
        if mu_point[0] <= 0.0 or not np.isclose(point_norm_sq, -1.0, rtol=1e-8, atol=1e-8):
            raise ValueError("mu_point must lie on the future unit hyperboloid")

        error = np.asarray(prediction_error, dtype=FP64)
        precision = np.asarray(precision, dtype=FP64)
        if error.shape != (D_SPATIAL,) or not np.all(np.isfinite(error)):
            raise ValueError(f"prediction_error must be a finite vector of length {D_SPATIAL}")
        if precision.shape != (D_SPATIAL, D_SPATIAL) or not np.all(np.isfinite(precision)):
            raise ValueError(f"precision must be a finite ({D_SPATIAL}, {D_SPATIAL}) matrix")
        if not np.allclose(precision, precision.T, rtol=1e-10, atol=1e-12):
            raise ValueError("precision must be symmetric")
        if np.min(np.linalg.eigvalsh(precision)) < -1e-10:
            raise ValueError("precision must be positive semidefinite")
        theta_spawn = _finite_scalar(theta_spawn, "theta_spawn")
        d_min = _finite_scalar(d_min, "d_min")
        noise_scale = _finite_scalar(noise_scale, "noise_scale")
        for name, value in (("theta_spawn", theta_spawn), ("d_min", d_min), ("noise_scale", noise_scale)):
            if value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

        divergence = float(error @ precision @ error)
        if divergence <= theta_spawn or np.count_nonzero(self.active) >= self.max_size:
            return False

        active_points = [
            exp_map(POLE.copy(), np.concatenate(([0.0], centroid)))
            for centroid in self.centroids[self.active]
        ]
        nearest_distance = min(chordal_distance(mu_point, center) for center in active_points)
        if nearest_distance <= d_min:
            return False

        tangent = log_map(POLE.copy(), mu_point)[1:]
        if noise_scale:
            tangent = tangent + self._rng.normal(0.0, noise_scale, size=D_SPATIAL)
        if not np.all(np.isfinite(tangent)):
            raise ValueError("spawned centroid must be finite")
        with np.errstate(over="ignore", invalid="ignore"):
            spawned_point = exp_map(POLE.copy(), np.concatenate(([0.0], tangent)))
        if not np.all(np.isfinite(spawned_point)) or spawned_point[0] <= 0.0:
            raise ValueError("spawned centroid does not define finite hyperbolic geometry")

        inactive_indices = np.flatnonzero(~self.active)
        if inactive_indices.size:
            index = int(inactive_indices[0])
            self.centroids[index] = tangent
            self.usage[index] = 1.0
            self.active[index] = True
        else:
            self.centroids = np.vstack((self.centroids, tangent))
            self.usage = np.append(self.usage, 1.0)
            self.active = np.append(self.active, True)
        self._validate_state()
        return True


__all__ = ["SoftVQCodebookEvolution"]
