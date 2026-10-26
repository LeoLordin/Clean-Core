import numpy as np

from clean_core.config import FP64


def constant_curvature_distance(curvature, x, y):
    x = np.asarray(x, dtype=FP64)
    y = np.asarray(y, dtype=FP64)
    if x.ndim != 1 or y.shape != x.shape or x.size == 0:
        raise ValueError("x and y must be non-empty vectors of equal shape")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("coordinates must be finite")
    if not np.isfinite(curvature):
        raise ValueError("curvature must be finite")
    squared_distance = float(np.dot(x - y, x - y))
    if abs(curvature) < 1e-12:
        return float(2.0 * np.sqrt(squared_distance))

    denominator = (1.0 + curvature * np.dot(x, x)) * (1.0 + curvature * np.dot(y, y))
    if curvature < 0.0 and (1.0 + curvature * np.dot(x, x) <= 0.0 or 1.0 + curvature * np.dot(y, y) <= 0.0):
        raise ValueError("negative-curvature coordinates must lie inside the Poincare ball")
    if denominator <= 0.0:
        raise ValueError("coordinates are outside the valid stereographic chart")

    magnitude = abs(curvature)
    if curvature < 0.0:
        argument = 1.0 + 2.0 * magnitude * squared_distance / denominator
        return float(np.arccosh(max(1.0, argument)) / np.sqrt(magnitude))
    cosine = 1.0 - 2.0 * magnitude * squared_distance / denominator
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)) / np.sqrt(magnitude))


class SectionalCurvatureAdapter:
    def __init__(self, curvature=0.0, learning_rate=1e-3):
        if not np.isfinite(curvature) or not np.isfinite(learning_rate) or learning_rate < 0.0:
            raise ValueError("curvature must be finite and learning_rate non-negative")
        self.curvature = float(curvature)
        self.learning_rate = float(learning_rate)

    def distance(self, x, y):
        return constant_curvature_distance(self.curvature, x, y)

    def update(self, free_energy_gradient, bounds=(-10.0, 10.0)):
        if not np.isfinite(free_energy_gradient):
            raise ValueError("free_energy_gradient must be finite")
        lower, upper = bounds
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError("bounds must be finite and increasing")
        self.curvature = float(np.clip(
            self.curvature - self.learning_rate * free_energy_gradient,
            lower,
            upper,
        ))
        return self.curvature