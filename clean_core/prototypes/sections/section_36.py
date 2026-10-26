"""Section 36 prototype for sectional-curvature chart distances and updates."""

from clean_core.geometry.curvature import (
    SectionalCurvatureAdapter,
    constant_curvature_distance,
)


def section_36_distance(curvature, x, y):
    """Return the signed-curvature chart distance for two coordinate vectors.

    Flat coordinates use the zero-curvature limit; positive curvature uses a
    spherical chart and negative curvature uses the Poincare-ball chart.
    """
    return constant_curvature_distance(curvature, x, y)


class Section36CurvatureAdapter(SectionalCurvatureAdapter):
    """Curvature-dependent distance adapter with bounded gradient updates."""


__all__ = ["Section36CurvatureAdapter", "section_36_distance"]
