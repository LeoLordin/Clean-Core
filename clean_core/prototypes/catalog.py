"""Uniform discovery helpers for the finite theory and algorithm prototypes."""

from importlib import import_module
from types import ModuleType


def _validate_number(number, maximum, kind):
    if isinstance(number, bool) or not isinstance(number, int):
        raise ValueError(f"{kind} number must be an integer")
    if not 1 <= number <= maximum:
        raise ValueError(f"{kind} number must be between 1 and {maximum}")
    return number


def load_section(number: int) -> ModuleType:
    """Import one formula-section prototype by its documented number."""
    number = _validate_number(number, 38, "section")
    return import_module(f"clean_core.prototypes.sections.section_{number:02d}")


def load_algorithm(number: int) -> ModuleType:
    """Import one algorithm prototype by its documented number."""
    number = _validate_number(number, 13, "algorithm")
    return import_module(f"clean_core.prototypes.algorithms.algorithm_{number:02d}")


def load_all_prototypes() -> tuple[tuple[ModuleType, ...], tuple[ModuleType, ...]]:
    """Import and return every section and algorithm prototype in numeric order."""
    sections = tuple(load_section(number) for number in range(1, 39))
    algorithms = tuple(load_algorithm(number) for number in range(1, 14))
    return sections, algorithms


__all__ = ["load_section", "load_algorithm", "load_all_prototypes"]
