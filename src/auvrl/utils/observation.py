"""Observation-space helper utilities."""

from __future__ import annotations


def obs_scale_from_range(value_range: tuple[float, float]) -> float:
    return 1.0 / max(abs(value_range[0]), abs(value_range[1]), 1.0e-6)


__all__ = ["obs_scale_from_range"]
