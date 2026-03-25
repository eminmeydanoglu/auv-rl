"""Shared tuple-based type aliases and constants."""

from __future__ import annotations


Vector3 = tuple[float, float, float]
Vector6 = tuple[float, float, float, float, float, float]
Vector8 = tuple[float, float, float, float, float, float, float, float]
Matrix6x6 = tuple[
    Vector6,
    Vector6,
    Vector6,
    Vector6,
    Vector6,
    Vector6,
]

_ZERO_3: Vector3 = (0.0, 0.0, 0.0)
_ZERO_6X6: Matrix6x6 = (
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
)


__all__ = [
    "Matrix6x6",
    "Vector3",
    "Vector6",
    "Vector8",
    "_ZERO_3",
    "_ZERO_6X6",
]
