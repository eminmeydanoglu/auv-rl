"""Thruster allocation helpers used by MJLab AUV tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _allocation_matrix_from_layout(
    positions_body_m: np.ndarray,
    directions_body: np.ndarray,
) -> np.ndarray:
    positions = np.asarray(positions_body_m, dtype=float)
    directions = np.asarray(directions_body, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"positions_body_m must have shape (N, 3), got {positions.shape}"
        )
    if directions.ndim != 2 or directions.shape != positions.shape:
        raise ValueError(
            f"directions_body must have shape {positions.shape}, got {directions.shape}"
        )

    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    if np.any(norms < 1.0e-12):
        raise ValueError("directions_body contains a near-zero direction vector")

    directions_unit = directions / norms
    moments = np.cross(positions, directions_unit)
    return np.vstack([directions_unit.T, moments.T])


def allocation_matrix_from_mujoco_sites(
    model: Any,
    data: Any,
    *,
    body_name: str,
    site_names: Sequence[str],
    local_force_axis: Sequence[float],
) -> np.ndarray:
    """Build a 6xN body-frame allocation matrix from MuJoCo site poses."""
    if not site_names:
        raise ValueError("site_names must contain at least one entry")

    try:
        import mujoco  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("mujoco package is required for allocation helpers") from exc

    mj_name2id = getattr(mujoco, "mj_name2id")
    mj_id2name = getattr(mujoco, "mj_id2name")
    mjt_obj = getattr(mujoco, "mjtObj")

    def resolve_name(obj_type: Any, name: str, count: int) -> int:
        exact_id = mj_name2id(model, obj_type, name)
        if exact_id >= 0:
            return int(exact_id)

        suffix = f"/{name}"
        matches: list[int] = []
        for object_id in range(count):
            resolved_name = mj_id2name(model, obj_type, object_id)
            if resolved_name == name or (
                isinstance(resolved_name, str) and resolved_name.endswith(suffix)
            ):
                matches.append(int(object_id))

        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Name '{name}' is ambiguous in MuJoCo model: {matches}")
        return -1

    body_id = resolve_name(mjt_obj.mjOBJ_BODY, body_name, model.nbody)
    if body_id < 0:
        raise ValueError(f"Body '{body_name}' not found in MuJoCo model")

    body_pos_w = np.asarray(data.xpos[body_id], dtype=float)
    body_rot_w = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
    rot_bw = body_rot_w.T

    local_axis = np.asarray(local_force_axis, dtype=float).reshape(-1)
    if local_axis.shape != (3,):
        raise ValueError(
            f"local_force_axis must have 3 elements, got {local_axis.shape}"
        )
    axis_norm = np.linalg.norm(local_axis)
    if axis_norm < 1.0e-12:
        raise ValueError("local_force_axis must be non-zero")
    local_axis = local_axis / axis_norm

    positions_b: list[np.ndarray] = []
    directions_b: list[np.ndarray] = []
    for site_name in site_names:
        site_id = resolve_name(mjt_obj.mjOBJ_SITE, site_name, model.nsite)
        if site_id < 0:
            raise ValueError(f"Site '{site_name}' not found in MuJoCo model")

        site_pos_w = np.asarray(data.site_xpos[site_id], dtype=float)
        site_rot_w = np.asarray(data.site_xmat[site_id], dtype=float).reshape(3, 3)
        direction_w = site_rot_w @ local_axis

        positions_b.append(rot_bw @ (site_pos_w - body_pos_w))
        directions_b.append(rot_bw @ direction_w)

    return _allocation_matrix_from_layout(
        np.vstack(positions_b),
        np.vstack(directions_b),
    )
