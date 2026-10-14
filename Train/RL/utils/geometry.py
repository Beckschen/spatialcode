"""
3D geometry utilities for spatial code operations.

Covers: quaternion math, bbox corners, IoU, coordinate transforms.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


# ── Quaternion operations ─────────────────────────────────────────────

def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to 3x3 rotation matrix."""
    qx, qy, qz, qw = q
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),     1 - 2*(qx**2 + qz**2),  2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),      1 - 2*(qx**2 + qy**2)],
    ])


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Euler angles (rad) -> quaternion (x, y, z, w)."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ], dtype=np.float32)


def quat_mean(quats: np.ndarray) -> np.ndarray:
    """Average quaternions with sign-ambiguity handling. Shape: (N, 4)."""
    # Flip quaternions to same hemisphere
    ref = quats[0]
    for i in range(1, len(quats)):
        if np.dot(quats[i], ref) < 0:
            quats[i] = -quats[i]
    avg = quats.mean(axis=0)
    return avg / np.linalg.norm(avg)


# ── Bounding box operations ──────────────────────────────────────────

_UNIT_CORNERS = np.array([
    [-0.5, -0.5, -0.5], [-0.5, -0.5,  0.5],
    [-0.5,  0.5, -0.5], [-0.5,  0.5,  0.5],
    [ 0.5, -0.5, -0.5], [ 0.5, -0.5,  0.5],
    [ 0.5,  0.5, -0.5], [ 0.5,  0.5,  0.5],
], dtype=np.float32)  # (8, 3)


def bbox_to_corners(
    position: np.ndarray,
    size: np.ndarray,
    orientation: np.ndarray,
) -> np.ndarray:
    """Compute 8 corners of an oriented 3D bounding box.

    Args:
        position:    (3,) center.
        size:        (3,) dimensions [w, h, l].
        orientation: (4,) quaternion [qx, qy, qz, qw].

    Returns:
        (8, 3) array of corner coordinates.
    """
    R = quat_to_rotation_matrix(orientation)
    corners = _UNIT_CORNERS * size[None, :]       # scale
    corners = (R @ corners.T).T                    # rotate
    corners += position[None, :]                   # translate
    return corners


def iou_3d(
    pos1: np.ndarray, size1: np.ndarray, ori1: np.ndarray,
    pos2: np.ndarray, size2: np.ndarray, ori2: np.ndarray,
) -> float:
    """Approximate 3D IoU between two oriented bounding boxes.

    Uses axis-aligned approximation for speed; for oriented IoU
    use `iou_3d_oriented` (requires shapely).
    """
    # Axis-aligned approximation: compute AABB of each OBB
    c1 = bbox_to_corners(pos1, size1, ori1)
    c2 = bbox_to_corners(pos2, size2, ori2)

    min1, max1 = c1.min(0), c1.max(0)
    min2, max2 = c2.min(0), c2.max(0)

    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_size = np.maximum(inter_max - inter_min, 0)
    inter_vol = float(np.prod(inter_size))

    vol1 = float(np.prod(max1 - min1))
    vol2 = float(np.prod(max2 - min2))
    union = vol1 + vol2 - inter_vol
    return inter_vol / max(union, 1e-8)


def min_distance(
    pos1: np.ndarray, size1: np.ndarray, ori1: np.ndarray,
    pos2: np.ndarray, size2: np.ndarray, ori2: np.ndarray,
) -> float:
    """Minimum Euclidean distance between closest points of two 3D boxes."""
    c1 = bbox_to_corners(pos1, size1, ori1)  # (8, 3)
    c2 = bbox_to_corners(pos2, size2, ori2)  # (8, 3)
    # Brute-force over corner pairs (fast for 8x8)
    dists = np.linalg.norm(c1[:, None, :] - c2[None, :, :], axis=-1)
    return float(dists.min())


# ── Coordinate transforms ────────────────────────────────────────────

def world_to_local(
    point: np.ndarray,
    origin: np.ndarray,
    forward: np.ndarray,
) -> Tuple[float, float]:
    """Project a point into a local 2D frame (front/right).

    Args:
        point:   (2,) or (3,) target position (uses x, y only).
        origin:  (2,) or (3,) observer position.
        forward: (2,) or (3,) forward direction vector.

    Returns:
        (front_component, right_component) in local frame.
    """
    fwd = np.array(forward[:2], dtype=np.float64)
    fwd = fwd / (np.linalg.norm(fwd) + 1e-8)
    right = np.array([fwd[1], -fwd[0]])
    delta = np.array(point[:2], dtype=np.float64) - np.array(origin[:2], dtype=np.float64)
    return float(delta @ fwd), float(delta @ right)
