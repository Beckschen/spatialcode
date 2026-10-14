"""
Core data structures for spatial code representation.

An ObjectCode encodes a single object's 3D state:
    - label:       semantic category (str)
    - position:    3D center (x, y, z) in meters
    - size:        3D extent (w, h, l) in meters
    - orientation: quaternion (x, y, z, w)

A SceneCode is a collection of ObjectCodes that represents an entire scene.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Sequence

import numpy as np

from spatialcode.utils.geometry import euler_to_quat


# ── ObjectCode ────────────────────────────────────────────────────────

@dataclass
class ObjectCode:
    """3D state of a single object in a scene.

    Attributes:
        label:       Semantic category string (e.g., "sofa", "table").
        position:    3D center coordinates [x, y, z] in meters.
        size:        3D bounding box dimensions [w, h, l] in meters.
        orientation: Rotation quaternion [qx, qy, qz, qw].
        track_id:    Unique tracking identity across frames (optional).
        confidence:  Detection confidence score in [0, 1] (optional).
        frame_ids:   List of frame indices where this object appears.
    """

    label: str
    position: np.ndarray  # (3,)
    size: np.ndarray      # (3,)
    orientation: np.ndarray = field(default_factory=lambda: np.array([0., 0., 0., 1.]))  # (4,) identity quat
    track_id: Optional[int] = None
    confidence: float = 1.0
    frame_ids: List[int] = field(default_factory=list)

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float32)
        self.size = np.asarray(self.size, dtype=np.float32)
        self.orientation = np.asarray(self.orientation, dtype=np.float32)

    # ── Properties ────────────────────────────────────────────────────

    @property
    def volume(self) -> float:
        """Volume of the bounding box in cubic meters."""
        return float(np.prod(self.size))

    @property
    def max_extent(self) -> float:
        """Largest dimension (useful for 'object size' queries)."""
        return float(np.max(self.size))

    @property
    def yaw(self) -> float:
        """Yaw angle in radians, extracted from quaternion."""
        qx, qy, qz, qw = self.orientation
        return float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2)))

    @property
    def corners_3d(self) -> np.ndarray:
        """8 corners of the oriented 3D bounding box. Shape: (8, 3)."""
        from spatialcode.utils.geometry import bbox_to_corners
        return bbox_to_corners(self.position, self.size, self.orientation)

    # ── Serialization ─────────────────────────────────────────────────

    def to_bbox_str(self) -> str:
        """Serialize to the compact bbox_3d format used in LLM prompts.

        Returns:
            e.g. '{"label":"sofa","bbox_3d":[-1.28,1.55,0.27,0.81,1.70,0.53,0,0,0]}'
        """
        if len(self.orientation) == 4:
            roll, pitch, yaw = _quat_to_euler(self.orientation)
        else:
            roll, pitch, yaw = self.orientation
        vals = [*self.position.tolist(), *self.size.tolist(), round(roll, 2), round(pitch, 2), round(yaw, 2)]
        vals_str = ",".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in vals)
        return f'{{"label":"{self.label}","bbox_3d":[{vals_str}]}}'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "position": self.position.tolist(),
            "size": self.size.tolist(),
            "orientation": self.orientation.tolist(),
            "track_id": self.track_id,
            "confidence": self.confidence,
            "frame_ids": self.frame_ids,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ObjectCode":
        return cls(
            label=d["label"],
            position=np.array(d["bbox_3d"][:3]),
            size=np.array(d["bbox_3d"][3:6]),
            orientation=np.array(d["bbox_3d"][6:]),
            track_id=d.get("track_id"),
            confidence=d.get("confidence", 1.0),
            frame_ids=d.get("frame_ids", [0]),
        )

    @classmethod
    def from_bbox_dict(cls, d: Dict[str, Any]) -> "ObjectCode":
        """Build ObjectCode from VSI-Bench / RoboSpatial bbox dict.

        Expects keys: "bbox_3d" [x, y, z, x_size, y_size, z_size, roll, pitch, yaw], "label".
        """
        bbox = d.get("bbox_3d", [])
        if len(bbox) < 9:
            bbox = list(bbox) + [0.0] * (9 - len(bbox))
        x, y, z = float(bbox[0]), float(bbox[1]), float(bbox[2])
        sx, sy, sz = float(bbox[3]), float(bbox[4]), float(bbox[5])
        roll, pitch, yaw = float(bbox[6]), float(bbox[7]), float(bbox[8])
        orientation = euler_to_quat(roll, pitch, yaw)
        return cls(
            label=d.get("label", "object"),
            position=np.array([x, y, z], dtype=np.float32),
            size=np.array([sx, sy, sz], dtype=np.float32),
            orientation=orientation,
        )

    def to_bbox_dict(self) -> Dict[str, Any]:
        """Serialize to VSI-Bench style dict: {"bbox_3d": [...], "label": ...}."""
        if len(self.orientation) == 4:
            roll, pitch, yaw = _quat_to_euler(self.orientation)
        else:
            roll, pitch, yaw = self.orientation
        bbox_3d = [
            float(self.position[0]), float(self.position[1]), float(self.position[2]),
            float(self.size[0]), float(self.size[1]), float(self.size[2]),
            float(roll), float(pitch), float(yaw),
        ]
        roll, pitch, yaw = _quat_to_euler(self.orientation)
        bbox_3d = [
            float(self.position[0]), float(self.position[1]), float(self.position[2]),
            float(self.size[0]), float(self.size[1]), float(self.size[2]),
            float(roll), float(pitch), float(yaw),
        ]
        return {"bbox_3d": bbox_3d, "label": self.label}

    def __repr__(self) -> str:
        pos = ", ".join(f"{v:.2f}" for v in self.position)
        return f'ObjectCode("{self.label}", pos=[{pos}])'


# ── SceneCode ─────────────────────────────────────────────────────────

@dataclass
class SceneCode:
    """Collection of ObjectCodes representing an entire scene.

    Attributes:
        objects:   List of ObjectCode instances.
        metadata:  Optional dict for scene-level info (camera params, etc.).
    """

    objects: List[ObjectCode] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Accessors ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.objects)

    def __getitem__(self, idx: int) -> ObjectCode:
        return self.objects[idx]

    def __iter__(self):
        return iter(self.objects)

    def get_by_label(self, label: str) -> List[ObjectCode]:
        """Return all objects matching a label (case-insensitive)."""
        label_lower = label.lower()
        return [o for o in self.objects if o.label.lower() == label_lower]

    def get_by_track_id(self, track_id: int) -> Optional[ObjectCode]:
        """Return object with given track_id."""
        for o in self.objects:
            if o.track_id == track_id:
                return o
        return None

    @property
    def labels(self) -> List[str]:
        return [o.label for o in self.objects]

    # ── Serialization for LLM prompts ─────────────────────────────────

    def to_prompt_str(self) -> str:
        """Serialize entire scene to the text format consumed by LLMs.

        Returns:
            Multi-line string of bbox_3d entries, one per object.
        """
        return "\n".join(o.to_bbox_str() for o in self.objects)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objects": [o.to_dict() for o in self.objects],
            "metadata": self.metadata,
        }

    def to_json(self, path: Optional[str] = None) -> str:
        s = json.dumps(self.to_dict(), indent=2)
        if path:
            with open(path, "w") as f:
                f.write(s)
        return s

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SceneCode":
        return cls(
            objects=[ObjectCode.from_dict(o) for o in d["gt_3d_bboxes"]],
            metadata=d.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, path: str) -> "SceneCode":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_bbox_list(cls, bboxes: List[Dict[str, Any]]) -> "SceneCode":
        """Build SceneCode from VSI-Bench / RoboSpatial list of bbox dicts.

        Each dict has "bbox_3d" and "label"; extra keys (e.g. room_id, wall_index) are
        preserved in metadata for PA-style prompt building.
        """
        objects = [ObjectCode.from_bbox_dict(b) for b in bboxes]
        return cls(objects=objects, metadata={"_bbox_list": list(bboxes)})

    def to_bbox_list(self) -> List[Dict[str, Any]]:
        """Return list of bbox dicts for PA-style prompts.

        If this SceneCode was built with from_bbox_list(), returns the original dicts
        (with room_id, wall_index, etc.). Otherwise returns minimal {"bbox_3d", "label"} per object.
        """
        if self.metadata.get("_bbox_list") is not None:
            return self.metadata["_bbox_list"]
        return [o.to_bbox_dict() for o in self.objects]

    def __repr__(self) -> str:
        labels = ", ".join(self.labels[:5])
        suffix = ", ..." if len(self) > 5 else ""
        return f"SceneCode({len(self)} objects: [{labels}{suffix}])"


# ── Helpers ───────────────────────────────────────────────────────────

def _quat_to_euler(q: np.ndarray) -> tuple:
    """Convert quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw)."""
    qx, qy, qz, qw = q
    # Roll
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx**2 + qy**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # Pitch
    sinp = 2 * (qw * qy - qz * qx)
    pitch = np.arcsin(np.clip(sinp, -1, 1))
    # Yaw
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy**2 + qz**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return float(roll), float(pitch), float(yaw)
