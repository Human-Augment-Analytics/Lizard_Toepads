"""Oriented bounding box: center/size/angle <-> ordered corner points.

Angle is in radians, counter-clockwise, applied about the box center. Corners
are returned in a fixed order (TL, TR, BR, BL as defined in the box's own
unrotated frame) so downstream consumers read them deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OBB:
    cx: float
    cy: float
    w: float
    h: float
    angle: float  # radians, CCW

    def corners(self) -> np.ndarray:
        """Return the 4x2 rotated corners in order TL, TR, BR, BL."""
        hw, hh = self.w / 2.0, self.h / 2.0
        local = np.array(
            [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=np.float64
        )
        c, s = np.cos(self.angle), np.sin(self.angle)
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        return local @ rot.T + np.array([self.cx, self.cy])

    @classmethod
    def from_corners(cls, pts: np.ndarray) -> "OBB":
        """Build an OBB from 4x2 corners ordered TL, TR, BR, BL."""
        p = np.asarray(pts, dtype=np.float64).reshape(4, 2)
        center = p.mean(axis=0)
        edge_w = p[1] - p[0]  # TL -> TR
        edge_h = p[3] - p[0]  # TL -> BL
        w = float(np.hypot(*edge_w))
        h = float(np.hypot(*edge_h))
        angle = float(np.arctan2(edge_w[1], edge_w[0]))
        return cls(float(center[0]), float(center[1]), w, h, angle)
