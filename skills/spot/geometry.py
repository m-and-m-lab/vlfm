"""Small Spot-style geometry helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EulerZXY:
    """Euler rotation matching the Spot SDK's Z-X-Y helper."""

    yaw: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0

    def to_quaternion(self) -> tuple[float, float, float, float]:
        """Return the orientation as a quaternion in ``(w, x, y, z)`` order."""

        half_yaw = 0.5 * self.yaw
        half_roll = 0.5 * self.roll
        half_pitch = 0.5 * self.pitch

        cy = math.cos(half_yaw)
        sy = math.sin(half_yaw)
        cr = math.cos(half_roll)
        sr = math.sin(half_roll)
        cp = math.cos(half_pitch)
        sp = math.sin(half_pitch)

        # Compose Rz(yaw) * Rx(roll) * Ry(pitch).
        w = cy * cr * cp - sy * sr * sp
        x = cy * sr * cp - sy * cr * sp
        y = cy * cr * sp + sy * sr * cp
        z = sy * cr * cp + cy * sr * sp
        return (w, x, y, z)
