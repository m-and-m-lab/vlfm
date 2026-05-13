# Copyright (c) 2026 M&M Lab. All rights reserved.

"""IsaacSim ObjectNav environment for VLFM.

`IsaacSimObjectNavEnv` produces the same observation-dict shape and consumes
the same action-dict shape as `vlfm.reality.objectnav_env.ObjectNavEnv`. The
policy layer above it is therefore unmodified.

Differences from the reality env (all confined to this class):

1. Action path uses **continuous velocity** via `IsaaclabSpot.drive_for_duration`
   instead of `set_base_position` waypoints, because that's what
   `SpotLocomotionClient` consumes natively.
2. The init-phase action (`arm_yaw != -1`) is reinterpreted as a **body-yaw
   target** in episodic frame: the env rotates the robot in place to that
   heading, so the value/object maps get populated from the front cameras
   before exploration starts. Reality used 8 arm-yaw poses; we use 8 body-yaw
   poses set by `IsaacSimMixin`.
3. No per-step visualization writes (reality dumps PNGs every step under
   `vis/<datestring>/`; that's orthogonal to the sensor/action contract and is
   omitted here to avoid an cv2-on-disk dependency).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from depth_camera_filtering import filter_depth

from vlfm.isaacsim.robots.camera_ids import (
    ALL_CAMS,
    POINT_CLOUD_CAMS,
    IsaacSimCamIds,
    VALUE_MAP_CAMS,
)
from vlfm.isaacsim.robots.isaaclab_spot import IsaaclabSpot
from vlfm.utils.geometry_utils import get_fov, wrap_heading


class IsaacSimObjectNavEnv:
    """ObjectNav env in IsaacSim.

    Args:
        robot: `IsaaclabSpot` already wired to a running IsaacLab scene.
        max_body_cam_depth: Per-frame depth clip (meters) for the front
            cameras. Mirrors `reality.yaml:env.max_body_cam_depth`.
        max_gripper_cam_depth: Same for the hand camera.
        max_lin_vel: Per-step linear velocity ceiling (m/s). Policy outputs in
            [-1, 1] are scaled by this. `max_lin_vel * time_step` should land
            near reality's `max_lin_dist` (~0.2 m) so the obstacle/value maps
            see comparable per-step displacements.
        max_ang_vel: Per-step angular velocity ceiling (rad/s).
        time_step: Real-time duration of one decision cycle's drive segment.
        max_steps: Safety cap on episode length.
    """

    tf_episodic_to_global: np.ndarray = np.eye(4)
    tf_global_to_episodic: np.ndarray = np.eye(4)
    episodic_start_yaw: float = 0.0
    target_object: str = ""

    def __init__(
        self,
        robot: IsaaclabSpot,
        max_body_cam_depth: float,
        max_gripper_cam_depth: float,
        max_lin_vel: float,
        max_ang_vel: float,
        time_step: float,
        max_steps: int = 500,
    ) -> None:
        self.robot = robot
        self._max_body_cam_depth = float(max_body_cam_depth)
        self._max_gripper_cam_depth = float(max_gripper_cam_depth)
        self._max_lin_vel = float(max_lin_vel)
        self._max_ang_vel = float(max_ang_vel)
        self._time_step = float(time_step)
        self._max_steps = int(max_steps)
        self._num_steps = 0

    # ------------------------------------------------------------------ lifecycle

    def reset(self, goal: Any) -> Dict[str, Any]:
        """Start a new episode. Captures the current world pose as the episodic
        origin (z forced to 0 to stay planar).
        """
        assert isinstance(goal, str), f"goal must be a class name string, got {type(goal)}"
        self.target_object = goal
        self.tf_episodic_to_global = self.robot.get_transform("body")
        self.tf_episodic_to_global[2, 3] = 0.0
        self.tf_global_to_episodic = np.linalg.inv(self.tf_episodic_to_global)
        self.episodic_start_yaw = self.robot.xy_yaw[1]
        self._num_steps = 0
        return self._get_obs()

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Apply one action and return the next observation.

        Action contract (matches reality):
            - ``arm_yaw != -1``: init-phase body-yaw target (radians, episodic frame).
              We rotate in place; no forward motion.
            - ``arm_yaw == -1``: ``linear``/``angular`` are normalized in [-1, 1]
              (consistent with the pointnav policy's output range). We scale
              them by ``max_lin_vel``/``max_ang_vel`` and drive for ``time_step``.
            - ``linear == 0.0 and angular == 0.0`` (with ``arm_yaw == -1``):
              policy is signaling stop → ``done=True``.

        Returns ``(obs, reward, done, info)``. Reward is always 0 (unused).
        """
        arm_yaw = action["arm_yaw"]

        if arm_yaw != -1:
            self._rotate_to_yaw(target_episodic_yaw=float(arm_yaw))
            self._num_steps += 1
            done = self._num_steps >= self._max_steps
            return self._get_obs(), 0.0, done, {}

        raw_linear = float(action["linear"])
        raw_angular = float(action["angular"])
        done = (raw_linear == 0.0 and raw_angular == 0.0)

        if done:
            self.robot.stand()
        else:
            lin_vel = float(np.clip(raw_linear, -1.0, 1.0)) * self._max_lin_vel
            ang_vel = float(np.clip(raw_angular, -1.0, 1.0)) * self._max_ang_vel
            self.robot.drive_for_duration(
                lin_vel=lin_vel, ang_vel=ang_vel, duration=self._time_step
            )

        self._num_steps += 1
        done = done or (self._num_steps >= self._max_steps)
        return self._get_obs(), 0.0, done, {}

    def _rotate_to_yaw(self, target_episodic_yaw: float) -> None:
        """Drive the base in place to face ``target_episodic_yaw`` (radians).

        Uses `wrap_heading` to take the short way around. The drive duration is
        chosen so that ``ang_vel * duration`` covers the wrapped delta exactly.
        """
        current_yaw = self._get_compass()
        delta = wrap_heading(target_episodic_yaw - current_yaw)
        if abs(delta) < 1e-3:
            return
        ang_vel = float(np.sign(delta)) * self._max_ang_vel
        duration = abs(delta) / self._max_ang_vel
        self.robot.drive_for_duration(lin_vel=0.0, ang_vel=ang_vel, duration=duration)

    # ----------------------------------------------------------------- observation

    def _get_obs(self) -> Dict[str, Any]:
        robot_xy, robot_heading = self._get_gps(), self._get_compass()
        nav_depth, obstacle_map_depths, value_map_rgbd, object_map_rgbd = self._get_camera_obs()
        return {
            "nav_depth": nav_depth,
            "robot_xy": robot_xy,
            "robot_heading": robot_heading,
            "objectgoal": self.target_object,
            "obstacle_map_depths": obstacle_map_depths,
            "value_map_rgbd": value_map_rgbd,
            "object_map_rgbd": object_map_rgbd,
        }

    def _get_camera_obs(self) -> Tuple[np.ndarray, List, List, List]:
        """Mirror of `ObjectNavEnv._get_camera_obs` adapted for IsaacSim cams.

        The dtype-dispatched normalization branches (uint16 depth → meters →
        [0,1], uint8 BGR → RGB) are identical to reality's, because
        `IsaaclabSpot.get_camera_data` returns the same dtype contract.
        """
        srcs: List[str] = ALL_CAMS
        cam_data = self.robot.get_camera_data(srcs)
        # Camera→xyz axis remap: rotates from optical convention (z-forward,
        # y-down, x-right) to VLFM's xyz convention (x-forward, y-left, z-up).
        # Same matrix as reality (vlfm/reality/objectnav_env.py:141).
        cam_to_xyz = np.array(
            [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float64
        )
        for src in ALL_CAMS:
            tf_global_episodic_cam = self.tf_global_to_episodic @ cam_data[src]["tf_camera_to_global"]
            cam_data[src]["tf_camera_to_global"] = np.dot(tf_global_episodic_cam, cam_to_xyz)

            img = cam_data[src]["image"]
            if img.dtype == np.uint16:
                max_depth = self._max_gripper_cam_depth if "hand" in src else self._max_body_cam_depth
                img = self._norm_depth(img, max_depth=max_depth)
                if "front" not in src:
                    img = filter_depth(img, blur_type=None, recover_nonzero=False)
                cam_data[src]["image"] = img

            if img.dtype == np.uint8:
                if img.ndim == 2 or img.shape[2] == 1:
                    cam_data[src]["image"] = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                else:
                    cam_data[src]["image"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        min_depth = 0

        # Object map: (rgb, depth_placeholder, tf_cam_episodic, min, max, fx, fy).
        # Hand depth is a placeholder ones-array — the mixin's `_cache_observations`
        # / `_infer_depth` re-fills it from a monocular depth estimator at run time.
        src = IsaacSimCamIds.HAND
        rgb = cam_data[src]["image"]
        hand_depth = np.ones(rgb.shape[:2], dtype=np.float32)
        tf = cam_data[src]["tf_camera_to_global"]
        max_depth = self._max_gripper_cam_depth
        fx, fy = cam_data[src]["fx"], cam_data[src]["fy"]
        object_map_rgbd = [(rgb, hand_depth, tf, min_depth, max_depth, fx, fy)]

        # Nav depth: stacked front-right + front-left, filtered. `reorient_images`
        # is a no-op on the IsaacLab cams (they stream upright); kept for API
        # parity so the env code reads cleanly against either backend.
        f_left = IsaacSimCamIds.FRONTLEFT
        f_right = IsaacSimCamIds.FRONTRIGHT
        nav_cam_data = self.robot.reorient_images(
            {k: cam_data[k]["image"] for k in [f_left, f_right]}
        )
        nav_depth = np.hstack([nav_cam_data[f_right], nav_cam_data[f_left]])
        nav_depth = filter_depth(nav_depth, blur_type=None, set_black_value=1.0)

        # Obstacle map: one tuple per point-cloud cam, plus a terminal "explore-
        # only" entry using the hand cam's tf (depth=None signals
        # ObstacleMap.update_map to grow the explored-area mask without writing
        # obstacles). Matches reality's `obstacle_map_depths[-1]` convention at
        # vlfm/reality/objectnav_env.py:202-209.
        obstacle_map_depths = []
        for cam_src in POINT_CLOUD_CAMS:
            depth = cam_data[cam_src]["image"]
            fx, fy = cam_data[cam_src]["fx"], cam_data[cam_src]["fy"]
            tf = cam_data[cam_src]["tf_camera_to_global"]
            fov = get_fov(fy, depth.shape[0])
            obstacle_map_depths.append(
                (depth, tf, min_depth, self._max_body_cam_depth, fx, fy, fov)
            )

        hand_tf = cam_data[IsaacSimCamIds.HAND]["tf_camera_to_global"]
        hand_fx = cam_data[IsaacSimCamIds.HAND]["fx"]
        hand_fy = cam_data[IsaacSimCamIds.HAND]["fy"]
        hand_fov = get_fov(hand_fx, cam_data[IsaacSimCamIds.HAND]["image"].shape[1])
        obstacle_map_depths.append(
            (None, hand_tf, min_depth, self._max_body_cam_depth, hand_fx, hand_fy, hand_fov)
        )

        # Value map: (rgb, depth, tf_cam_episodic, min, max, fov). Same hand-
        # depth placeholder convention as object_map_rgbd.
        value_map_rgbd = []
        for rgb_src in VALUE_MAP_CAMS:
            rgb = cam_data[rgb_src]["image"]
            fx_rgb = cam_data[rgb_src]["fx"]
            tf_rgb = cam_data[rgb_src]["tf_camera_to_global"]
            fov_rgb = get_fov(fx_rgb, rgb.shape[1])
            value_map_rgbd.append(
                (rgb, hand_depth, tf_rgb, min_depth, max_depth, fov_rgb)
            )

        return nav_depth, obstacle_map_depths, value_map_rgbd, object_map_rgbd

    def _norm_depth(self, depth: np.ndarray, max_depth: float) -> np.ndarray:
        """Convert raw uint16 mm depth to normalized [0, 1] float meters/max.

        Matches `PointNavEnv._norm_depth` semantics so the obstacle/value maps
        receive numerically equivalent input as reality.
        """
        norm = depth.astype(np.float32) / 1000.0  # mm -> m
        norm = np.clip(norm, 0.0, max_depth) / max_depth
        return norm

    def _get_gps(self) -> np.ndarray:
        """Return ``(x, y)`` in episodic frame (x=forward, y=left)."""
        global_xy = self.robot.xy_yaw[0]
        start_xy = self.tf_episodic_to_global[:2, 3]
        offset = global_xy - start_xy
        rot = np.array(
            [
                [np.cos(-self.episodic_start_yaw), -np.sin(-self.episodic_start_yaw)],
                [np.sin(-self.episodic_start_yaw), np.cos(-self.episodic_start_yaw)],
            ]
        )
        return rot @ offset

    def _get_compass(self) -> float:
        """Return yaw (radians, [-π, π]) in episodic frame."""
        global_yaw = self.robot.xy_yaw[1]
        return wrap_heading(global_yaw - self.episodic_start_yaw)
