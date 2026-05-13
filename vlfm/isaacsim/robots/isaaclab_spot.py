# Copyright (c) 2026 M&M Lab. All rights reserved.

"""IsaacLab-backed Spot facade for VLFM.

`IsaaclabSpot` reproduces the public surface of `vlfm.reality.robots.bdsw_robot.BDSWRobot`
that `vlfm.reality.objectnav_env.ObjectNavEnv` reads from — `xy_yaw`,
`get_transform`, `get_camera_data`, `reorient_images` — and the action surface
the env writes to — `drive_for_duration`, `stand`, `command_base_velocity`.

The facade is intentionally shape-identical to `BDSWRobot` so the IsaacSim env
wrapper can reuse the reality observation-assembly logic without conditional
branches. In particular:

- `get_camera_data` returns `{src: {"image", "fx", "fy", "tf_camera_to_global"}}`
  where `image` is uint16 millimeters for depth cams (mirrors Spot's raw output)
  and uint8 BGR for color cams. The env applies its own float-meters
  normalization in `_get_camera_obs`.
- `xy_yaw` returns `(np.array([x, y]), yaw)` — world-frame planar pose.
- `get_transform` returns a 4x4 homogeneous body-in-world matrix.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Tuple

import numpy as np

from skills.cam_utils import capture_camera_payload, quat_wxyz_to_matrix


class IsaaclabSpot:
    """Thin facade that exposes a `BDSWRobot`-shaped API over `skills/`.

    The caller is responsible for setting up the IsaacLab scene, articulation,
    and camera sensors before instantiating this class, and for plumbing the
    locomotion + service binder. See `vlfm/isaacsim/run_isaacsim_objnav.py` for
    the expected wiring pattern.

    Args:
        robot: A `skills.spot.sdk.Robot` whose service backends have already
            been installed by `skills.spot.isaaclab_backend.bind_isaaclab_spot_robot`.
        locomotion: A `skills.locomotion.api.SpotLocomotionClient` driving the
            12-joint leg policy.
        binder: The `IsaacLabSpotServiceBinder` returned from
            `bind_isaaclab_spot_robot`. Its `.step()` is pumped on every control
            tick to advance the manipulation state machine (currently unused by
            VLFM but harmless).
        cameras: Mapping from camera name (matching the keys in
            `skills/config/spot_arm_cameras.yaml`) to the IsaacLab `Camera`
            sensor object. The env reads these via `get_camera_data`.
        scene_step: Callable that advances the IsaacLab scene by one `sim_dt`
            (typically `lambda: scene.step(render=True)` plus any per-step
            articulation writes). Called once per control tick inside
            `drive_for_duration`.
        max_body_cam_depth_m: Clip ceiling (meters) applied before the float→
            uint16-mm conversion. Mirrors `reality.yaml:max_body_cam_depth`.
        max_gripper_cam_depth_m: Same, for the hand camera.
    """

    def __init__(
        self,
        *,
        robot: Any,
        locomotion: Any,
        binder: Any,
        cameras: Mapping[str, Any],
        scene_step: Callable[[], None],
        max_body_cam_depth_m: float = 3.5,
        max_gripper_cam_depth_m: float = 5.0,
    ) -> None:
        self._robot = robot
        self._locomotion = locomotion
        self._binder = binder
        self._cameras = dict(cameras)
        self._scene_step = scene_step
        self._max_body_cam_depth_m = float(max_body_cam_depth_m)
        self._max_gripper_cam_depth_m = float(max_gripper_cam_depth_m)
        self._gripper_cam_names = {"hand"}

    # ------------------------------------------------------------------ state

    @property
    def xy_yaw(self) -> Tuple[np.ndarray, float]:
        """Planar world-frame pose: ``(np.array([x, y]), yaw)``.

        Yaw is extracted from the body quaternion as a z-axis rotation. Spot
        moves on flat ground, so reducing the orientation to planar yaw matches
        the reality contract (`BDSWRobot.xy_yaw`).
        """
        pos, quat_wxyz = self._read_root_pose_w()
        yaw = _planar_yaw_from_quat_wxyz(quat_wxyz)
        return np.array([float(pos[0]), float(pos[1])], dtype=np.float64), float(yaw)

    def get_transform(self, frame: str = "body") -> np.ndarray:
        """Return the 4x4 body-in-world transform.

        Only ``frame="body"`` is implemented — the env never asks for any other
        link in reality, and supporting more would mean replicating Spot's TF
        snapshot machinery, which the IsaacLab scene already exposes through
        the articulation directly.
        """
        if frame != "body":
            raise NotImplementedError(
                f"IsaaclabSpot.get_transform only supports frame='body' (got {frame!r})."
            )
        pos, quat_wxyz = self._read_root_pose_w()
        tf = np.eye(4, dtype=np.float64)
        tf[:3, :3] = quat_wxyz_to_matrix(quat_wxyz)
        tf[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
        return tf

    def _read_root_pose_w(self) -> Tuple[np.ndarray, np.ndarray]:
        """Read world-frame ``(position, quat_wxyz)`` from the bound RobotState."""
        state_client = self._robot.ensure_client("robot-state")
        state = state_client.get_robot_state()
        root_pose = _to_numpy_1d(state.root_pose_w)
        if root_pose.shape != (7,):
            raise ValueError(
                f"Expected root_pose_w shape (7,), got {root_pose.shape}. "
                "Layout must be [x, y, z, qw, qx, qy, qz]."
            )
        return root_pose[:3], root_pose[3:]

    # ----------------------------------------------------------------- camera

    def get_camera_data(self, srcs: List[str]) -> Dict[str, Dict[str, Any]]:
        """Capture one frame from each named camera.

        Returns a dict shape-identical to `BDSWRobot.get_camera_data`:
        ``{src: {"image": np.ndarray, "fx": float, "fy": float,
        "tf_camera_to_global": np.ndarray (4, 4)}}``.

        For depth-only cameras, ``image`` is uint16 millimeters (clipped to
        per-camera max-depth). For color cameras (currently only ``hand``),
        ``image`` is uint8 BGR. The env's `_get_camera_obs` does the
        meters-normalization + RGB conversion downstream.
        """
        data: Dict[str, Dict[str, Any]] = {}
        for src in srcs:
            camera = self._cameras.get(src)
            if camera is None:
                raise KeyError(
                    f"Unknown camera '{src}'. Known: {sorted(self._cameras)}."
                )
            payload = capture_camera_payload(camera, camera_axes="usd")
            fx, fy = _read_intrinsics_fx_fy(camera)
            tf_cam_to_world = _compose_tf(payload["position"], payload["quat_wxyz"])
            image = self._payload_to_image(src, payload)
            data[src] = {
                "image": image,
                "fx": float(fx),
                "fy": float(fy),
                "tf_camera_to_global": tf_cam_to_world,
            }
        return data

    def _payload_to_image(self, src: str, payload: Dict[str, Any]) -> np.ndarray:
        """Convert the cam_utils payload into the dtype/layout the env expects.

        - Depth cams: float meters → uint16 millimeters (mirrors Spot's raw
          depth source dtype, so the env's ``if img.dtype == np.uint16`` branch
          fires and `_norm_depth` does the meters+normalization).
        - Color cams: rgba → uint8 BGR (mirrors `image_response_to_cv2`, so the
          env's `cv2.cvtColor(..., COLOR_BGR2RGB)` branch produces correct RGB).
        """
        rgba = payload.get("rgba")
        depth = payload.get("depth")
        if rgba is None and depth is None:
            raise ValueError(f"Camera '{src}' produced neither rgba nor depth.")

        if rgba is not None:
            rgba_np = np.asarray(rgba)
            if rgba_np.ndim != 3 or rgba_np.shape[2] < 3:
                raise ValueError(
                    f"Camera '{src}': expected HxWx{{3,4}} rgba, got shape "
                    f"{rgba_np.shape}."
                )
            rgb_uint8 = rgba_np[..., :3].astype(np.uint8, copy=False)
            return rgb_uint8[..., ::-1].copy()  # RGB -> BGR

        depth_np = np.asarray(depth, dtype=np.float32)
        if depth_np.ndim == 3 and depth_np.shape[2] == 1:
            depth_np = depth_np[..., 0]
        if depth_np.ndim != 2:
            raise ValueError(
                f"Camera '{src}': expected HxW or HxWx1 depth, got shape "
                f"{depth_np.shape}."
            )
        max_m = (
            self._max_gripper_cam_depth_m
            if src in self._gripper_cam_names
            else self._max_body_cam_depth_m
        )
        depth_np = np.where(np.isfinite(depth_np), depth_np, 0.0)
        depth_np = np.clip(depth_np, 0.0, max_m)
        depth_mm = np.clip(depth_np * 1000.0, 0.0, 65535.0).astype(np.uint16, copy=False)
        return depth_mm

    @staticmethod
    def reorient_images(imgs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """No-op for IsaacLab cameras (they stream upright at their native
        orientation). Kept for API parity with `BDSWRobot.reorient_images`.
        """
        return dict(imgs_dict)

    # ----------------------------------------------------------------- action

    def command_base_velocity(self, ang_vel: float, lin_vel: float) -> None:
        """Set the locomotion command (non-blocking). Mirrors `BDSWRobot.command_base_velocity`.

        Note the argument order matches `BDSWRobot`: angular first, linear
        second. Internally calls `SpotLocomotionClient.command_velocity(v_x=lin,
        v_y=0.0, v_rot=ang)`.
        """
        if abs(ang_vel) < 1e-3 and abs(lin_vel) < 1e-3:
            self.stand()
        else:
            self._locomotion.command_velocity(
                v_x=float(lin_vel), v_y=0.0, v_rot=float(ang_vel)
            )

    def stand(self) -> None:
        """Stop and swap to stand-mode actuator gains."""
        self._locomotion.stand()

    def drive_for_duration(
        self,
        lin_vel: float,
        ang_vel: float,
        duration: float,
    ) -> None:
        """Drive the base at constant velocity for ``duration`` seconds.

        This is the action-sink the IsaacSim env calls in place of reality's
        `set_base_position`. We use continuous velocity rather than discrete
        waypoints because that matches what `SpotLocomotionClient` consumes
        natively.

        The inner loop pumps three things per control tick (≈20 ms):
          1. `SpotLocomotionClient.step()` — runs the 12-joint flat-terrain
             policy on the cached velocity command.
          2. `binder.step()` — advances the manipulation backend state machine
             (no-op for VLFM, kept for parity with skills' contract).
          3. `scene_step()` — advances the IsaacLab scene by one `sim_dt` and
             renders, so the next read sees the moved robot.
        """
        if duration <= 0.0:
            return
        self._locomotion.command_velocity(
            v_x=float(lin_vel), v_y=0.0, v_rot=float(ang_vel)
        )
        ticks = max(1, int(round(duration / self._locomotion.control_dt)))
        for _ in range(ticks):
            self._locomotion.step()
            self._binder.step()
            self._scene_step()
        self._locomotion.stand()

    # --------------------------------------------------------- unused but req

    def set_arm_joints(self, joints: np.ndarray, travel_time: float) -> None:
        """No-op stub. The IsaacSim env replaces the arm-yaw init sweep with a
        body-yaw sweep, so the policy never asks the env to move the arm.
        """
        del joints, travel_time

    def open_gripper(self) -> None:
        """No-op stub (manipulation is deferred per the Section 6 scope cut)."""


# ---------------------------------------------------------------------- helpers


def _to_numpy_1d(value: Any) -> np.ndarray:
    """Coerce a tensor / numpy array / sequence to a 1-D numpy array of floats."""
    if hasattr(value, "detach"):  # torch.Tensor
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float64)
    return arr.reshape(-1)


def _planar_yaw_from_quat_wxyz(quat_wxyz: np.ndarray) -> float:
    """Extract yaw (rotation about z) from a wxyz quaternion.

    Assumes the robot's roll/pitch are negligible (Spot on flat ground). This
    matches reality where ``BDSWRobot.xy_yaw`` returns a single scalar yaw.
    """
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    w, x, y, z = q[0], q[1], q[2], q[3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _compose_tf(position: Any, quat_wxyz: Any) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a translation and wxyz quaternion."""
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    rot = quat_wxyz_to_matrix(quat_wxyz)
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = rot
    tf[:3, 3] = pos
    return tf


def _read_intrinsics_fx_fy(camera: Any) -> Tuple[float, float]:
    """Read ``(fx, fy)`` in pixels from an IsaacLab `Camera`.

    Tries `camera.data.intrinsic_matrices` (the standard IsaacLab field) first.
    The matrix is (N, 3, 3) for batched envs or (3, 3) for a single env; we
    take the first env. ``K[0,0]=fx``, ``K[1,1]=fy``.
    """
    cam_data = getattr(camera, "data", None)
    K = getattr(cam_data, "intrinsic_matrices", None) if cam_data is not None else None
    if K is None:
        raise ValueError(
            "IsaacLab Camera has no `data.intrinsic_matrices`; cannot read fx/fy."
        )
    K_np = np.asarray(K.detach().cpu().numpy()) if hasattr(K, "detach") else np.asarray(K)
    if K_np.ndim == 3:
        K_np = K_np[0]
    if K_np.shape != (3, 3):
        raise ValueError(
            f"Expected intrinsic matrix shape (3, 3), got {tuple(K_np.shape)}."
        )
    return float(K_np[0, 0]), float(K_np[1, 1])
