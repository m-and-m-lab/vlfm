# Copyright (c) 2026 M&M Lab. All rights reserved.

"""Shared scene/runner/wiring helpers for the VLFM IsaacSim smoke tests.

**IMPORTANT**: This module imports `isaaclab.*` at module load time. It must
not be imported until *after* `AppLauncher(args_cli)` has started the
SimulationApp in the calling script. See `smoke_camera_parity.py` /
`smoke_env_loop.py` for the correct import order.

The scene-build pattern (SceneCfg, `_reset_robot`, `_warmup_default_pose`) is
copied verbatim from
`skills/locomotion/isaaclab_test/spot_locomotion_wasd.py` so the smokes share
the same physics setup as the existing teleop harness. The differences are:

1. Three `CameraCfg` attachments at the prim paths VLFM expects.
2. A minimal `_RunnerStub` that exposes `_build_robot_state()` — required by
   the `IsaacLabRobotStateBackend` that `IsaaclabSpot.xy_yaw` reads through.
3. A `make_robot_facade` helper that does the `bind_isaaclab_spot_robot` +
   `IsaaclabSpot` construction inline (we don't call
   `vlfm.isaacsim.run_isaacsim_objnav.build_isaacsim_components` because that
   also instantiates a real `IsaacSimITMPolicyV2`, which pulls in BLIP-2 / VLM
   servers / etc. — overkill for these smokes).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, Tuple

import numpy as np
import torch
import yaml

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg, RemotizedPDActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.spot import joint_parameter_lookup

from omegaconf import OmegaConf

from skills.core import RobotState
from skills.locomotion.api import SpotLocomotionClient
from skills.spot.isaaclab_backend import bind_isaaclab_spot_robot
from skills.spot.sdk import create_standard_sdk

from vlfm.isaacsim.robots.isaaclab_spot import IsaaclabSpot


# Arm joint order — must match the existing teleop harness so the articulation
# spawns into the same default pose. VLFM doesn't drive the arm but the binder
# still wants the joint ids/names for shape.
_ARM_JOINT_ORDER: Tuple[str, ...] = (
    "arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1",
)
_GRIPPER_JOINT_NAME: str = "arm_f1x"


def read_yaml(path: str | Path) -> dict:
    """Load a smoke-test YAML and resolve the robot's `usd_path` relative to
    the YAML's own directory (so it works regardless of cwd).
    """
    path_obj = Path(path).resolve()
    data = yaml.safe_load(path_obj.read_text()) or {}
    usd_path = Path(data["robot"]["usd_path"])
    if not usd_path.is_absolute():
        data["robot"]["usd_path"] = str((path_obj.parent / usd_path).resolve())
    return data


def make_omegaconf_cfg(yaml_data: dict) -> Any:
    """Wrap the parsed YAML in OmegaConf so it has the dotted-attribute access
    that VLFM env code (and the cfg.env.* lookups) expects.
    """
    return OmegaConf.create(yaml_data)


# ----------------------------------------------------------------- scene config


def make_scene_cfg(config: dict) -> type[InteractiveSceneCfg]:
    """Build an `InteractiveSceneCfg` subclass: Spot + ground + dome light + 3 cameras.

    The robot config (actuator gains, init joint pose, USD spawn) is identical
    to the existing teleop harness so the locomotion policy has the same
    operating point. The three `CameraCfg`s attach to existing camera prims
    inside `spot_arm_w_cam.usd` — IsaacLab wraps them without re-spawning.
    """
    robot_cfg = config["robot"]
    scene_cfg = config["scene"]
    cameras_cfg = config["cameras"]

    robot_articulation_cfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=robot_cfg["usd_path"],
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(map(float, robot_cfg["position"])),
            rot=tuple(map(float, robot_cfg["rotation_wxyz"])),
            joint_pos={
                ".*_left_hip_x": 0.1,
                ".*_right_hip_x": -0.1,
                "front_.*_hip_y": 0.9,
                "rear_.*_hip_y": 1.1,
                ".*_knee": -1.5,
                "arm_sh0": 0.0, "arm_sh1": -3.13, "arm_el0": 3.13,
                "arm_el1": 0.0, "arm_wr0": 0.0, "arm_wr1": 0.0,
                "arm_f1x": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "spot_hip": DelayedPDActuatorCfg(
                joint_names_expr=[".*_hip_[xy]"], effort_limit=45.0,
                stiffness=60.0, damping=1.5, min_delay=0, max_delay=4,
            ),
            "spot_knee": RemotizedPDActuatorCfg(
                joint_names_expr=[".*_knee"],
                joint_parameter_lookup=joint_parameter_lookup,
                effort_limit=None, stiffness=60.0, damping=1.5,
                min_delay=0, max_delay=4,
            ),
            "arm_hold": ImplicitActuatorCfg(
                joint_names_expr=list(_ARM_JOINT_ORDER),
                effort_limit_sim=300.0, stiffness=400.0, damping=60.0,
            ),
            "gripper_hold": ImplicitActuatorCfg(
                joint_names_expr=[_GRIPPER_JOINT_NAME],
                effort_limit_sim=200.0, stiffness=120.0, damping=20.0,
            ),
        },
    )

    # Camera sensors. `spawn=None` tells IsaacLab to wrap an existing prim
    # rather than create a new one. The data_types match what
    # `capture_camera_payload` expects (`rgb` + `distance_to_image_plane`).
    fl_cam_cfg = CameraCfg(
        prim_path=cameras_cfg["frontleft"]["prim_path"],
        data_types=["rgb", "distance_to_image_plane"],
        update_period=0.0,
        spawn=None,
    )
    fr_cam_cfg = CameraCfg(
        prim_path=cameras_cfg["frontright"]["prim_path"],
        data_types=["rgb", "distance_to_image_plane"],
        update_period=0.0,
        spawn=None,
    )
    hand_cam_cfg = CameraCfg(
        prim_path=cameras_cfg["hand"]["prim_path"],
        data_types=["rgb", "distance_to_image_plane"],
        update_period=0.0,
        spawn=None,
    )

    @configclass
    class _SmokeSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
        light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(
                intensity=float(scene_cfg["dome_light_intensity"]),
                color=tuple(map(float, scene_cfg["dome_light_color"])),
            ),
        )
        robot: ArticulationCfg = robot_articulation_cfg
        frontleft_cam: CameraCfg = fl_cam_cfg
        frontright_cam: CameraCfg = fr_cam_cfg
        hand_cam: CameraCfg = hand_cam_cfg

    return _SmokeSceneCfg


# ----------------------------------------------------------- physics scaffolding


def reset_robot(scene: InteractiveScene) -> None:
    """Write the default pose/joint state into the sim (copied from teleop)."""
    robot = scene["robot"]
    root_state = robot.data.default_root_state.clone()
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()
    robot.set_joint_position_target(joint_pos)
    robot.set_joint_velocity_target(joint_vel)


def warmup_default_pose(sim: sim_utils.SimulationContext, scene: InteractiveScene, steps: int = 10) -> None:
    """Step physics a few times to let the default pose settle (copied from teleop)."""
    sim_dt = sim.get_physics_dt()
    for _ in range(steps):
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(sim_dt)
    sim.render()


# --------------------------------------------------------- runner + facade wiring


class _RunnerStub:
    """Minimal runner: just needs `_build_robot_state()` for VLFM's needs.

    `IsaacLabRobotStateBackend.get_robot_state()` calls this each time
    `IsaaclabSpot.xy_yaw` is read. VLFM only reads `root_pose_w[:7]`; the
    other `RobotState` fields are filled with sensible defaults so the
    dataclass constructor is satisfied.
    """

    def __init__(self, scene: InteractiveScene) -> None:
        self._scene = scene

    def _build_robot_state(self) -> RobotState:
        data = self._scene["robot"].data
        # IsaacLab `root_state_w` shape: (num_envs, 13) = pos(3) + quat_wxyz(4) + lin_vel_w(3) + ang_vel_w(3).
        # We snapshot env 0 and pack the [pos, quat] into a (7,) tensor matching
        # `vlfm.isaacsim.robots.isaaclab_spot._read_root_pose_w`'s expectation.
        root_state_w = data.root_state_w[0]
        root_pose_w = root_state_w[:7]
        return RobotState(
            joint_pos=data.joint_pos[0],
            joint_vel=data.joint_vel[0],
            joint_names=tuple(data.joint_names),
            root_pose_w=root_pose_w,
            ee_pose_w=root_pose_w,  # placeholder; VLFM does not read this field
        )


def resolve_joint_ids(scene: InteractiveScene) -> Tuple[Sequence[int], Sequence[str], Sequence[int]]:
    """Find arm and gripper joint ids using the same regex pattern as the teleop."""
    arm_ids, arm_names = scene["robot"].find_joints(list(_ARM_JOINT_ORDER), preserve_order=True)
    if tuple(arm_names) != _ARM_JOINT_ORDER:
        raise RuntimeError(f"Unexpected arm joint order: {tuple(arm_names)}")
    gripper_ids, _ = scene["robot"].find_joints([_GRIPPER_JOINT_NAME])
    return arm_ids, arm_names, gripper_ids


def make_robot_facade(
    *,
    cfg: Any,
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
) -> Tuple[IsaaclabSpot, Any]:
    """Bind the SDK facade to the IsaacLab scene and wrap into `IsaaclabSpot`.

    Returns ``(robot, locomotion)`` — both held by the caller so that
    locomotion can be stepped independently if needed.

    This is essentially `vlfm.isaacsim.run_isaacsim_objnav.build_isaacsim_components`
    minus the env + policy construction (those vary per smoke).
    """
    arm_ids, arm_names, gripper_ids = resolve_joint_ids(scene)
    locomotion = SpotLocomotionClient.create(robot=scene["robot"])
    locomotion.stand()

    sdk = create_standard_sdk("vlfm-smoke")
    robot_sdk = sdk.create_robot(name="vlfm-spot")
    runner = _RunnerStub(scene)

    cameras = {
        "frontleft": scene["frontleft_cam"],
        "frontright": scene["frontright_cam"],
        "hand": scene["hand_cam"],
    }

    def _stage_getter_unused() -> Any:
        # The manipulation backend only calls stage_getter inside its planning
        # phase, which fires only if `ManipulationApiClient.manipulation_api_command`
        # is invoked. VLFM never invokes it; if this ever runs, the smoke is
        # exercising a path it shouldn't.
        raise RuntimeError(
            "stage_getter called from a VLFM smoke test — manipulation should "
            "not be active. If you need real manipulation here, pass a real "
            "stage_getter to make_robot_facade()."
        )

    binder = bind_isaaclab_spot_robot(
        robot_sdk,
        runner=runner,
        robot_articulation=scene["robot"],
        scene=scene,
        arm_joint_ids=arm_ids,
        arm_joint_names=arm_names,
        gripper_joint_ids=gripper_ids,
        cameras=cameras,
        sim_dt=float(cfg.isaaclab.sim_dt),
        stage_getter=_stage_getter_unused,
    )

    def _scene_step() -> None:
        # Write any queued joint targets, step physics, then propagate the new
        # physics state back into scene-data buffers so the next read sees it.
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(sim.get_physics_dt())

    robot = IsaaclabSpot(
        robot=robot_sdk,
        locomotion=locomotion,
        binder=binder,
        cameras=cameras,
        scene_step=_scene_step,
        max_body_cam_depth_m=float(cfg.env.max_body_cam_depth),
        max_gripper_cam_depth_m=float(cfg.env.max_gripper_cam_depth),
    )
    return robot, locomotion


# --------------------------------------------------------------- sim + scene init


def build_sim_and_scene(cfg: Any) -> Tuple[sim_utils.SimulationContext, InteractiveScene]:
    """Boot the sim + scene with the locomotion policy's expected control dt.

    Returns the SimulationContext and InteractiveScene, both initialized and
    warmed up so the robot is standing at its default pose.
    """
    sim_dt = SpotLocomotionClient.default_control_dt()
    render_dt = 1.0 / 60.0
    render_interval = max(1, int(round(render_dt / sim_dt)))
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=sim_dt, render_interval=render_interval)
    )

    robot_pos = np.asarray(cfg.robot.position, dtype=np.float32)
    eye_offset = np.asarray(cfg.camera.eye_offset, dtype=np.float32)
    target = robot_pos + np.array([0.0, 0.0, float(cfg.camera.target_height_offset)], dtype=np.float32)
    sim.set_camera_view(eye=(robot_pos + eye_offset).tolist(), target=target.tolist())

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    scene_cfg_cls = make_scene_cfg(cfg_dict)
    scene = InteractiveScene(
        scene_cfg_cls(num_envs=1, env_spacing=float(cfg.scene.env_spacing), replicate_physics=False)
    )

    sim.reset()
    reset_robot(scene)
    scene.update(sim.get_physics_dt())
    warmup_default_pose(sim, scene, steps=10)
    return sim, scene


# ---------------------------------------------------------------- assertion helpers


def assert_close(actual: float, expected: float, tol: float, label: str) -> None:
    """Tiny helper used by smoke 3 to verify rotation deltas are close enough."""
    if not (abs(actual - expected) <= tol):
        raise AssertionError(
            f"{label}: expected {expected:.4f} ± {tol:.4f}, got {actual:.4f} "
            f"(delta={actual - expected:+.4f})"
        )


def torch_to_yaw_deg(yaw_rad: float) -> float:
    """Convenience for printable yaw output."""
    return float(np.rad2deg(yaw_rad))
