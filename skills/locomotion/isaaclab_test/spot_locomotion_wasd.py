#!/usr/bin/env python3
"""Standalone Isaac Lab Spot locomotion teleop."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
_PROJECT_SCRIPTS_DIR = Path(__file__).resolve().parents[3]
if str(_PROJECT_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_SCRIPTS_DIR))

from isaaclab.app import AppLauncher

_DEFAULT_CONFIG_PATH = Path(__file__).with_suffix(".yaml")
_ARM_POSE_LIBRARY_PATH = Path(__file__).with_name("spot_arm_pose_library.npz")
_ARM_POSE_JOINT_ORDER = ("arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1")
_ARM_POSE_PERIOD_SECONDS = 10.0

parser = argparse.ArgumentParser(description="Spot locomotion teleop with InteractiveScene.")
parser.add_argument("--config", type=str, default=str(_DEFAULT_CONFIG_PATH))
AppLauncher.add_app_launcher_args(parser)
# Default to CPU for this single-robot teleop harness: at 500 Hz the workload is too small to
# amortize GPU launch/sync overhead, so CPU usually reaches a much better real-time factor.
# Pass `--device cuda:0` when you want to validate the full GPU integration path instead.
parser.set_defaults(device="cpu")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import numpy as np
import omni.appwindow

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg, RemotizedPDActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.spot import joint_parameter_lookup

from skills.locomotion import SpotLocomotionClient


def _read_yaml(path: str | Path, *, loader: type[yaml.Loader] = yaml.SafeLoader) -> dict:
    data = yaml.load(Path(path).read_text(), Loader=loader)
    return {} if data is None else data


def _load_arm_pose_library(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        joint_order = tuple(np.asarray(data["joint_order"]).astype(np.str_).tolist())
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)

    if joint_order != _ARM_POSE_JOINT_ORDER:
        raise ValueError(f"Unexpected arm-pose joint order in {path}: {joint_order}")
    if joint_pos.ndim != 2 or joint_pos.shape[1] != len(_ARM_POSE_JOINT_ORDER):
        raise ValueError(f"Unexpected arm-pose library shape in {path}: {joint_pos.shape}")
    return joint_pos


class TeleopKeyboard:
    """Tiny keyboard helper that keeps a base-velocity command alive while keys are held."""

    def __init__(self, linear_speed: float, strafe_speed: float, yaw_speed: float) -> None:
        self._command = np.zeros(3, dtype=np.float32)
        self._mapping = {
            "NUMPAD_8": np.array([linear_speed, 0.0, 0.0], dtype=np.float32),
            "UP": np.array([linear_speed, 0.0, 0.0], dtype=np.float32),
            "NUMPAD_2": np.array([-linear_speed, 0.0, 0.0], dtype=np.float32),
            "DOWN": np.array([-linear_speed, 0.0, 0.0], dtype=np.float32),
            "NUMPAD_6": np.array([0.0, -strafe_speed, 0.0], dtype=np.float32),
            "RIGHT": np.array([0.0, -strafe_speed, 0.0], dtype=np.float32),
            "NUMPAD_4": np.array([0.0, strafe_speed, 0.0], dtype=np.float32),
            "LEFT": np.array([0.0, strafe_speed, 0.0], dtype=np.float32),
            "NUMPAD_7": np.array([0.0, 0.0, yaw_speed], dtype=np.float32),
            "N": np.array([0.0, 0.0, yaw_speed], dtype=np.float32),
            "NUMPAD_9": np.array([0.0, 0.0, -yaw_speed], dtype=np.float32),
            "M": np.array([0.0, 0.0, -yaw_speed], dtype=np.float32),
        }
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._subscription = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_event)

    def __del__(self) -> None:
        if hasattr(self, "_subscription") and self._subscription is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._subscription)
            self._subscription = None

    @property
    def command(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self._command)

    def _on_event(self, event, *args, **kwargs) -> bool:
        del args, kwargs
        key_name = event.input if isinstance(event.input, str) else event.input.name
        if key_name not in self._mapping:
            return True
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._command += self._mapping[key_name]
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._command -= self._mapping[key_name]
        return True


def _make_scene_cfg(config: dict) -> type[InteractiveSceneCfg]:
    robot_cfg = config["robot"]
    scene_cfg = config["scene"]
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
                "arm_sh0": 0.0,
                "arm_sh1": -3.13,
                "arm_el0": 3.13,
                "arm_el1": 0.0,
                "arm_wr0": 0.0,
                "arm_wr1": 0.0,
                "arm_f1x": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "spot_hip": DelayedPDActuatorCfg(
                joint_names_expr=[".*_hip_[xy]"],
                effort_limit=45.0,
                stiffness=60.0,
                damping=1.5,
                min_delay=0,
                max_delay=4,
            ),
            "spot_knee": RemotizedPDActuatorCfg(
                joint_names_expr=[".*_knee"],
                joint_parameter_lookup=joint_parameter_lookup,
                effort_limit=None,
                stiffness=60.0,
                damping=1.5,
                min_delay=0,
                max_delay=4,
            ),
            "arm_hold": ImplicitActuatorCfg(
                joint_names_expr=["arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1"],
                effort_limit_sim=300.0,
                stiffness=400.0,
                damping=60.0,
            ),
            "gripper_hold": ImplicitActuatorCfg(
                joint_names_expr=["arm_f1x"],
                effort_limit_sim=200.0,
                stiffness=120.0,
                damping=20.0,
            ),
        },
    )

    @configclass
    class SpotLocomotionSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
        light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(
                intensity=float(scene_cfg["dome_light_intensity"]),
                color=tuple(map(float, scene_cfg["dome_light_color"])),
            ),
        )
        robot: ArticulationCfg = robot_articulation_cfg

    return SpotLocomotionSceneCfg


def _reset_robot(scene: InteractiveScene) -> None:
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


def _warmup_default_pose(sim: sim_utils.SimulationContext, scene: InteractiveScene, steps: int = 10) -> None:
    sim_dt = sim.get_physics_dt()
    for _ in range(steps):
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(sim_dt)
    sim.render()


def main() -> None:
    config_path = Path(args_cli.config).resolve()
    config = _read_yaml(config_path)
    robot_usd_path = Path(config["robot"]["usd_path"])
    if not robot_usd_path.is_absolute():
        config["robot"]["usd_path"] = str((config_path.parent / robot_usd_path).resolve())

    sim_dt = SpotLocomotionClient.default_control_dt()
    render_dt = 1.0 / 60.0
    render_interval = max(1, int(round(render_dt / sim_dt)))
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=sim_dt, render_interval=render_interval, device=args_cli.device)
    )

    robot_pos = np.asarray(config["robot"]["position"], dtype=np.float32)
    eye_offset = np.asarray(config["camera"]["eye_offset"], dtype=np.float32)
    target = robot_pos + np.array([0.0, 0.0, float(config["camera"]["target_height_offset"])], dtype=np.float32)
    sim.set_camera_view(eye=(robot_pos + eye_offset).tolist(), target=target.tolist())

    scene = InteractiveScene(
        _make_scene_cfg(config)(
            num_envs=1,
            env_spacing=float(config["scene"]["env_spacing"]),
            replicate_physics=False,
        )
    )

    sim.reset()
    _reset_robot(scene)
    scene.update(sim.get_physics_dt())

    locomotion = SpotLocomotionClient.create(
        robot=scene["robot"],
    )
    locomotion.stand()
    scene.write_data_to_sim()
    scene.update(sim.get_physics_dt())
    _warmup_default_pose(sim, scene, steps=10)

    keyboard = TeleopKeyboard(
        linear_speed=float(config["teleop"]["linear_speed"]),
        strafe_speed=float(config["teleop"]["strafe_speed"]),
        yaw_speed=float(config["teleop"]["yaw_speed"]),
    )
    arm_pose_library = _load_arm_pose_library(_ARM_POSE_LIBRARY_PATH)
    arm_joint_ids, arm_joint_names = scene["robot"].find_joints(list(_ARM_POSE_JOINT_ORDER), preserve_order=True)
    if tuple(arm_joint_names) != _ARM_POSE_JOINT_ORDER:
        raise RuntimeError(f"Unexpected arm joints: {tuple(arm_joint_names)}")

    rng = np.random.default_rng()
    arm_pose_elapsed = 0.0
    arm_target = scene["robot"].data.default_joint_pos[:, arm_joint_ids].clone()

    def _sample_arm_pose() -> None:
        pose = arm_pose_library[int(rng.integers(arm_pose_library.shape[0]))]
        arm_target[0] = arm_target.new_tensor(pose)

    print("[INFO] Controls: Up/Down forward-back, Left/Right strafe, N/M yaw.")
    print("[INFO] Alternate controls: NUMPAD 8/2 translation, 4/6 strafe, 7/9 yaw.")

    def _on_physics_step(step_size: float) -> None:
        nonlocal arm_pose_elapsed
        scene.update(step_size)
        locomotion.command_velocity(*keyboard.command)
        locomotion.step()
        arm_pose_elapsed += step_size
        if arm_pose_elapsed >= _ARM_POSE_PERIOD_SECONDS:
            arm_pose_elapsed %= _ARM_POSE_PERIOD_SECONDS
            _sample_arm_pose()
        scene["robot"].set_joint_position_target(arm_target, joint_ids=arm_joint_ids)
        scene.write_data_to_sim()

    sim.add_physics_callback("spot_forward", _on_physics_step)

    while simulation_app.is_running():
        sim.step(render=True)


if __name__ == "__main__":
    main()
    simulation_app.close()