# Copyright (c) 2026 M&M Lab. All rights reserved.

"""Hydra entrypoint for running VLFM in IsaacSim.

This module is the IsaacSim analogue of
`vlfm.reality.run_bdsw_objnav_env`. The control loop is byte-identical (and
deliberately so — the policy stack above the env doesn't care which backend it
sits on). The only thing this file owns is the *wiring* between the IsaacLab
scene, the `skills/` adapter layer, and VLFM's env + policy.

**Out-of-scope by plan Section 6**: the IsaacLab scene-launch boilerplate
itself (loading the USD, spawning the Spot articulation, attaching cameras,
implementing the `runner._build_robot_state()` contract). The caller is
expected to follow the pattern from `skills/locomotion/isaaclab_test/` and
pass the resulting handles to `build_isaacsim_components` below. We provide
the wiring helper but not the launch loop — that lives where IsaacLab's
`AppLauncher` is invoked, which is environment-specific.

Typical usage shape (pseudocode):

    from isaaclab.app import AppLauncher
    launcher = AppLauncher({"headless": True})
    # ... build scene, Spot articulation, named Cameras, runner that
    # implements `_build_robot_state()` ...
    components = build_isaacsim_components(
        cfg=hydra_cfg,
        runner=my_runner,
        robot_articulation=spot_articulation,
        scene=scene,
        arm_joint_ids=[...], arm_joint_names=[...], gripper_joint_ids=[...],
        cameras={"frontleft": cam_fl, "frontright": cam_fr, "hand": cam_hand},
        stage_getter=lambda: stage,
        scene_step=lambda: scene.step(render=True),
    )
    run_env(components.env, components.policy, cfg.env.goal)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from skills.locomotion.api import SpotLocomotionClient
from skills.spot.isaaclab_backend import bind_isaaclab_spot_robot
from skills.spot.sdk import create_standard_sdk

from vlfm.isaacsim.objectnav_env import IsaacSimObjectNavEnv
from vlfm.isaacsim.policies import IsaacSimITMPolicyV2
from vlfm.isaacsim.robots.isaaclab_spot import IsaaclabSpot


@dataclass
class IsaacSimComponents:
    """Bundle of wired-up objects ready for the run loop."""

    robot: IsaaclabSpot
    env: IsaacSimObjectNavEnv
    policy: IsaacSimITMPolicyV2


def build_isaacsim_components(
    *,
    cfg: DictConfig,
    runner: Any,
    robot_articulation: Any,
    scene: Any,
    arm_joint_ids: Sequence[int],
    arm_joint_names: Sequence[str],
    gripper_joint_ids: Sequence[int],
    cameras: Mapping[str, Any],
    stage_getter: Callable[[], Any],
    scene_step: Callable[[], None],
) -> IsaacSimComponents:
    """Wire `skills/` + IsaacLab handles into a ready-to-run VLFM stack.

    Args:
        cfg: Loaded Hydra config (`vlfm/isaacsim/config/isaacsim.yaml`).
        runner: An object exposing ``_build_robot_state() -> RobotState`` —
            the `skills.spot.isaaclab_backend.IsaacLabRobotStateBackend` calls
            this every time the env asks for robot state. Typically also owns
            the manipulation state machine for the gripper.
        robot_articulation: The IsaacLab `Articulation` for Spot.
        scene: The IsaacLab `Scene` containing the articulation.
        arm_joint_ids: Index list for the 6 arm joints in
            ``robot_articulation``. Stored by the bound robot-command backend;
            VLFM doesn't drive the arm, but the binder needs it for shape.
        arm_joint_names: Names matching ``arm_joint_ids`` (Spot SDK style).
        gripper_joint_ids: Index list for gripper joints. VLFM doesn't drive
            the gripper either; the manipulation backend pumps it.
        cameras: Mapping ``{"frontleft": Camera, "frontright": Camera,
            "hand": Camera}``. Keys must match
            `vlfm/isaacsim/robots/camera_ids.py`.
        stage_getter: ``lambda: stage`` — passed through to the manipulation
            backend by `bind_isaaclab_spot_robot`.
        scene_step: ``lambda: scene.step(render=True)`` (or equivalent). Pumped
            once per control tick inside `IsaaclabSpot.drive_for_duration`.

    Returns:
        Wired `IsaacSimComponents` with `robot`, `env`, `policy` ready to use.
    """
    sdk = create_standard_sdk("vlfm")
    robot_sdk = sdk.create_robot(name="vlfm-spot")
    binder = bind_isaaclab_spot_robot(
        robot_sdk,
        runner=runner,
        robot_articulation=robot_articulation,
        scene=scene,
        arm_joint_ids=arm_joint_ids,
        arm_joint_names=arm_joint_names,
        gripper_joint_ids=gripper_joint_ids,
        cameras=cameras,
        sim_dt=float(cfg.isaaclab.sim_dt),
        stage_getter=stage_getter,
    )
    locomotion = SpotLocomotionClient.create(robot=robot_articulation)

    robot = IsaaclabSpot(
        robot=robot_sdk,
        locomotion=locomotion,
        binder=binder,
        cameras=cameras,
        scene_step=scene_step,
        max_body_cam_depth_m=float(cfg.env.max_body_cam_depth),
        max_gripper_cam_depth_m=float(cfg.env.max_gripper_cam_depth),
    )
    env = IsaacSimObjectNavEnv(
        robot=robot,
        max_body_cam_depth=float(cfg.env.max_body_cam_depth),
        max_gripper_cam_depth=float(cfg.env.max_gripper_cam_depth),
        max_lin_vel=float(cfg.env.max_lin_vel),
        max_ang_vel=float(cfg.env.max_ang_vel),
        time_step=float(cfg.env.time_step),
        max_steps=int(cfg.env.max_steps),
    )
    policy = IsaacSimITMPolicyV2.from_config(cfg)
    return IsaacSimComponents(robot=robot, env=env, policy=policy)


def run_env(env: IsaacSimObjectNavEnv, policy: IsaacSimITMPolicyV2, goal: str) -> None:
    """Drive one episode to completion.

    Mirrors `vlfm.reality.run_bdsw_objnav_env.run_env` exactly. The mask=0
    initial call triggers the policy's `_reset()` via `_pre_step`, which in
    turn resets the body-yaw sweep state inside `IsaacSimMixin`.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    observations = env.reset(goal)
    done = False
    mask = torch.zeros(1, 1, device=device, dtype=torch.bool)
    st = time.time()
    action = policy.get_action(observations, mask)
    print(f"get_action took {time.time() - st:.2f} seconds")
    while not done:
        observations, _, done, _ = env.step(action)
        st = time.time()
        action = policy.get_action(observations, mask, deterministic=True)
        print(f"get_action took {time.time() - st:.2f} seconds")
        mask = torch.ones_like(mask)
        if done:
            print("Episode finished because done is True")
            break


@hydra.main(version_base=None, config_path="config", config_name="isaacsim")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint stub.

    The scene-launch boilerplate is out of scope for this module (see the
    file-level docstring). A working invocation looks like:

        from vlfm.isaacsim.run_isaacsim_objnav import build_isaacsim_components, run_env
        # ... start IsaacLab, build scene/articulation/cameras/runner ...
        components = build_isaacsim_components(cfg=cfg, runner=..., ...)
        run_env(components.env, components.policy, cfg.env.goal)
    """
    print(OmegaConf.to_yaml(cfg))
    raise NotImplementedError(
        "IsaacLab scene-launch boilerplate is out of scope for this entrypoint. "
        "Construct your scene/articulation/cameras following the "
        "`skills/locomotion/isaaclab_test/` harness, then call "
        "`build_isaacsim_components(...)` and `run_env(...)` from your launch script."
    )


if __name__ == "__main__":
    main()
