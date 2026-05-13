#!/usr/bin/env python3
# Copyright (c) 2026 M&M Lab. All rights reserved.

"""Smoke test 3: end-to-end env loop with a no-op stub policy.

Builds a minimal IsaacLab scene + `IsaaclabSpot` + `IsaacSimObjectNavEnv`, then
runs the env through one synthetic episode using a stub policy that emits the
eight body-yaw init targets followed by a zero-velocity stop. Asserts the
observable env behavior:

  (a) Each init-phase step changes the body's episodic yaw by ~45° (the spacing
      of `INITIAL_BODY_YAWS`).
  (b) Each `_get_obs()` return has exactly the seven keys VLFM's policy reads,
      with the documented dtypes/shapes.
  (c) The first all-zero post-init action returns ``done=True``.

This test deliberately uses a stub policy (not `IsaacSimITMPolicyV2`) to keep
the smoke cheap: the real policy instantiates BLIP-2 ITM, GroundingDINO,
MobileSAM and YOLOv7 clients and expects the four REST servers from
`scripts/launch_vlm_servers.sh` to be up. Those are out of scope here — what
this smoke verifies is the sensor/action plumbing, not the policy.

Run with:
    ${ISAACSIM_PYTHON} vlfm/isaacsim/isaaclab_test/smoke_env_loop.py --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

# Make repo root importable for direct script invocation (see the camera-parity
# smoke for the rationale).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from isaaclab.app import AppLauncher  # noqa: E402 — must precede other isaaclab imports

_DEFAULT_CONFIG_PATH = Path(__file__).with_name("smoke_scene.yaml")

parser = argparse.ArgumentParser(description="VLFM smoke 3: env loop with no-op policy.")
parser.add_argument("--config", type=str, default=str(_DEFAULT_CONFIG_PATH))
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Post-AppLauncher imports.
import numpy as np  # noqa: E402

from vlfm.isaacsim.isaaclab_test._scene import (  # noqa: E402
    assert_close,
    build_sim_and_scene,
    make_omegaconf_cfg,
    make_robot_facade,
    read_yaml,
    torch_to_yaw_deg,
)
from vlfm.isaacsim.objectnav_env import IsaacSimObjectNavEnv  # noqa: E402
from vlfm.isaacsim.policies import INITIAL_BODY_YAWS  # noqa: E402
from vlfm.utils.geometry_utils import wrap_heading  # noqa: E402


class _NoOpStubPolicy:
    """Minimal stand-in for `IsaacSimITMPolicyV2`.

    Emits the eight `INITIAL_BODY_YAWS` body-yaw targets one per call, then
    falls into a permanent zero-velocity stop. Mirrors the action-dict shape
    produced by `IsaacSimMixin.act` so the env code can't tell the difference.
    """

    def __init__(self) -> None:
        self._remaining = list(INITIAL_BODY_YAWS)
        self.calls = 0

    def get_action(self, observations: Dict[str, Any], masks: Any, deterministic: bool = True) -> Dict[str, Any]:
        self.calls += 1
        if self._remaining:
            yaw = self._remaining.pop(0)
            return {"angular": 0, "linear": 0, "arm_yaw": float(yaw), "info": {}}
        return {"angular": 0.0, "linear": 0.0, "arm_yaw": -1, "info": {}}


def _check_obs_shape(obs: Dict[str, Any]) -> List[str]:
    """Return a list of assertion failure strings (empty if obs is well-formed)."""
    errors: List[str] = []
    expected_keys = {
        "nav_depth", "robot_xy", "robot_heading", "objectgoal",
        "obstacle_map_depths", "value_map_rgbd", "object_map_rgbd",
    }
    missing = expected_keys - set(obs.keys())
    if missing:
        errors.append(f"obs keys missing: {missing}")

    nd = obs.get("nav_depth")
    if not isinstance(nd, np.ndarray) or nd.ndim != 2:
        errors.append(f"nav_depth: expected HxW ndarray, got {type(nd).__name__} {getattr(nd, 'shape', None)}")

    xy = obs.get("robot_xy")
    if not isinstance(xy, np.ndarray) or xy.shape != (2,):
        errors.append(f"robot_xy: expected (2,) ndarray, got shape {getattr(xy, 'shape', None)}")

    hdg = obs.get("robot_heading")
    if not isinstance(hdg, float):
        errors.append(f"robot_heading: expected float, got {type(hdg).__name__}")

    goal = obs.get("objectgoal")
    if not isinstance(goal, str):
        errors.append(f"objectgoal: expected str, got {type(goal).__name__}")

    omd = obs.get("obstacle_map_depths")
    # 2 point-cloud cams + 1 terminal explore-only entry = 3 tuples.
    if not isinstance(omd, list) or len(omd) != 3:
        errors.append(f"obstacle_map_depths: expected len-3 list, got len={len(omd) if hasattr(omd, '__len__') else 'n/a'}")

    vmr = obs.get("value_map_rgbd")
    if not isinstance(vmr, list) or len(vmr) != 1:
        errors.append(f"value_map_rgbd: expected len-1 list, got len={len(vmr) if hasattr(vmr, '__len__') else 'n/a'}")

    omrgbd = obs.get("object_map_rgbd")
    if not isinstance(omrgbd, list) or len(omrgbd) != 1:
        errors.append(f"object_map_rgbd: expected len-1 list, got len={len(omrgbd) if hasattr(omrgbd, '__len__') else 'n/a'}")

    return errors


def main() -> int:
    cfg = make_omegaconf_cfg(read_yaml(args_cli.config))
    sim, scene = build_sim_and_scene(cfg)
    robot, _locomotion = make_robot_facade(cfg=cfg, sim=sim, scene=scene)

    env = IsaacSimObjectNavEnv(
        robot=robot,
        max_body_cam_depth=float(cfg.env.max_body_cam_depth),
        max_gripper_cam_depth=float(cfg.env.max_gripper_cam_depth),
        max_lin_vel=float(cfg.env.max_lin_vel),
        max_ang_vel=float(cfg.env.max_ang_vel),
        time_step=float(cfg.env.time_step),
        max_steps=int(cfg.env.max_steps),
    )
    policy = _NoOpStubPolicy()

    errors: List[str] = []

    print("[smoke3] env.reset('office chair') ...")
    obs = env.reset("office chair")
    errors.extend(_check_obs_shape(obs))
    print(f"[smoke3] initial yaw: {torch_to_yaw_deg(obs['robot_heading']):+.2f}°")

    # ----- (a) eight init-phase rotations, each ~45° from the prior --------
    yaw_after_each_init: List[float] = [obs["robot_heading"]]
    for i in range(len(INITIAL_BODY_YAWS)):
        action = policy.get_action(obs, masks=None)
        if action["arm_yaw"] == -1:
            errors.append(f"init step {i}: policy returned arm_yaw=-1 (expected sweep target)")
            break
        obs, _, done, _ = env.step(action)
        yaw_after_each_init.append(obs["robot_heading"])
        print(
            f"[smoke3] init step {i}: target={np.rad2deg(INITIAL_BODY_YAWS[i]):+.1f}° "
            f"-> yaw now {torch_to_yaw_deg(obs['robot_heading']):+.2f}°  done={done}"
        )
        errors.extend(_check_obs_shape(obs))
        if done:
            errors.append(f"init step {i}: env returned done=True before sweep completed")
            break

    # Compare to expected per-step yaw delta (≈45° = π/4 rad).
    expected_delta_per_step = float(np.deg2rad(45.0))
    # IsaacLab's policy won't hit the target exactly in one drive segment; the
    # locomotion loop runs for time_step seconds with max_ang_vel rad/s and
    # stops, so we expect *up to* expected_delta_per_step of rotation per step.
    # Use a permissive tolerance — this assertion is a sanity check, not a
    # precision claim.
    rotation_tol = float(np.deg2rad(15.0))
    for i in range(1, len(yaw_after_each_init)):
        delta = wrap_heading(yaw_after_each_init[i] - yaw_after_each_init[i - 1])
        try:
            assert_close(delta, expected_delta_per_step, rotation_tol,
                         label=f"yaw delta after init step {i - 1}")
        except AssertionError as e:
            errors.append(str(e))

    # ----- (c) one more get_action — should be all-zero, env returns done ---
    print("[smoke3] post-init: emitting zero action ...")
    action = policy.get_action(obs, masks=None)
    if action["arm_yaw"] != -1:
        errors.append(f"post-init policy: expected arm_yaw=-1, got {action['arm_yaw']}")
    if not (action["linear"] == 0.0 and action["angular"] == 0.0):
        errors.append(
            f"post-init policy: expected zero lin/ang, got "
            f"linear={action['linear']} angular={action['angular']}"
        )
    obs, _, done, _ = env.step(action)
    errors.extend(_check_obs_shape(obs))
    if not done:
        errors.append("env.step on zero action: expected done=True")
    print(f"[smoke3] zero-action step: done={done}")

    if errors:
        print("\n[smoke3] FAIL — the following assertions failed:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\n[smoke3] PASS — env loop, body-yaw sweep, and stop convention all behaved as expected.")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    finally:
        simulation_app.close()
    sys.exit(rc)
