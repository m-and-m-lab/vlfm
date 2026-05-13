#!/usr/bin/env python3
# Copyright (c) 2026 M&M Lab. All rights reserved.

"""Smoke test 2: camera parity between `IsaaclabSpot` and `BDSWRobot`.

Runs a one-shot capture: build a minimal scene with Spot + 3 cameras, wire up
`IsaaclabSpot`, call `get_camera_data(["frontleft", "frontright", "hand"])`,
and assert the returned dict has the exact shape `vlfm.reality.objectnav_env`
relies on:

    {src: {"image": np.ndarray, "fx": float, "fy": float,
           "tf_camera_to_global": np.ndarray (4, 4)}}

The depth cams must hand back uint16 millimeters and the hand cam uint8 BGR —
this is what unlocks the env's existing dtype-dispatched normalization
branches.

Run with:
    ${ISAACSIM_PYTHON} vlfm/isaacsim/isaaclab_test/smoke_camera_parity.py --headless
where ISAACSIM_PYTHON is your IsaacLab-bundled Python launcher.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the repository root is importable so `vlfm.*` and `skills.*` resolve
# when this script is invoked directly. Mirrors the path-juggling in
# `skills/locomotion/isaaclab_test/spot_locomotion_wasd.py`.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from isaaclab.app import AppLauncher  # must be the very first isaaclab import

_DEFAULT_CONFIG_PATH = Path(__file__).with_name("smoke_scene.yaml")

parser = argparse.ArgumentParser(description="VLFM smoke 2: camera-data parity.")
parser.add_argument("--config", type=str, default=str(_DEFAULT_CONFIG_PATH))
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# All isaaclab.* imports MUST come after AppLauncher fires. The helper module
# pulls in isaaclab internally, so it can only be imported here.
import numpy as np  # noqa: E402

from vlfm.isaacsim.isaaclab_test._scene import (  # noqa: E402
    build_sim_and_scene,
    make_omegaconf_cfg,
    make_robot_facade,
    read_yaml,
)

_EXPECTED_CAMS = ["frontleft", "frontright", "hand"]
_EXPECTED_KEYS = {"image", "fx", "fy", "tf_camera_to_global"}


def main() -> int:
    cfg = make_omegaconf_cfg(read_yaml(args_cli.config))
    sim, scene = build_sim_and_scene(cfg)
    robot, _locomotion = make_robot_facade(cfg=cfg, sim=sim, scene=scene)

    # Step physics once so the cameras have rendered at least one frame.
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(sim.get_physics_dt())

    print(f"[smoke2] calling get_camera_data({_EXPECTED_CAMS}) ...")
    cam_data = robot.get_camera_data(_EXPECTED_CAMS)

    errors: list[str] = []

    if set(cam_data.keys()) != set(_EXPECTED_CAMS):
        errors.append(
            f"keys mismatch: got {sorted(cam_data)} expected {_EXPECTED_CAMS}"
        )

    for src in _EXPECTED_CAMS:
        entry = cam_data.get(src)
        if entry is None:
            errors.append(f"{src}: missing entry")
            continue

        missing = _EXPECTED_KEYS - set(entry.keys())
        if missing:
            errors.append(f"{src}: missing keys {missing}")
            continue

        image = entry["image"]
        if not isinstance(image, np.ndarray):
            errors.append(f"{src}.image: not an ndarray ({type(image).__name__})")
        else:
            # Depth cams should be uint16 (mm); hand should be uint8 BGR.
            if src == "hand":
                if image.dtype != np.uint8:
                    errors.append(f"hand.image: dtype expected uint8, got {image.dtype}")
                if image.ndim != 3 or image.shape[2] != 3:
                    errors.append(f"hand.image: expected HxWx3, got shape {image.shape}")
            else:
                if image.dtype != np.uint16:
                    errors.append(f"{src}.image: dtype expected uint16, got {image.dtype}")
                if image.ndim != 2:
                    errors.append(f"{src}.image: expected HxW, got shape {image.shape}")

        for k in ("fx", "fy"):
            v = entry[k]
            if not isinstance(v, float):
                errors.append(f"{src}.{k}: expected float, got {type(v).__name__}")
            elif v <= 0.0:
                errors.append(f"{src}.{k}: expected positive, got {v}")

        tf = entry["tf_camera_to_global"]
        if not isinstance(tf, np.ndarray) or tf.shape != (4, 4):
            errors.append(
                f"{src}.tf_camera_to_global: expected ndarray(4,4), got "
                f"{type(tf).__name__} shape={getattr(tf, 'shape', None)}"
            )
        else:
            # Bottom row must be [0, 0, 0, 1] for a well-formed homogeneous transform.
            if not np.allclose(tf[3], np.array([0.0, 0.0, 0.0, 1.0])):
                errors.append(
                    f"{src}.tf_camera_to_global: bottom row not [0,0,0,1], got {tf[3]}"
                )

        # Print a quick summary for the operator running the smoke.
        if isinstance(image, np.ndarray):
            print(
                f"[smoke2] {src}: image shape={image.shape} dtype={image.dtype} "
                f"fx={entry['fx']:.1f} fy={entry['fy']:.1f}"
            )

    if errors:
        print("\n[smoke2] FAIL — the following assertions failed:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\n[smoke2] PASS — camera-data shape matches BDSWRobot contract.")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    finally:
        simulation_app.close()
    sys.exit(rc)
