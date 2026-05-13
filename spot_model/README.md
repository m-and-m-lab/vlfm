# Spot Model Assets

This directory contains Spot robot assets used by the Interactive Search benchmark.

## Contents

| Path | Purpose |
| --- | --- |
| `spot_arm_w_cam.usd` | Spot arm USD with mounted cameras for general manipulation and perception. |
| `spot_arm_w_cam_static.usd` | Fixed-base Spot arm USD used by manipulation smoke tests. |
| `spot_arm_only.usd` | Arm-focused USD asset. |
| `spot.urdf` | URDF source/bridge asset for Spot. |
| `spot_arm_only.urdf` | Arm-only URDF source/bridge asset. |
| `configuration/spot_arm_curobo.yaml` | CuRobo robot configuration used by the manipulation planner. |
| `configuration/*.usd` | Robot base, physics, sensor, and fixed-root component layers. |
| `configuration/whole_spot.yaml` | Whole robot configuration metadata. |
| `spot_description/` | ROS-style Spot description submodule and generated build/install artifacts. |

## Camera Paths

Default robot-mounted depth cameras:

```text
{ENV_REGEX_NS}/Robot/body/Gemini2_front_left/Orbbec_Gemini2/camera_ir_left/camera_left/Stream_depth
{ENV_REGEX_NS}/Robot/body/Gemini2_front_right/Orbbec_Gemini2/camera_ir_left/camera_left/Stream_depth
{ENV_REGEX_NS}/Robot/arm_link_wr1/Gemini2_arm/Orbbec_Gemini2/camera_ir_left/camera_left/Stream_depth
```

## Joint Names

Default arm joints:

```text
arm_sh0
arm_sh1
arm_el0
arm_el1
arm_wr0
arm_wr1
```

Default gripper joint:

```text
arm_f1x
```

Default end-effector body:

```text
arm_link_fngr
```

## Usage

<!-- General manipulation:

```bash
./isaaclab.sh -p scripts/interactive-search/scripts/interactive_search.py \
  --robot_usd scripts/interactive-search/spot_model/spot_arm_w_cam.usd \
  --enable_cameras
```

Fixed-base drawer manipulation smoke:

```bash
./isaaclab.sh -p scripts/interactive-search/tests/isaaclab_test/spot_manipulation_drawer.py --enable_cameras
``` -->

## Rules

- Keep CuRobo joint names synchronized with IsaacLab articulation joint names.
- Do not rename camera prims without updating smoke-test YAML configs.
- Use fixed-root assets for arm-only manipulation tests.
- Use movable-root assets for locomotion and navigation tests.
