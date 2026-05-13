# Skills Package

The `skills` package contains reusable robot APIs and skill implementations for the Interactive Search benchmark.

## Package Layout

| Path | Responsibility |
| --- | --- |
| `base.py` | Base skill interface and camera capture helpers. |
| `core.py` | Shared skill lifecycle types, robot state, and command outputs. |
| `spot/` | Spot-SDK-shaped facade and IsaacLab service binders. |
| `locomotion/` | Spot locomotion policy API and IsaacLab backend. |
| `manipulation/` | AO-Grasp client, point-cloud pipeline, CuRobo integration, and manipulation runner. |
| `navigation/` | Reserved for navigation and search skills. |

## API Boundary

Benchmark scripts should prefer SDK-shaped code:

```python
from skills.spot import create_standard_sdk

sdk = create_standard_sdk("interactive-search")
robot = sdk.create_robot("sim://spot", name="spot-sim")
lease_client = robot.ensure_client("lease")
command_client = robot.ensure_client("robot-command")
manipulation_client = robot.ensure_client("manipulation")
```

Simulation details should be installed through an IsaacLab binder:

```python
from skills.spot import bind_isaaclab_spot_robot

binder = bind_isaaclab_spot_robot(robot, runner=runner, robot_articulation=articulation, scene=scene, ...)
```

## Skill Lifecycle

Skills use the shared `SkillStatus` states:

| State | Meaning |
| --- | --- |
| `idle` | No active command. |
| `planning` | A plan is being created. |
| `executing` | Arm or base command execution is in progress. |
| `grasping` | Gripper close/open phase is in progress. |
| `done` | Command succeeded. |
| `failed` | Command failed and should be surfaced to the benchmark. |

Long-running SDK clients should expose command ids and feedback responses instead of blocking hidden loops.

## Adding a Skill

1. Define request/result dataclasses.
2. Add a small public API surface using existing Spot SDK naming when robot-facing.
3. Keep IsaacLab dependencies in a backend module.
4. Add pure unit tests for API behavior.
5. Add an IsaacLab smoke launcher if simulation behavior is part of the skill.

## Rules

- Fail fast on invalid robot state, missing target data, missing planner output, or missing sensor data.
- Keep public APIs stable for benchmark scripts.
- Avoid broad `try/except` blocks that convert real failures into default behavior.
- Do not make vendor-specific APIs leak into benchmark launchers unless wrapped by a skill or service client.
