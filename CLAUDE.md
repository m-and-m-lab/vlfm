# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VLFM (Vision-Language Frontier Maps) is a zero-shot semantic navigation system. Given a target object category, it builds an occupancy map from depth, identifies frontiers (boundaries between explored/unexplored space), and uses a BLIP-2 image-text matching model on RGB to score frontiers — picking the most promising one to explore next via a pretrained PointNav policy.

This is the **M&M Lab fork** of [bdaiinstitute/vlfm](https://github.com/bdaiinstitute/vlfm), updated for Python 3.11 compatibility and IsaacSim/IsaacLab integration. The `spot_model/` directory (USD/URDF assets, CuRobo configs) exists only in this fork and supports the Interactive Search benchmark on the Spot robot in IsaacSim — it is not part of upstream VLFM.

## Environment & Setup

- Use **mamba** (or conda), not bare pip. The env is named `vlfm`.
- `setup.sh` and `environment_setup.sh` are **two different setup flows** — they are not equivalent:
  - `setup.sh`: original upstream flow targeting Python 3.9, CUDA 11.7, PyTorch 1.13.1, habitat-sim 0.2.5. Use for Habitat evaluation.
  - `environment_setup.sh` + `environment.yml`: the M&M Lab Python-3.11 / CUDA 12.8 flow. Use for IsaacSim / Spot work. This expects the conda env to already be created from `environment.yml` and then activated.
- `pyproject.toml2` is intentionally **not** named `pyproject.toml` (note the trailing `2`). The build is driven by `setup.py` (find_packages on `vlfm/`). Don't rename or "fix" this — the renamed file disables PEP 517 build hooks while preserving the tooling config (ruff/black/mypy) for reference. If you need tool configs to be picked up, pass `--config pyproject.toml2` explicitly (e.g., the pre-commit hooks point at `pyproject.toml`, which is a known mismatch in this fork).
- The `pip install -e .` line in `environment_setup.sh` installs `vlfm[reality]` — this requires `spot_wrapper` and `bosdyn-*` packages.
- Required model weights (must exist before running anything):
  - `data/groundingdino_swint_ogc.pth`
  - `data/yolov7-e6e.pt`
  - `data/mobile_sam.pt`
  - `data/pointnav_weights.pth` (Habitat) / `data/spot_pointnav_weights.pth` (reality)
  - `data/dummy_policy.pth` — generated via `python -m vlfm.utils.generate_dummy_policy`; `vlfm/run.py` hard-asserts its presence.

## Common Commands

```bash
# Start the four VLM REST servers (GroundingDINO, BLIP2-ITM, MobileSAM, YOLOv7).
# Required before ANY policy run — policies are HTTP clients to these servers.
./scripts/launch_vlm_servers.sh           # creates a tmux session; wait ~90s for weights to load

# Evaluate VLFM (BLIP-2 ITM policy) in Habitat on HM3D
./scripts/eval_itm_policy.sh

# Evaluate dummy / oracle FBE baselines
./scripts/eval_dummy_policy.sh
./scripts/eval_oracle_fbe_policy.sh
# NOTE: eval_dummy_policy.sh and eval_oracle_fbe_policy.sh still invoke `zsos.run`
# (the old package name). Use `vlfm.run` instead, or fix the script before running.

# Run the core Habitat eval entrypoint directly
python -m vlfm.run                                                # uses config/experiments/vlfm_objectnav_hm3d.yaml
python -m vlfm.run habitat.dataset.data_path=data/datasets/objectnav/mp3d/val/val.json.gz   # override via Hydra

# Real-world Spot deployment (requires [reality] extras + a Spot on the network)
python -m vlfm.reality.run_bdsw_objnav_env   # uses config/experiments/reality.yaml

# Tests
pytest                                # full suite (matches CI in .github/workflows/test.yml)
pytest test/test_setup.py             # single file
pytest test/test_visualization.py::test_visualization   # single test
```

The VLM servers are configured via env vars (`GROUNDING_DINO_PORT=12181`, `BLIP2ITM_PORT=12182`, `SAM_PORT=12183`, `YOLOV7_PORT=12184`). Policies read these env vars to construct HTTP clients — keep them in sync if you change the ports.

## Linting / Type Checking

Tool configs live in `pyproject.toml2` (see note above). Conventions:
- **Black**: line length 120, target py39
- **Ruff**: rules `E`, `F`, `I`; line length 120; `__init__.py` ignores `F401`
- **mypy**: strict typing (`disallow_untyped_defs`, `check_untyped_defs`, `strict_equality`), checks `vlfm/`, `test/`, `scripts/`
- **Pre-commit**: ruff, black, mypy, clang-format, cpplint, plus a `forbid-binary` hook and a 200KB max-file-size enforcement (binary additions to `data/` will be rejected by default — extend the exclude list if intentional).

CI is two pipelines: `.github/workflows/test.yml` (pytest inside the `ghcr.io/bdaiinstitute/bdaii_vlfm:main` Docker image with `pip install -e .[habitat]`) and `.github/workflows/pre_commit.yml` (pre-commit on Python 3.9.16).

## Architecture

The policy is a layered system. Bottom-up:

1. **VLM servers** (`vlfm/vlm/`) — independent Flask processes, each loading one model. `server_wrapper.py` provides `host_model()` and the bool-array codec used to ship masks over HTTP. The matching `*Client` classes (e.g., `GroundingDINOClient`, `BLIP2ITM`, `MobileSAMClient`, `YOLOv7Client`) are the in-process consumers used by policies. **Always start these before running a policy.**

2. **Mapping** (`vlfm/mapping/`):
   - `obstacle_map.py` — occupancy grid from depth + camera pose, with explored-area tracking.
   - `frontier_map.py` — frontier extraction (depends on `frontier_exploration` git dep).
   - `value_map.py` — per-frontier value scores from the BLIP-2 ITM head. `_vis_reduce_fn` controls how multi-channel prompt scores collapse for visualization.
   - `object_point_cloud_map.py` — 3D point cloud of detected target objects, used to decide when to stop ("we've localized the object well enough").
   - `traj_visualizer.py` — overlay drawing on top of the maps.
   - All maps inherit shared pose/projection logic from `base_map.py`.

3. **Policies** (`vlfm/policy/`):
   - `base_objectnav_policy.BaseObjectNavPolicy` — the abstract ObjectNav loop: detect object → update maps → decide explore-vs-pointnav-to-object → emit action. Owns the VLM clients and the pointnav policy.
   - `base_policy.BasePolicy` — Habitat baseline-registry stub used only to load weights (just goes forward).
   - `itm_policy.BaseITMPolicy` — adds the `ValueMap` + acyclic-frontier-selection on top of `BaseObjectNavPolicy`.
   - `habitat_policies.py` registers Habitat-specific subclasses (`HabitatITMPolicyV2`, etc.) with `baseline_registry`. `reality_policies.py` registers the real-Spot equivalents.
   - `utils/pointnav_policy.py` — wraps the pretrained PointNav ResNet policy used to actually drive to a chosen frontier or target.
   - `utils/acyclic_enforcer.py` — prevents the agent from oscillating between two frontiers.
   - `action_replay_policy.py` — replays recorded action sequences (debug/repro tool).

4. **Entrypoints** — three parallel deployment paths share the policy layer but each has its own environment glue:
   - `vlfm/run.py` — Habitat (HM3D/MP3D ObjectNav). Hydra-driven, requires `data/dummy_policy.pth`. Habitat-only imports happen here, not in `vlfm/__init__.py`, so `import vlfm` works in habitat-free envs.
   - `vlfm/reality/run_bdsw_objnav_env.py` — real Spot via `spot_wrapper` + `bosdyn-*`. Uses `ObjectNavEnv` (`vlfm/reality/objectnav_env.py`) which wraps `BDSWRobot`/`PointNavEnv` and the gripper/body cameras.
   - `vlfm/semexp_env/eval.py` — Semantic Exploration benchmark (Chaplot et al.). Imports `arguments`, `envs`, `moviepy` from outside this repo; depends on the SemExp codebase being on `PYTHONPATH`.

5. **Hydra config flow**:
   - Configs live in `config/experiments/` (top-level experiment configs) and `config/tasks/` (task-level overrides).
   - `vlfm/run.py` registers a `HabitatConfigPlugin` so configs can reference `habitat`-prefixed defaults.
   - The `defaults:` list in `vlfm_objectnav_hm3d.yaml` pulls in: the `objectnav_hm3d` benchmark, `base_explorer`/`frontier_sensor`/etc. lab sensors, the `frontier_exploration_map` + `traveled_stairs` measurements, and the `vlfm_policy` policy config.
   - The `vlfm` trainer (`vlfm/utils/vlfm_trainer.py`) is registered via `baseline_registry` and selected with `habitat_baselines.trainer_name: "vlfm"`.
   - Most CLI usage overrides Hydra keys directly: `habitat_baselines.evaluate=True habitat_baselines.eval.split=val_50 ...`.

6. **Spot model assets** (`spot_model/`, M&M Lab fork only) — USD/URDF files, CuRobo config, and joint/camera prim path conventions for IsaacSim. See [spot_model/README.md](spot_model/README.md) for camera paths and joint names. Rules: keep CuRobo joint names in sync with IsaacLab articulation joint names; do not rename camera prims without updating dependent smoke-test YAMLs.

7. **Skills package** (`skills/`, M&M Lab fork only) — reusable robot APIs and skill implementations for the Interactive Search benchmark. Layered to keep IsaacLab/IsaacSim dependencies isolated:
   - [skills/base.py](skills/base.py) — `BaseSkill` interface (lifecycle, config loading, RGB+depth capture helpers) and the `CameraCapture` dataclass (rgba, depth, semantic mask, pose, timestamp). RGB and depth cameras are injected per-skill; `capture_rgbd()` returns them keyed by `"rgb"` / `"depth"`.
   - [skills/core.py](skills/core.py) — shared types: `SkillStatus` (idle/planning/executing/grasping/done/failed), `SkillCommand` (composes mobility + arm + gripper into a single `RobotCommandBuilder.build_synchro_command`), `SkillTriggerResult`, and `RobotState` snapshot.
   - [skills/spot/](skills/spot/) — Spot-SDK-shaped facade so benchmark code mirrors bosdyn naming without depending on the real SDK. `create_standard_sdk()` returns an in-memory `Sdk`; `bind_isaaclab_spot_robot()` installs IsaacLab backends behind the same interface (lease, robot-command, image, manipulation, robot-state). The `IsaacLabImageBackend` ([skills/spot/isaaclab_backend.py:178](skills/spot/isaaclab_backend.py#L178)) wraps IsaacLab `Camera` sensors as Spot `ImageClient` sources.
   - [skills/locomotion/](skills/locomotion/) — velocity-driven Spot locomotion:
     - `SpotLocomotionClient` ([skills/locomotion/api.py](skills/locomotion/api.py)) — small user-facing API: `command_velocity(vx, vy, vrot)`, `stand()`/`stop()`, `step()`; `mode` returns `"stand"` or `"velocity"`.
     - `SpotFlatTerrainPolicy` ([skills/locomotion/spot_policy.py](skills/locomotion/spot_policy.py)) — IsaacLab runtime wrapper around a TorchScript flat-terrain policy. Loads `policy.pt` + `env.yaml`, finds the 12 leg joints by regex (`.*_hip_[xy]`, `.*_knee`), freezes non-leg joints at their default positions, swaps actuator gains between walk and stand modes, runs the policy every `decimation` steps. Observation is 48-d: `root_lin_vel_b(3) + root_ang_vel_b(3) + projected_gravity_b(3) + command(3) + leg_joint_pos_rel(12) + leg_joint_vel(12) + last_action(12)`.
     - `spot_loco.yaml` ([skills/locomotion/spot_loco.yaml](skills/locomotion/spot_loco.yaml)) — points the policy wrapper at `policy.pt` and `env.yaml` (relative paths resolved against the YAML's directory).
     - `env.yaml` is a **frozen IsaacLab training-config snapshot** (timestamped `log_dir`, 1875 lines of full-graph serialization). The wrapper only reads `actions.joint_pos.{scale,joint_names}`, `sim.dt`, and `decimation` — everything else is metadata. The `_PolicyYamlLoader` silently maps unknown `!!python/` tags to `None`, so do not rely on other fields being structurally valid.

The skills `Robot` is a simulation-backed facade — `Robot.power_on()` requires `authenticate()` first (when `require_authentication=True`) and an acquired lease, mirroring real-Spot preflight. `ensure_client(service_name)` accepts both `"robot-command"` and `"robotcommand"`/`"robot_command"` styles via [skills/spot/sdk.py:12](skills/spot/sdk.py#L12) `_normalize_service_name`.

## Gotchas Specific to This Fork

- The legacy package name was `zsos`; `eval_dummy_policy.sh` and `eval_oracle_fbe_policy.sh` still reference `zsos.run` — they will fail until updated to `vlfm.run`.
- `base_objectnav_policy.py` defines a local stub `class BasePolicy: pass` (under a commented-out try/except for `habitat_baselines`) so the module is importable without Habitat. Don't "clean up" that stub without restoring the import guard.
- `transformers` is pinned at exactly `4.26.0` in `pyproject.toml2` because newer versions break BLIP-2 in LAVIS. `timm` must stay at `0.6.12` for the same reason. Bumping either requires verifying BLIP-2 still loads.
- `setup.sh` installs `habitat-sim=0.2.5`, while `pyproject.toml2[habitat]` pins `habitat-sim @ v0.2.4`. Pick one and stick with it for a given env — mixing causes silent config-schema drift.
- The GroundingDINO and yolov7 clones in `environment_setup.sh` point at `chahyon-ku/` forks (not upstream IDEA-Research / WongKinYiu) because of Python-3.11 patches. Don't switch them back without re-verifying.

### Known path inconsistencies in `skills/` and `spot_model/`

The blockers that prevented `import skills` from working outside the parent `interactive-search` host have been fixed:
- [skills/base.py:12](skills/base.py#L12) and [skills/spot/isaaclab_backend.py:38](skills/spot/isaaclab_backend.py#L38) now import `capture_camera_payload` from the in-repo [skills/cam_utils.py](skills/cam_utils.py) instead of the external `helpers.cam_utils`.
- [skills/locomotion/env.yaml:134](skills/locomotion/env.yaml#L134) `usd_path` now points at `spot_model/spot_arm_w_cam.usd` (relative to repo root) instead of `/workspace/isaaclab/scripts/interactive-search/spot_model/...`.
- [skills/config/spot_arm_cameras.yaml](skills/config/spot_arm_cameras.yaml) now provides the camera-rig defaults expected by `DEFAULT_SPOT_ARM_CAMERA_RIG_CONFIG_PATH` ([skills/cam_utils.py:179](skills/cam_utils.py#L179)). Defines three named cameras (`frontleft`, `frontright`, `hand`) with their `{ENV_REGEX_NS}/Robot/...` depth-stream prim paths; both `grasp` and `depth_collision` groups span all three; `hand` has a `left: 38` crop.
- [spot_model/configuration/spot_arm_curobo.yaml](spot_model/configuration/spot_arm_curobo.yaml) `usd_path` and `urdf_path` now use `${VLFM_ROOT}/spot_model/spot_arm_only.{usd,urdf}`. **CuRobo does not expand `$VARS` natively.** Always load this config via [`skills.manipulation.load_curobo_robot_config`](skills/manipulation/curobo_config.py) — it does `yaml.safe_load` + env-var substitution + fail-fast on any unresolved variable. Do **not** call `yaml.safe_load` directly on this file. Set `VLFM_ROOT` to the repository root (i.e., the directory containing `spot_model/`), or pass it inline via the helper's `extra_env={"VLFM_ROOT": ...}` argument.

Remaining inconsistencies that are *not* on the locomotion-policy hot path but will bite anything that re-parses these configs:

- [skills/cam_utils.py:605](skills/cam_utils.py#L605) still does a lazy `from helpers.pcd_utils import fuse_camera_frames` inside `run_camera_pipeline()`. That code path is only reached when `CameraPipelineConfig.include_pointcloud=True`. `capture_camera_payload` (the function consumed by `BaseSkill.capture_rgbd`) does not touch it, so the basic skill flow works — but point-cloud fusion will fail until a `helpers.pcd_utils` replacement is provided or vendored.
- [skills/locomotion/env.yaml:1259](skills/locomotion/env.yaml#L1259) still hardcodes `log_dir: /workspace/isaaclab/logs/rsl_rl/spot_arm_w_cam_flat/2026-04-25_15-21-25` — frozen training metadata, not used at deploy time.
- `env.yaml` references external Python paths (`spot_sim_rl.tasks.events:*`, `isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp.rewards:*`) — must be on `PYTHONPATH` if anything beyond the locomotion wrapper re-instantiates the env from this file.

Treat `env.yaml` as a training-time snapshot — don't hand-edit anything beyond clearly-non-functional paths. If you need to regenerate it, re-export from IsaacLab rather than patching the YAML.

## Workflow Orchestration

### 1. Plan Node Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity
- Ask yourself whether this is best practice or a quick hack. Quick hack is not acceptable

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- Spawn multiple agents to work on each isolatable tasks and use git worktree as appropriate
- For complex problems, throw more compute at it via subagents
- One tack per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update "tasks/lessons.md" with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness
- Don't create tests just for the sake of the tests, make sure it capture main logic, user flow, and edge-cases.

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management
1. **Plan First**: Write plan to "tasks/todo.md" with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to "tasks/todo.md"
6. **Capture Lessons**: Update "tasks/lessons.md" after corrections

## Core Principles
- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.

## Other
- if you move file or rename, prefer `git mv`!
- Max 500 lines per file; split if larger
