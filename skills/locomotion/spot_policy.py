"""Isaac Lab deployment wrapper for the Spot flat-terrain locomotion policy."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import yaml


class _PolicyYamlLoader(yaml.SafeLoader):
    """Safe YAML loader that understands train-time tuple/slice tags."""


def _construct_python_tuple(loader: yaml.SafeLoader, node: yaml.nodes.SequenceNode) -> tuple[object, ...]:
    return tuple(loader.construct_sequence(node))


def _construct_python_slice(loader: yaml.SafeLoader, node: yaml.nodes.SequenceNode) -> slice:
    values = loader.construct_sequence(node)
    return slice(*values)


_PolicyYamlLoader.add_constructor("tag:yaml.org,2002:python/tuple", _construct_python_tuple)
_PolicyYamlLoader.add_constructor("tag:yaml.org,2002:python/object/apply:builtins.slice", _construct_python_slice)
_PolicyYamlLoader.add_constructor(None, lambda loader, node: None)

_DEFAULT_CONFIG_PATH = Path(__file__).with_name("spot_loco.yaml")


def _load_yaml(path: str | Path) -> dict:
    data = yaml.load(Path(path).read_text(), Loader=_PolicyYamlLoader)
    return {} if data is None else data


def _resolve_default_paths(config_path: Path = _DEFAULT_CONFIG_PATH) -> tuple[str, str]:
    config = _load_yaml(config_path)
    policy_path = Path(config["policy_path"])
    env_config_path = Path(config["env_config_path"])
    if not policy_path.is_absolute():
        policy_path = config_path.parent / policy_path
    if not env_config_path.is_absolute():
        env_config_path = config_path.parent / env_config_path
    return str(policy_path), str(env_config_path)


class SpotFlatTerrainPolicy:
    """Minimal Isaac Lab runtime wrapper for the exported Spot locomotion policy."""

    def __init__(self, policy_path: str | None = None, env_config_path: str | None = None) -> None:
        if policy_path is None and env_config_path is None:
            policy_path, env_config_path = _resolve_default_paths()
        if policy_path is None or env_config_path is None:
            raise ValueError("policy_path and env_config_path must both be provided, or both omitted.")

        self.policy_path = str(policy_path)
        self.env_config_path = str(env_config_path)

        config = _load_yaml(env_config_path)
        action_cfg = config["actions"]["joint_pos"]
        self._dt = float(config["sim"]["dt"])
        self._decimation = int(config["decimation"])
        self._action_scale = float(action_cfg["scale"])
        self._joint_name_patterns = tuple(str(name) for name in action_cfg["joint_names"])

        self.robot = None
        self.policy = None
        self._leg_joint_ids: list[int] | None = None
        self._leg_joint_names: tuple[str, ...] | None = None
        self._held_joint_ids: list[int] | None = None
        self._held_joint_names: tuple[str, ...] | None = None
        self._leg_actuator_names: tuple[str, ...] | None = None
        self._default_leg_pos: torch.Tensor | None = None
        self._held_joint_pos: torch.Tensor | None = None
        self._walk_actuator_gains: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None
        self._last_action: torch.Tensor | None = None
        self._cached_action: torch.Tensor | None = None
        self._stand_actuator_stiffness = 800.0
        self._stand_actuator_damping = 25.0
        self._stand_settle_time = 0.4
        self._zero_command_time = 0.0
        self._standing = False
        self._policy_counter = 0

    @property
    def control_dt(self) -> float:
        return self._dt

    @classmethod
    def default_control_dt(cls) -> float:
        _, env_config_path = _resolve_default_paths()
        return float(_load_yaml(env_config_path)["sim"]["dt"])

    def initialize(self, robot) -> None:
        if robot.data.joint_pos.shape[0] != 1:
            raise ValueError(f"SpotFlatTerrainPolicy expects one robot instance, got {robot.data.joint_pos.shape[0]}.")

        self.robot = robot
        self.policy = torch.jit.load(self.policy_path, map_location=robot.device).eval()

        leg_joint_ids, leg_joint_names = robot.find_joints(list(self._joint_name_patterns), preserve_order=False)
        if len(leg_joint_ids) != 12:
            raise RuntimeError(
                "Spot locomotion policy expected 12 leg joints, "
                f"resolved {len(leg_joint_ids)}: {tuple(leg_joint_names)}"
            )

        self._leg_joint_ids = list(leg_joint_ids)
        self._leg_joint_names = tuple(leg_joint_names)
        self._held_joint_ids = [index for index in range(robot.num_joints) if index not in self._leg_joint_ids]
        self._held_joint_names = tuple(robot.joint_names[index] for index in self._held_joint_ids)
        leg_joint_id_set = set(self._leg_joint_ids)
        self._leg_actuator_names = tuple(
            name
            for name, actuator in robot.actuators.items()
            if leg_joint_id_set.intersection(int(joint_id) for joint_id in actuator.joint_indices)
        )
        if not self._leg_actuator_names:
            raise RuntimeError("Spot locomotion policy expected leg actuators, found none.")
        self._default_leg_pos = robot.data.default_joint_pos[:, self._leg_joint_ids].clone()
        self._held_joint_pos = robot.data.default_joint_pos[:, self._held_joint_ids].clone()
        self._walk_actuator_gains = {
            name: (robot.actuators[name].stiffness.clone(), robot.actuators[name].damping.clone())
            for name in self._leg_actuator_names
        }
        self._last_action = torch.zeros((1, len(self._leg_joint_ids)), dtype=torch.float32, device=robot.device)
        self._cached_action = torch.zeros_like(self._last_action)
        self._zero_command_time = 0.0
        self._standing = False
        self._policy_counter = 0

    def forward(self, dt: float, command: Sequence[float] | torch.Tensor) -> None:
        if (
            self.robot is None
            or self.policy is None
            or self._leg_joint_ids is None
            or self._held_joint_ids is None
            or self._default_leg_pos is None
            or self._held_joint_pos is None
            or self._leg_actuator_names is None
            or self._walk_actuator_gains is None
            or self._last_action is None
            or self._cached_action is None
        ):
            raise RuntimeError("SpotFlatTerrainPolicy.initialize(robot) must be called before forward().")

        command_tensor = torch.as_tensor(command, dtype=torch.float32, device=self.robot.device).view(1, 3)
        if torch.count_nonzero(command_tensor).item() == 0:
            if self._standing:
                self.stand()
                return
            self._zero_command_time += float(dt)
            if self._zero_command_time >= self._stand_settle_time:
                self.stand()
                return
        else:
            self._zero_command_time = 0.0

        self._set_stand_gains(False)
        self.robot.set_joint_effort_target(torch.zeros_like(self._cached_action), joint_ids=self._leg_joint_ids)

        if self._policy_counter % self._decimation == 0:
            observation = self._compute_observation(command_tensor)
            with torch.inference_mode():
                action = self.policy(observation).to(dtype=torch.float32)
            self._cached_action.copy_(action)
            self._last_action.copy_(self._cached_action)

        joint_target = self.robot.data.joint_pos.clone()
        joint_target[:, self._held_joint_ids] = self._held_joint_pos
        joint_target[:, self._leg_joint_ids] = self._default_leg_pos + self._cached_action * self._action_scale
        self.robot.set_joint_position_target(joint_target)
        self._policy_counter += 1

    def stand(self) -> None:
        if (
            self.robot is None
            or self._leg_joint_ids is None
            or self._leg_actuator_names is None
            or self._walk_actuator_gains is None
            or self._last_action is None
            or self._cached_action is None
        ):
            raise RuntimeError("SpotFlatTerrainPolicy.initialize(robot) must be called before stand().")

        self._set_stand_gains(True)
        joint_target = self.robot.data.default_joint_pos.clone()
        self.robot.set_joint_position_target(joint_target)
        self.robot.set_joint_effort_target(torch.zeros_like(self._cached_action), joint_ids=self._leg_joint_ids)
        self._last_action.zero_()
        self._cached_action.zero_()
        self._zero_command_time = 0.0
        self._policy_counter = 0

    def _set_stand_gains(self, enabled: bool) -> None:
        if self._standing == enabled:
            return
        if (
            self.robot is None
            or self._leg_joint_ids is None
            or self._leg_actuator_names is None
            or self._walk_actuator_gains is None
        ):
            raise RuntimeError("SpotFlatTerrainPolicy.initialize(robot) must be called before changing stand gains.")

        if enabled:
            for name in self._leg_actuator_names:
                actuator = self.robot.actuators[name]
                actuator.stiffness.fill_(self._stand_actuator_stiffness)
                actuator.damping.fill_(self._stand_actuator_damping)
        else:
            for name in self._leg_actuator_names:
                stiffness, damping = self._walk_actuator_gains[name]
                actuator = self.robot.actuators[name]
                actuator.stiffness.copy_(stiffness)
                actuator.damping.copy_(damping)
        self._standing = enabled

    def _compute_observation(self, command: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                self.robot.data.root_lin_vel_b[:, :3],
                self.robot.data.root_ang_vel_b[:, :3],
                self.robot.data.projected_gravity_b[:, :3],
                command,
                self.robot.data.joint_pos[:, self._leg_joint_ids] - self._default_leg_pos,
                self.robot.data.joint_vel[:, self._leg_joint_ids],
                self._last_action,
            ],
            dim=-1,
        ).to(dtype=torch.float32)
