"""Shared generic types for interactive-search skills."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional, Sequence

from skills.spot import JointPositionCommand, RobotCommand, RobotCommandBuilder

if TYPE_CHECKING:
    import torch


class SkillStatus(str, Enum):
    """Lifecycle status for skills managed by the interactive-search API."""

    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    GRASPING = "grasping"
    DONE = "done"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (SkillStatus.DONE, SkillStatus.FAILED)


ArmJointCommand = JointPositionCommand


@dataclass
class SkillCommand:
    """Generic command output from a skill step."""

    status: SkillStatus
    robot_command: Optional[RobotCommand] = None
    arm_command: Optional[JointPositionCommand] = None
    gripper_position: Optional[float] = None
    message: str = ""

    def __post_init__(self) -> None:
        commands: list[RobotCommand] = []
        if self.robot_command is not None:
            commands.append(self.robot_command)
        if self.arm_command is not None:
            commands.append(
                RobotCommandBuilder.arm_joint_move_command(
                    joint_positions=self.arm_command.positions,
                    joint_names=self.arm_command.joint_names,
                )
            )
        if self.gripper_position is not None:
            commands.append(RobotCommandBuilder.claw_gripper_open_angle_command(self.gripper_position))
        if commands:
            self.robot_command = RobotCommandBuilder.build_synchro_command(*commands)
            self.arm_command = self.robot_command.arm_command
            if self.robot_command.gripper_command is not None:
                self.gripper_position = self.robot_command.gripper_command.position
        else:
            self.robot_command = None
            self.arm_command = None

    @property
    def mobility_command(self):
        return self.robot_command.mobility_command if self.robot_command is not None else None

    @property
    def gripper_command(self):
        return self.robot_command.gripper_command if self.robot_command is not None else None


@dataclass(frozen=True)
class SkillTriggerResult:
    """Outcome of a skill trigger request."""

    success: bool
    status: str
    message: str = ""
    target_pos_w: Optional[torch.Tensor] = None
    target_quat_w: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class RobotState:
    """Snapshot of robot state needed by skills."""

    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    joint_names: Sequence[str]
    root_pose_w: torch.Tensor
    ee_pose_w: torch.Tensor
    root_lin_vel_b: Optional[torch.Tensor] = None
    root_ang_vel_b: Optional[torch.Tensor] = None
    projected_gravity_b: Optional[torch.Tensor] = None
    last_action: Optional[torch.Tensor] = None
