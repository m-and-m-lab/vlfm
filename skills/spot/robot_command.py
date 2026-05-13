"""Small Spot-SDK-like robot command surface for Isaac Lab backends."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Optional, Protocol, Sequence

from .frame_helpers import BODY_FRAME_NAME
from .geometry import EulerZXY

if TYPE_CHECKING:
    import torch


class RobotCommandFeedbackStatus(str, Enum):
    """High-level command feedback state."""

    STATUS_UNKNOWN = "STATUS_UNKNOWN"
    STATUS_PROCESSING = "STATUS_PROCESSING"
    STATUS_COMMAND_COMPLETE = "STATUS_COMMAND_COMPLETE"
    STATUS_COMMAND_OVERRIDDEN = "STATUS_COMMAND_OVERRIDDEN"
    STATUS_FAILED = "STATUS_FAILED"


class StandCommandFeedbackStatus(str, Enum):
    """Stand feedback state."""

    STATUS_UNKNOWN = "STATUS_UNKNOWN"
    STATUS_IN_PROGRESS = "STATUS_IN_PROGRESS"
    STATUS_IS_STANDING = "STATUS_IS_STANDING"


class SitCommandFeedbackStatus(str, Enum):
    """Sit feedback state."""

    STATUS_UNKNOWN = "STATUS_UNKNOWN"
    STATUS_IN_PROGRESS = "STATUS_IN_PROGRESS"
    STATUS_IS_SITTING = "STATUS_IS_SITTING"


@dataclass(frozen=True)
class MobilityParams:
    """Subset of Spot mobility params used by this repo."""

    body_height: float = 0.0
    footprint_R_body: EulerZXY = field(default_factory=EulerZXY)
    locomotion_hint: str = "auto"
    stair_hint: bool = False
    stairs_mode: Optional[str] = None
    external_force_params: object | None = None


@dataclass(frozen=True)
class Se2VelocityCommand:
    """Planar base velocity command."""

    v_x: float
    v_y: float
    v_rot: float
    params: MobilityParams
    frame_name: str = BODY_FRAME_NAME
    end_time_secs: Optional[float] = None


@dataclass(frozen=True)
class StandCommand:
    """Stand command."""

    params: MobilityParams


@dataclass(frozen=True)
class SitCommand:
    """Sit command."""

    params: MobilityParams


@dataclass(frozen=True)
class JointPositionCommand:
    """Joint-space command for a named set of joints."""

    positions: torch.Tensor
    joint_names: Sequence[str]


@dataclass(frozen=True)
class GripperPositionCommand:
    """Single gripper position command."""

    position: float


@dataclass(frozen=True)
class RobotCommand:
    """Composed robot command carrying mobility, arm, and gripper components."""

    mobility_command: Se2VelocityCommand | StandCommand | SitCommand | None = None
    arm_command: JointPositionCommand | None = None
    gripper_command: GripperPositionCommand | None = None


@dataclass(frozen=True)
class StandCommandFeedback:
    """Stand feedback payload."""

    status: StandCommandFeedbackStatus = StandCommandFeedbackStatus.STATUS_UNKNOWN


@dataclass(frozen=True)
class SitCommandFeedback:
    """Sit feedback payload."""

    status: SitCommandFeedbackStatus = SitCommandFeedbackStatus.STATUS_UNKNOWN


@dataclass(frozen=True)
class MobilityCommandFeedback:
    """Mobility feedback payload."""

    status: RobotCommandFeedbackStatus = RobotCommandFeedbackStatus.STATUS_UNKNOWN
    stand_feedback: StandCommandFeedback = field(default_factory=StandCommandFeedback)
    sit_feedback: SitCommandFeedback = field(default_factory=SitCommandFeedback)


@dataclass(frozen=True)
class SynchronizedCommandFeedback:
    """Grouped synchronized feedback payload."""

    mobility_command_feedback: MobilityCommandFeedback = field(default_factory=MobilityCommandFeedback)


@dataclass(frozen=True)
class RobotCommandFeedback:
    """Top-level feedback payload."""

    status: RobotCommandFeedbackStatus = RobotCommandFeedbackStatus.STATUS_UNKNOWN
    synchronized_feedback: SynchronizedCommandFeedback = field(default_factory=SynchronizedCommandFeedback)


@dataclass(frozen=True)
class RobotCommandFeedbackResponse:
    """Feedback response returned by the command client."""

    command_id: int
    feedback: RobotCommandFeedback
    message: str = ""

    @property
    def robot_command_id(self) -> int:
        """Alias used by the Spot SDK response objects."""

        return self.command_id


class RobotCommandBackend(Protocol):
    """Backend interface used by :class:`RobotCommandClient`."""

    def robot_command(
        self,
        command: RobotCommand,
        end_time_secs: Optional[float] = None,
        lease: object | None = None,
        timesync_endpoint: object | None = None,
    ) -> int:
        """Submit a command and return a command id."""

    def robot_command_feedback(self, command_id: int) -> RobotCommandFeedbackResponse:
        """Return the current feedback for the provided command id."""


class RobotCommandClient:
    """Small Spot-style command client."""

    default_service_name = "robot-command"
    service_type = "bosdyn.api.RobotCommandService"

    def __init__(self, backend: RobotCommandBackend) -> None:
        self._backend = backend

    def robot_command(
        self,
        command: RobotCommand,
        end_time_secs: Optional[float] = None,
        timesync_endpoint: object | None = None,
        lease: object | None = None,
        **_: object,
    ) -> int:
        return self._backend.robot_command(
            command,
            end_time_secs=end_time_secs,
            lease=lease,
            timesync_endpoint=timesync_endpoint,
        )

    def robot_command_feedback(
        self,
        robot_command_id: int | None = None,
        *,
        command_id: int | None = None,
        **_: object,
    ) -> RobotCommandFeedbackResponse:
        resolved_command_id = robot_command_id if robot_command_id is not None else command_id
        if resolved_command_id is None:
            raise ValueError("robot_command_feedback requires robot_command_id or command_id.")
        return self._backend.robot_command_feedback(int(resolved_command_id))


class RobotCommandBuilder:
    """Helpers that mirror the Spot SDK's robot command builder surface."""

    @staticmethod
    def mobility_params(
        body_height: float = 0.0,
        footprint_R_body: EulerZXY = EulerZXY(),
        locomotion_hint: str = "auto",
        stair_hint: bool = False,
        external_force_params: object | None = None,
        stairs_mode: Optional[str] = None,
    ) -> MobilityParams:
        return MobilityParams(
            body_height=body_height,
            footprint_R_body=footprint_R_body,
            locomotion_hint=locomotion_hint,
            stair_hint=stair_hint,
            external_force_params=external_force_params,
            stairs_mode=stairs_mode,
        )

    @staticmethod
    def synchro_velocity_command(
        v_x: float,
        v_y: float,
        v_rot: float,
        params: Optional[MobilityParams] = None,
        body_height: float = 0.0,
        locomotion_hint: str = "auto",
        frame_name: str = BODY_FRAME_NAME,
        build_on_command: Optional[RobotCommand] = None,
    ) -> RobotCommand:
        command = RobotCommand(
            mobility_command=Se2VelocityCommand(
                v_x=float(v_x),
                v_y=float(v_y),
                v_rot=float(v_rot),
                params=params or RobotCommandBuilder.mobility_params(
                    body_height=body_height,
                    locomotion_hint=locomotion_hint,
                ),
                frame_name=frame_name,
            )
        )
        return RobotCommandBuilder.build_synchro_command(build_on_command, command) if build_on_command else command

    @staticmethod
    def synchro_stand_command(
        params: Optional[MobilityParams] = None,
        body_height: float = 0.0,
        footprint_R_body: EulerZXY = EulerZXY(),
        build_on_command: Optional[RobotCommand] = None,
    ) -> RobotCommand:
        command = RobotCommand(
            mobility_command=StandCommand(
                params=params
                or RobotCommandBuilder.mobility_params(
                    body_height=body_height,
                    footprint_R_body=footprint_R_body,
                )
            )
        )
        return RobotCommandBuilder.build_synchro_command(build_on_command, command) if build_on_command else command

    @staticmethod
    def synchro_sit_command(
        params: Optional[MobilityParams] = None,
        build_on_command: Optional[RobotCommand] = None,
    ) -> RobotCommand:
        command = RobotCommand(mobility_command=SitCommand(params=params or RobotCommandBuilder.mobility_params()))
        return RobotCommandBuilder.build_synchro_command(build_on_command, command) if build_on_command else command

    @staticmethod
    def arm_joint_move_command(
        joint_positions: torch.Tensor,
        joint_names: Sequence[str],
        build_on_command: Optional[RobotCommand] = None,
    ) -> RobotCommand:
        command = RobotCommand(
            arm_command=JointPositionCommand(
                positions=joint_positions,
                joint_names=tuple(joint_names),
            )
        )
        return RobotCommandBuilder.build_synchro_command(build_on_command, command) if build_on_command else command

    @staticmethod
    def claw_gripper_open_angle_command(
        gripper_q: float,
        build_on_command: Optional[RobotCommand] = None,
    ) -> RobotCommand:
        command = RobotCommand(gripper_command=GripperPositionCommand(position=float(gripper_q)))
        return RobotCommandBuilder.build_synchro_command(build_on_command, command) if build_on_command else command

    @staticmethod
    def build_synchro_command(*commands: Optional[RobotCommand]) -> RobotCommand:
        combined = RobotCommand()
        for command in commands:
            if command is None:
                continue
            if command.mobility_command is not None:
                combined = replace(combined, mobility_command=command.mobility_command)
            if command.arm_command is not None:
                combined = replace(combined, arm_command=command.arm_command)
            if command.gripper_command is not None:
                combined = replace(combined, gripper_command=command.gripper_command)
        return combined


def blocking_command(
    command_client: RobotCommandClient,
    command: RobotCommand,
    check_status_fn,
    end_time_secs: Optional[float] = None,
    timeout_sec: float = 10.0,
    update_frequency: float = 10.0,
) -> None:
    """Issue a command and block until the supplied status callback succeeds."""

    command_id = command_client.robot_command(command, end_time_secs=end_time_secs)
    start_time = time.monotonic()
    sleep_dt = 1.0 / max(update_frequency, 1.0e-6)
    while True:
        response = command_client.robot_command_feedback(command_id)
        if check_status_fn(response):
            return
        if time.monotonic() - start_time > timeout_sec:
            raise TimeoutError(f"Command {command_id} did not complete within {timeout_sec:.1f}s.")
        time.sleep(sleep_dt)


def blocking_stand(
    command_client: RobotCommandClient,
    timeout_sec: float = 10.0,
    update_frequency: float = 10.0,
    params: Optional[MobilityParams] = None,
) -> None:
    """Issue a stand command and wait until the backend reports a standing state."""

    def check_stand_status(response: RobotCommandFeedbackResponse) -> bool:
        status = response.feedback.synchronized_feedback.mobility_command_feedback.stand_feedback.status
        return status == StandCommandFeedbackStatus.STATUS_IS_STANDING

    blocking_command(
        command_client,
        RobotCommandBuilder.synchro_stand_command(params=params),
        check_stand_status,
        timeout_sec=timeout_sec,
        update_frequency=update_frequency,
    )


def blocking_sit(
    command_client: RobotCommandClient,
    timeout_sec: float = 10.0,
    update_frequency: float = 10.0,
    params: Optional[MobilityParams] = None,
) -> None:
    """Issue a sit command and wait until the backend reports a sitting state."""

    def check_sit_status(response: RobotCommandFeedbackResponse) -> bool:
        status = response.feedback.synchronized_feedback.mobility_command_feedback.sit_feedback.status
        return status == SitCommandFeedbackStatus.STATUS_IS_SITTING

    blocking_command(
        command_client,
        RobotCommandBuilder.synchro_sit_command(params=params),
        check_sit_status,
        timeout_sec=timeout_sec,
        update_frequency=update_frequency,
    )
