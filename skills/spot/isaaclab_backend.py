"""IsaacLab-backed binders and service backends for the Spot SDK facade."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Sequence

try:
    import torch
except ModuleNotFoundError:
    torch = None

from .robot_command import (
    MobilityCommandFeedback,
    RobotCommand,
    RobotCommandBackend,
    RobotCommandClient,
    RobotCommandFeedback,
    RobotCommandFeedbackResponse,
    RobotCommandFeedbackStatus,
    SynchronizedCommandFeedback,
)
from .sdk import (
    ImageClient,
    ImageResponse,
    ImageSource,
    InMemoryLeaseBackend,
    InMemoryRobotControlBackend,
    LeaseClient,
    ManipulationApiClient,
    ManipulationApiFeedbackResponse,
    ManipulationApiResponse,
    ManipulationFeedbackState,
    Robot,
    RobotStateClient,
)
from skills.cam_utils import capture_camera_payload


def _noop(_: str) -> None:
    return None


def _capture_camera(camera: Any) -> dict[str, Any]:
    return capture_camera_payload(camera, camera_axes="usd")


class _ManipulationPhase(str, Enum):
    REQUESTED = "requested"
    OPENING_GRIPPER = "opening_gripper"
    SETTLING_CAMERA = "settling_camera"
    PLANNING = "planning"
    RUNNING = "running"


@dataclass
class _ManipulationRecord:
    command_id: int
    state: ManipulationFeedbackState
    message: str = ""
    request: Any | None = None
    phase: _ManipulationPhase = _ManipulationPhase.REQUESTED
    open_steps_remaining: int = 0
    settle_steps_remaining: int = 0


class IsaacLabRobotCommandBackend(RobotCommandBackend):
    """Apply arm and gripper commands directly to an IsaacLab articulation."""

    def __init__(
        self,
        *,
        robot,
        scene,
        arm_joint_ids: Sequence[int],
        arm_joint_names: Sequence[str],
        gripper_joint_ids: Sequence[int],
        control_backend: InMemoryRobotControlBackend,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self._robot = robot
        self._scene = scene
        self._arm_joint_ids = list(arm_joint_ids)
        self._arm_joint_names = list(arm_joint_names)
        self._gripper_joint_ids = list(gripper_joint_ids)
        self._control_backend = control_backend
        self._log = log or _noop
        self._last_command_id = 0
        self._feedback_by_id: dict[int, RobotCommandFeedbackResponse] = {}

    def robot_command(
        self,
        command: RobotCommand,
        end_time_secs: float | None = None,
        lease: object | None = None,
        timesync_endpoint: object | None = None,
    ) -> int:
        del end_time_secs, lease, timesync_endpoint
        if torch is None:
            raise ModuleNotFoundError("torch")
        if not self._control_backend.is_powered_on():
            raise RuntimeError("Robot must be powered on before issuing arm or gripper commands.")
        if command.mobility_command is not None:
            raise ValueError("This fixed-base IsaacLab backend does not accept mobility commands.")
        if command.arm_command is None and command.gripper_command is None:
            raise ValueError("RobotCommand must contain an arm or gripper command.")

        if command.arm_command is not None:
            arm_target = self._resolve_arm_positions(command)
            self._robot.set_joint_position_target(arm_target, joint_ids=self._arm_joint_ids)
        if command.gripper_command is not None and self._gripper_joint_ids:
            grip_target = torch.full(
                (self._scene.num_envs, len(self._gripper_joint_ids)),
                float(command.gripper_command.position),
                device=self._robot.data.joint_pos.device,
            )
            self._robot.set_joint_position_target(grip_target, joint_ids=self._gripper_joint_ids)

        self._last_command_id += 1
        response = RobotCommandFeedbackResponse(
            command_id=self._last_command_id,
            feedback=RobotCommandFeedback(
                status=RobotCommandFeedbackStatus.STATUS_COMMAND_COMPLETE,
                synchronized_feedback=SynchronizedCommandFeedback(
                    mobility_command_feedback=MobilityCommandFeedback(
                        status=RobotCommandFeedbackStatus.STATUS_COMMAND_COMPLETE,
                    )
                ),
            ),
            message="Applied arm/gripper command to IsaacLab articulation.",
        )
        self._feedback_by_id[self._last_command_id] = response
        self._log(f"[SpotSDK] Applied robot-command id={self._last_command_id}.")
        return self._last_command_id

    def robot_command_feedback(self, command_id: int) -> RobotCommandFeedbackResponse:
        response = self._feedback_by_id.get(int(command_id))
        if response is None:
            return RobotCommandFeedbackResponse(
                command_id=int(command_id),
                feedback=RobotCommandFeedback(status=RobotCommandFeedbackStatus.STATUS_UNKNOWN),
                message="Unknown robot command id.",
            )
        return response

    def _resolve_arm_positions(self, command: RobotCommand) -> torch.Tensor:
        if torch is None:
            raise ModuleNotFoundError("torch")
        arm_command = command.arm_command
        if arm_command is None:
            raise RuntimeError("Arm command resolution requires arm_command to be present.")
        positions = arm_command.positions
        if positions.ndim == 1:
            positions = positions.unsqueeze(0)
        if list(arm_command.joint_names) == list(self._arm_joint_names):
            return positions
        name_to_idx = {name: idx for idx, name in enumerate(arm_command.joint_names)}
        missing = [name for name in self._arm_joint_names if name not in name_to_idx]
        if missing:
            raise RuntimeError(f"Arm command is missing required joints: {missing}")
        order = [name_to_idx[name] for name in self._arm_joint_names]
        return positions[:, order]


class IsaacLabRobotStateBackend:
    """Return the current IsaacLab robot state snapshot."""

    def __init__(self, runner: Any) -> None:
        self._runner = runner

    def get_robot_state(self, **kwargs) -> Any:
        del kwargs
        return self._runner._build_robot_state()


class IsaacLabImageBackend:
    """Expose mounted IsaacLab cameras through an SDK-shaped image service."""

    def __init__(self, cameras: Mapping[str, object]) -> None:
        self._cameras = dict(cameras)

    def list_image_sources(self) -> Sequence[ImageSource]:
        return tuple(ImageSource(name=name) for name in self._cameras)

    def get_image_from_sources(self, image_sources: Sequence[str]) -> Sequence[ImageResponse]:
        responses: list[ImageResponse] = []
        for source_name in image_sources:
            camera = self._cameras.get(str(source_name))
            if camera is None:
                raise KeyError(f"Unknown image source '{source_name}'.")
            responses.append(ImageResponse(source=str(source_name), image=_capture_camera(camera)))
        return tuple(responses)


class IsaacLabManipulationApiBackend:
    """Wrap an :class:`IsaacLabManipulationRunner` with a Spot-SDK-like API."""

    def __init__(
        self,
        runner: Any,
        *,
        sim_dt: float,
        stage_getter: Callable[[], Any],
        update_world: bool = True,
        robot_reference_prim_path: str | None = None,
        ignore_substring: Optional[Sequence[str]] = None,
        gripper_open_timeout_sec: float = 3.0,
        gripper_settle_steps: int = 10,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self._runner = runner
        self._sim_dt = float(sim_dt)
        self._stage_getter = stage_getter
        self._update_world = bool(update_world)
        self._robot_reference_prim_path = robot_reference_prim_path
        self._ignore_substring = list(ignore_substring or [])
        self._gripper_open_timeout_sec = float(gripper_open_timeout_sec)
        self._gripper_settle_steps = int(max(0, gripper_settle_steps))
        self._log = log or _noop
        self._next_command_id = 0
        self._active_command_id: int | None = None
        self._records: dict[int, _ManipulationRecord] = {}

    def manipulation_api_command(self, manipulation_api_request: Any, **kwargs) -> ManipulationApiResponse:
        del kwargs
        if self._active_command_id is not None:
            active_record = self._records[self._active_command_id]
            if not active_record.state.is_terminal:
                active_record.state = ManipulationFeedbackState.OVERRIDDEN
                active_record.message = "Command was overridden by a newer manipulation request."

        self._next_command_id += 1
        command_id = self._next_command_id
        self._records[command_id] = _ManipulationRecord(
            command_id=command_id,
            state=ManipulationFeedbackState.PLANNING,
            message="Manipulation request scheduled.",
            request=manipulation_api_request,
            phase=_ManipulationPhase.REQUESTED,
        )

        reason = str(getattr(manipulation_api_request, "trigger_reason", "manual"))
        self._active_command_id = command_id
        self._log(f"[Manipulation] command_id={command_id} scheduled reason={reason}.")
        return ManipulationApiResponse(command_id=command_id, message="Manipulation request scheduled.")

    def manipulation_api_feedback_command(
        self,
        manipulation_api_feedback_request: Any,
        **kwargs,
    ) -> ManipulationApiFeedbackResponse:
        del kwargs
        command_id = self._coerce_command_id(manipulation_api_feedback_request)
        record = self._records.get(command_id)
        if record is None:
            return ManipulationApiFeedbackResponse(
                command_id=command_id,
                state=ManipulationFeedbackState.UNKNOWN,
                message="Unknown manipulation command id.",
            )
        return ManipulationApiFeedbackResponse(
            command_id=record.command_id,
            state=record.state,
            message=record.message,
        )

    def step(self) -> None:
        if self._active_command_id is None:
            return
        record = self._records[self._active_command_id]
        if record.state.is_terminal:
            self._active_command_id = None
            return

        if record.phase == _ManipulationPhase.REQUESTED:
            self._start_open_gripper_phase(record)
            return
        if record.phase == _ManipulationPhase.OPENING_GRIPPER:
            self._step_open_gripper_phase(record)
            return
        if record.phase == _ManipulationPhase.SETTLING_CAMERA:
            self._step_camera_settle_phase(record)
            return
        if record.phase == _ManipulationPhase.PLANNING:
            self._run_planning_phase(record)
            return

        command = self._runner.step()
        current_skill_state = self._map_skill_status(getattr(self._runner._skill.status, "value", self._runner._skill.status))
        if record.state == ManipulationFeedbackState.PLANNING and current_skill_state == ManipulationFeedbackState.PLANNING:
            current_skill_state = ManipulationFeedbackState.EXECUTING
        record.state = current_skill_state
        if command is not None and getattr(command, "message", ""):
            record.message = str(command.message)
        elif current_skill_state == ManipulationFeedbackState.DONE:
            record.message = "Manipulation command completed."
        elif current_skill_state == ManipulationFeedbackState.GRASPING and not record.message:
            record.message = "Closing gripper at grasp target."
        elif current_skill_state == ManipulationFeedbackState.EXECUTING and not record.message:
            record.message = "Executing planned arm trajectory."
        if current_skill_state.is_terminal:
            self._active_command_id = None

    def _start_open_gripper_phase(self, record: _ManipulationRecord) -> None:
        try:
            self._runner.initialize_grasp_request()
        except Exception as exc:
            record.state = ManipulationFeedbackState.FAILED
            record.message = str(exc)
            self._active_command_id = None
            self._log(f"[Manipulation] command_id={record.command_id} initialization failed: {exc}")
            return
        record.phase = _ManipulationPhase.OPENING_GRIPPER
        record.state = ManipulationFeedbackState.PLANNING
        record.open_steps_remaining = self._open_timeout_steps()
        record.message = "Opening gripper before AO-Grasp capture."
        self._log(f"[Manipulation] command_id={record.command_id} opening gripper before capture.")

    def _step_open_gripper_phase(self, record: _ManipulationRecord) -> None:
        try:
            is_open = bool(self._runner.is_gripper_open())
        except Exception as exc:
            record.state = ManipulationFeedbackState.FAILED
            record.message = str(exc)
            self._active_command_id = None
            return
        if is_open:
            record.phase = _ManipulationPhase.SETTLING_CAMERA
            record.settle_steps_remaining = max(0, self._gripper_settle_steps - 1)
            record.message = "Gripper open; settling cameras before AO-Grasp capture."
            self._log(f"[Manipulation] command_id={record.command_id} gripper open; settling cameras.")
            return
        record.open_steps_remaining -= 1
        if record.open_steps_remaining < 0:
            record.state = ManipulationFeedbackState.FAILED
            record.message = (
                f"Gripper did not reach open target within {self._gripper_open_timeout_sec:.2f}s before capture."
            )
            self._active_command_id = None

    def _step_camera_settle_phase(self, record: _ManipulationRecord) -> None:
        if record.settle_steps_remaining > 0:
            record.settle_steps_remaining -= 1
            return
        record.phase = _ManipulationPhase.PLANNING
        record.message = "Planning AO-Grasp + CuRobo request."
        self._log(f"[Manipulation] command_id={record.command_id} planning started.")

    def _run_planning_phase(self, record: _ManipulationRecord) -> None:
        reason = str(getattr(record.request, "trigger_reason", "manual"))
        result = self._runner.trigger(
            sim_dt=self._sim_dt,
            update_world=self._update_world,
            stage=self._stage_getter(),
            ignore_substring=self._ignore_substring,
            reference_prim_path=self._robot_reference_prim_path,
            reason=reason,
            manipulation_request=record.request,
        )
        if not result.success:
            record.state = ManipulationFeedbackState.FAILED
            record.message = result.message
            self._active_command_id = None
            self._log(f"[Manipulation] command_id={record.command_id} planning failed: {result.message}")
            return
        record.phase = _ManipulationPhase.RUNNING
        record.state = ManipulationFeedbackState.PLANNING
        record.message = result.message or "Manipulation request accepted."
        self._log(f"[Manipulation] command_id={record.command_id} planning accepted: {record.message}")

    def _open_timeout_steps(self) -> int:
        if self._sim_dt <= 0.0:
            raise RuntimeError(f"Simulation dt must be positive, got {self._sim_dt}.")
        return max(1, int((self._gripper_open_timeout_sec / self._sim_dt) + 0.999999))

    @staticmethod
    def _coerce_command_id(request: Any) -> int:
        if isinstance(request, int):
            return int(request)
        if hasattr(request, "command_id"):
            return int(request.command_id)
        if hasattr(request, "manipulation_cmd_id"):
            return int(request.manipulation_cmd_id)
        raise TypeError("Manipulation feedback request must be an int or expose command_id.")

    @staticmethod
    def _map_skill_status(raw_status: object) -> ManipulationFeedbackState:
        status = str(raw_status or "").strip().lower()
        mapping = {
            "planning": ManipulationFeedbackState.PLANNING,
            "executing": ManipulationFeedbackState.EXECUTING,
            "grasping": ManipulationFeedbackState.GRASPING,
            "done": ManipulationFeedbackState.DONE,
            "failed": ManipulationFeedbackState.FAILED,
        }
        return mapping.get(status, ManipulationFeedbackState.UNKNOWN)


@dataclass
class IsaacLabSpotServiceBinder:
    """Own the Spot-SDK-style service backends for an IsaacLab manipulation scene."""

    robot: Robot
    command_backend: IsaacLabRobotCommandBackend
    manipulation_backend: IsaacLabManipulationApiBackend
    state_backend: IsaacLabRobotStateBackend
    image_backend: IsaacLabImageBackend
    lease_backend: InMemoryLeaseBackend
    control_backend: InMemoryRobotControlBackend

    def step(self) -> None:
        self.manipulation_backend.step()


def bind_isaaclab_spot_robot(
    robot: Robot,
    *,
    runner: Any,
    robot_articulation: Any,
    scene: Any,
    arm_joint_ids: Sequence[int],
    arm_joint_names: Sequence[str],
    gripper_joint_ids: Sequence[int],
    cameras: Mapping[str, object],
    sim_dt: float,
    stage_getter: Callable[[], Any],
    update_world: bool = True,
    robot_reference_prim_path: str | None = None,
    ignore_substring: Optional[Sequence[str]] = None,
    gripper_open_timeout_sec: float = 3.0,
    gripper_settle_steps: int = 10,
    log: Callable[[str], None] | None = None,
) -> IsaacLabSpotServiceBinder:
    """Bind a simulation-backed Spot robot facade to an IsaacLab manipulation runner."""

    logger = log or _noop
    lease_backend = InMemoryLeaseBackend()
    control_backend = InMemoryRobotControlBackend(client_name=robot.name, lease_backend=lease_backend)
    command_backend = IsaacLabRobotCommandBackend(
        robot=robot_articulation,
        scene=scene,
        arm_joint_ids=arm_joint_ids,
        arm_joint_names=arm_joint_names,
        gripper_joint_ids=gripper_joint_ids,
        control_backend=control_backend,
        log=logger,
    )
    manipulation_backend = IsaacLabManipulationApiBackend(
        runner,
        sim_dt=sim_dt,
        stage_getter=stage_getter,
        update_world=update_world,
        robot_reference_prim_path=robot_reference_prim_path,
        ignore_substring=ignore_substring,
        gripper_open_timeout_sec=gripper_open_timeout_sec,
        gripper_settle_steps=gripper_settle_steps,
        log=logger,
    )
    state_backend = IsaacLabRobotStateBackend(runner)
    image_backend = IsaacLabImageBackend(cameras)

    robot.install_control_backend(control_backend)
    robot.install_service_factory(LeaseClient.default_service_name, lambda: LeaseClient(lease_backend, client_name=robot.name))
    robot.install_service_factory(RobotCommandClient.default_service_name, lambda: RobotCommandClient(command_backend))
    robot.install_service_factory(RobotStateClient.default_service_name, lambda: RobotStateClient(state_backend))
    robot.install_service_factory(ImageClient.default_service_name, lambda: ImageClient(image_backend))
    robot.install_service_factory(
        ManipulationApiClient.default_service_name,
        lambda: ManipulationApiClient(manipulation_backend),
    )

    return IsaacLabSpotServiceBinder(
        robot=robot,
        command_backend=command_backend,
        manipulation_backend=manipulation_backend,
        state_backend=state_backend,
        image_backend=image_backend,
        lease_backend=lease_backend,
        control_backend=control_backend,
    )


__all__ = [
    "IsaacLabImageBackend",
    "IsaacLabManipulationApiBackend",
    "IsaacLabRobotCommandBackend",
    "IsaacLabRobotStateBackend",
    "IsaacLabSpotServiceBinder",
    "bind_isaaclab_spot_robot",
]
