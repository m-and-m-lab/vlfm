"""Small simulation-backed Spot SDK facade used by interactive-search."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence


def _normalize_service_name(service_name: str) -> str:
    normalized = str(service_name or "").strip().lower()
    if not normalized:
        raise ValueError("service_name must be a non-empty string.")
    normalized = normalized.replace("_", "-")
    aliases = {
        "robotcommand": "robot-command",
        "robot-command": "robot-command",
        "robot-state": "robot-state",
        "robotstate": "robot-state",
        "image": "image",
        "lease": "lease",
        "manipulation": "manipulation",
        "manipulation-api": "manipulation",
        "manipulationapi": "manipulation",
    }
    return aliases.get(normalized, normalized)


class TimeSyncEndpoint:
    """Tiny placeholder endpoint matching the Spot SDK's timesync concept."""

    def __init__(self) -> None:
        self.established_at = time.monotonic()


class TimeSyncThread:
    """Minimal Spot-SDK-style time-sync facade."""

    def __init__(self) -> None:
        self.endpoint = TimeSyncEndpoint()

    def wait_for_sync(self, timeout_sec: float | None = None) -> TimeSyncEndpoint:
        del timeout_sec
        return self.endpoint


@dataclass(frozen=True)
class Lease:
    """Simple lease token for simulation-backed clients."""

    resource: str = "body"
    sequence: int = 0
    client_name: str = ""


@dataclass(frozen=True)
class LeaseOwner:
    """Current lease owner."""

    client_name: str


@dataclass(frozen=True)
class LeaseResourceEntry:
    """Lease state for a single resource."""

    resource: str
    lease: Lease | None
    lease_owner: LeaseOwner | None = None


class LeaseBackend(Protocol):
    """Backend used by :class:`LeaseClient`."""

    def acquire(self, *, client_name: str, resource: str = "body") -> Lease: ...

    def return_lease(self, lease: Lease) -> None: ...

    def retain_lease(self, lease: Lease) -> Lease: ...

    def list_leases(self) -> Sequence[LeaseResourceEntry]: ...


class InMemoryLeaseBackend:
    """In-memory lease backend shared by the SDK facade and tests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sequence = 0
        self._current_lease: Lease | None = None
        self._retain_count = 0

    @property
    def retain_count(self) -> int:
        return self._retain_count

    def acquire(self, *, client_name: str, resource: str = "body") -> Lease:
        with self._lock:
            if self._current_lease is not None:
                raise RuntimeError(
                    f"Resource '{self._current_lease.resource}' is already leased by "
                    f"'{self._current_lease.client_name}'."
                )
            self._sequence += 1
            self._current_lease = Lease(resource=resource, sequence=self._sequence, client_name=client_name)
            return self._current_lease

    def return_lease(self, lease: Lease) -> None:
        with self._lock:
            if self._current_lease is None:
                return
            if lease != self._current_lease:
                raise RuntimeError(
                    "Returned lease does not match the active lease: "
                    f"current={self._current_lease}, returned={lease}."
                )
            self._current_lease = None

    def retain_lease(self, lease: Lease) -> Lease:
        with self._lock:
            if self._current_lease is None:
                raise RuntimeError("No active lease exists to retain.")
            if lease.sequence != self._current_lease.sequence:
                raise RuntimeError(
                    f"Lease sequence mismatch: current={self._current_lease.sequence}, retained={lease.sequence}."
                )
            self._retain_count += 1
            return self._current_lease

    def list_leases(self) -> Sequence[LeaseResourceEntry]:
        with self._lock:
            owner = None
            if self._current_lease is not None:
                owner = LeaseOwner(client_name=self._current_lease.client_name)
            return (
                LeaseResourceEntry(resource="all-leases", lease=self._current_lease, lease_owner=owner),
                LeaseResourceEntry(resource="body", lease=self._current_lease, lease_owner=owner),
            )

    def has_active_lease(self, *, client_name: str | None = None) -> bool:
        with self._lock:
            if self._current_lease is None:
                return False
            if client_name is None:
                return True
            return self._current_lease.client_name == client_name


class LeaseClient:
    """Spot-SDK-style lease client."""

    default_service_name = "lease"
    service_type = "bosdyn.api.LeaseService"

    def __init__(self, backend: LeaseBackend, *, client_name: str) -> None:
        self._backend = backend
        self._client_name = client_name
        self._active_lease: Lease | None = None

    @property
    def active_lease(self) -> Lease | None:
        return self._active_lease

    def acquire(self, resource: str = "body") -> Lease:
        lease = self._backend.acquire(client_name=self._client_name, resource=resource)
        self._active_lease = lease
        return lease

    def return_lease(self, lease: Lease | None = None) -> None:
        lease_to_return = self._active_lease if lease is None else lease
        if lease_to_return is None:
            raise RuntimeError("No active lease is available to return.")
        self._backend.return_lease(lease_to_return)
        if self._active_lease == lease_to_return:
            self._active_lease = None

    def retain_lease(self, lease: Lease | None = None) -> Lease:
        lease_to_retain = self._active_lease if lease is None else lease
        if lease_to_retain is None:
            raise RuntimeError("No active lease is available to retain.")
        retained_lease = self._backend.retain_lease(lease_to_retain)
        self._active_lease = retained_lease
        return retained_lease

    def list_leases(self) -> Sequence[LeaseResourceEntry]:
        return tuple(self._backend.list_leases())


class LeaseKeepAlive:
    """Background lease keepalive that mirrors the Spot SDK helper."""

    def __init__(
        self,
        lease_client: LeaseClient,
        *,
        rpc_interval_seconds: float = 0.5,
        must_acquire: bool = False,
        return_at_exit: bool = False,
        keep_running_cb: Callable[[], bool] | None = None,
    ) -> None:
        self._lease_client = lease_client
        self._interval_s = max(float(rpc_interval_seconds), 1.0e-3)
        self._return_at_exit = bool(return_at_exit)
        self._keep_running_cb = keep_running_cb
        if lease_client.active_lease is None:
            if must_acquire:
                lease_client.acquire()
            else:
                raise RuntimeError("LeaseKeepAlive requires an acquired lease.")
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="spot-lease-keepalive", daemon=True)
        self._thread.start()

    @property
    def lease(self) -> Lease:
        active_lease = self._lease_client.active_lease
        if active_lease is None:
            raise RuntimeError("LeaseKeepAlive no longer has an active lease.")
        return active_lease

    def _loop(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            if self._keep_running_cb is not None and not bool(self._keep_running_cb()):
                break
            self._lease_client.retain_lease()

    def shutdown(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        if self._return_at_exit and self._lease_client.active_lease is not None:
            self._lease_client.return_lease()

    def __enter__(self) -> "LeaseKeepAlive":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        self.shutdown()


class RobotControlBackend(Protocol):
    """Robot-level power/auth/state backend."""

    def authenticate(self, username: str, password: str, timeout: float | None = None) -> None: ...

    def power_on(
        self,
        timeout_sec: float = 20.0,
        update_frequency: float = 1.0,
        timeout: float | None = None,
    ) -> None: ...

    def power_off(
        self,
        cut_immediately: bool = False,
        timeout_sec: float = 20.0,
        update_frequency: float = 1.0,
        timeout: float | None = None,
    ) -> None: ...

    def is_powered_on(self, timeout: float | None = None) -> bool: ...

    def is_estopped(self, timeout: float | None = None) -> bool: ...


class InMemoryRobotControlBackend:
    """Simple power/auth backend for simulation and unit tests."""

    def __init__(
        self,
        *,
        client_name: str,
        lease_backend: InMemoryLeaseBackend | None = None,
        require_authentication: bool = True,
    ) -> None:
        self._client_name = client_name
        self._lease_backend = lease_backend
        self._require_authentication = bool(require_authentication)
        self._authenticated = False
        self._powered_on = False
        self._estopped = False

    @property
    def authenticated(self) -> bool:
        return self._authenticated

    def authenticate(self, username: str, password: str, timeout: float | None = None) -> None:
        del timeout
        if not str(username).strip():
            raise RuntimeError("username must be a non-empty string.")
        if not str(password).strip():
            raise RuntimeError("password must be a non-empty string.")
        self._authenticated = True

    def power_on(
        self,
        timeout_sec: float = 20.0,
        update_frequency: float = 1.0,
        timeout: float | None = None,
    ) -> None:
        del timeout_sec, update_frequency, timeout
        if self._require_authentication and not self._authenticated:
            raise RuntimeError("authenticate() must be called before power_on().")
        if self._estopped:
            raise RuntimeError("Robot is estopped and cannot be powered on.")
        if self._lease_backend is not None and not self._lease_backend.has_active_lease(client_name=self._client_name):
            raise RuntimeError("A body lease must be acquired before power_on().")
        self._powered_on = True

    def power_off(
        self,
        cut_immediately: bool = False,
        timeout_sec: float = 20.0,
        update_frequency: float = 1.0,
        timeout: float | None = None,
    ) -> None:
        del cut_immediately, timeout_sec, update_frequency, timeout
        self._powered_on = False

    def is_powered_on(self, timeout: float | None = None) -> bool:
        del timeout
        return self._powered_on

    def is_estopped(self, timeout: float | None = None) -> bool:
        del timeout
        return self._estopped


class RobotStateBackend(Protocol):
    """Backend used by :class:`RobotStateClient`."""

    def get_robot_state(self, **kwargs) -> Any: ...


class RobotStateClient:
    """Spot-SDK-style robot-state client."""

    default_service_name = "robot-state"
    service_type = "bosdyn.api.RobotStateService"

    def __init__(self, backend: RobotStateBackend) -> None:
        self._backend = backend

    def get_robot_state(self, **kwargs) -> Any:
        return self._backend.get_robot_state(**kwargs)


@dataclass(frozen=True)
class ImageSource:
    """Logical image source mounted on the simulated robot."""

    name: str


@dataclass(frozen=True)
class ImageResponse:
    """Response payload for a single image request."""

    source: str
    image: Any


class ImageBackend(Protocol):
    """Backend used by :class:`ImageClient`."""

    def list_image_sources(self) -> Sequence[ImageSource]: ...

    def get_image_from_sources(self, image_sources: Sequence[str]) -> Sequence[ImageResponse]: ...


class ImageClient:
    """Spot-SDK-style image client."""

    default_service_name = "image"
    service_type = "bosdyn.api.ImageService"

    def __init__(self, backend: ImageBackend) -> None:
        self._backend = backend

    def list_image_sources(self) -> Sequence[ImageSource]:
        return tuple(self._backend.list_image_sources())

    def get_image_from_sources(self, image_sources: Sequence[str]) -> Sequence[ImageResponse]:
        return tuple(self._backend.get_image_from_sources(image_sources))


def require_robot_state_ready(state_client: RobotStateClient) -> Any:
    """Return a robot-state snapshot or fail fast if the state service is not ready."""

    state_snapshot = state_client.get_robot_state()
    if state_snapshot is None:
        raise RuntimeError("RobotStateClient returned no state snapshot.")
    return state_snapshot


def require_image_sources_ready(
    image_client: ImageClient,
    camera_names: Sequence[str],
) -> tuple[ImageResponse, ...]:
    """Return image responses after validating requested sources and depth frames."""

    requested_names = tuple(str(name) for name in camera_names)
    image_sources = tuple(source.name for source in image_client.list_image_sources())
    missing_sources = [name for name in requested_names if name not in image_sources]
    if missing_sources:
        raise RuntimeError(f"Missing required image sources: {missing_sources}. Available sources: {image_sources}")

    image_responses = tuple(image_client.get_image_from_sources(requested_names))
    for response in image_responses:
        image = response.image
        depth = image.get("depth") if isinstance(image, Mapping) else None
        if depth is None:
            raise RuntimeError(f"Camera '{response.source}' has no depth frame available after warm-up.")
    return image_responses


class ManipulationFeedbackState(str, Enum):
    """High-level manipulation lifecycle state."""

    UNKNOWN = "unknown"
    PLANNING = "planning"
    EXECUTING = "executing"
    GRASPING = "grasping"
    DONE = "done"
    FAILED = "failed"
    OVERRIDDEN = "overridden"

    @property
    def is_terminal(self) -> bool:
        return self in (ManipulationFeedbackState.DONE, ManipulationFeedbackState.FAILED, ManipulationFeedbackState.OVERRIDDEN)


@dataclass(frozen=True)
class ManipulationApiResponse:
    """Response returned when a manipulation command is accepted."""

    command_id: int
    message: str = ""

    @property
    def manipulation_cmd_id(self) -> int:
        return self.command_id


@dataclass(frozen=True)
class ManipulationApiFeedbackResponse:
    """Feedback for a submitted manipulation command."""

    command_id: int
    state: ManipulationFeedbackState
    message: str = ""

    @property
    def manipulation_cmd_id(self) -> int:
        return self.command_id

    @property
    def current_state(self) -> ManipulationFeedbackState:
        return self.state


@dataclass(frozen=True)
class ManipulationApiFeedbackRequest:
    """Feedback request matching the Spot SDK's command-id request shape."""

    manipulation_cmd_id: int

    @property
    def command_id(self) -> int:
        return self.manipulation_cmd_id


class ManipulationBackend(Protocol):
    """Backend used by :class:`ManipulationApiClient`."""

    def manipulation_api_command(self, manipulation_api_request: Any, **kwargs) -> ManipulationApiResponse: ...

    def manipulation_api_feedback_command(
        self,
        manipulation_api_feedback_request: Any,
        **kwargs,
    ) -> ManipulationApiFeedbackResponse: ...


class ManipulationApiClient:
    """Spot-SDK-style manipulation client."""

    default_service_name = "manipulation"
    service_type = "bosdyn.api.ManipulationApiService"

    def __init__(self, backend: ManipulationBackend) -> None:
        self._backend = backend

    def manipulation_api_command(self, manipulation_api_request: Any, **kwargs) -> ManipulationApiResponse:
        return self._backend.manipulation_api_command(manipulation_api_request, **kwargs)

    def manipulation_api_feedback_command(
        self,
        manipulation_api_feedback_request: Any,
        **kwargs,
    ) -> ManipulationApiFeedbackResponse:
        return self._backend.manipulation_api_feedback_command(manipulation_api_feedback_request, **kwargs)


class Robot:
    """Simulation-backed Spot robot facade."""

    def __init__(self, *, sdk: "Sdk", address: str, name: str) -> None:
        self.sdk = sdk
        self.address = address
        self.name = name
        self.time_sync = TimeSyncThread()
        self._control_backend: RobotControlBackend = InMemoryRobotControlBackend(client_name=name)
        self._service_factories: dict[str, Callable[[], Any]] = {}
        self._client_cache: dict[str, Any] = {}

    def install_control_backend(self, backend: RobotControlBackend) -> None:
        self._control_backend = backend

    def install_service_factory(self, service_name: str, factory: Callable[[], Any]) -> None:
        normalized = _normalize_service_name(service_name)
        self._service_factories[normalized] = factory
        self._client_cache.pop(normalized, None)

    def install_service_client(self, service_name: str, client: Any) -> None:
        self.install_service_factory(service_name, lambda: client)

    def ensure_client(self, service_name: str) -> Any:
        normalized = _normalize_service_name(service_name)
        cached = self._client_cache.get(normalized)
        if cached is not None:
            return cached
        factory = self._service_factories.get(normalized)
        if factory is None:
            available = ", ".join(sorted(self._service_factories)) or "<none>"
            raise KeyError(f"Unknown service '{service_name}'. Available services: {available}.")
        client = factory()
        self._client_cache[normalized] = client
        return client

    def authenticate(self, username: str, password: str, timeout: float | None = None) -> None:
        self._control_backend.authenticate(username, password, timeout=timeout)

    def power_on(
        self,
        timeout_sec: float = 20.0,
        update_frequency: float = 1.0,
        timeout: float | None = None,
    ) -> None:
        self._control_backend.power_on(
            timeout_sec=timeout_sec,
            update_frequency=update_frequency,
            timeout=timeout,
        )

    def power_off(
        self,
        cut_immediately: bool = False,
        timeout_sec: float = 20.0,
        update_frequency: float = 1.0,
        timeout: float | None = None,
    ) -> None:
        self._control_backend.power_off(
            cut_immediately=cut_immediately,
            timeout_sec=timeout_sec,
            update_frequency=update_frequency,
            timeout=timeout,
        )

    def is_powered_on(self, timeout: float | None = None) -> bool:
        return self._control_backend.is_powered_on(timeout=timeout)

    def is_estopped(self, timeout: float | None = None) -> bool:
        return self._control_backend.is_estopped(timeout=timeout)


class Sdk:
    """Tiny Spot-SDK-style registry of robots."""

    def __init__(self, name: str) -> None:
        self.name = str(name or "").strip()
        if not self.name:
            raise ValueError("SDK name must be a non-empty string.")
        self._robots: dict[str, Robot] = {}

    def create_robot(self, address: str, name: str | None = None) -> Robot:
        normalized_address = str(address or "").strip()
        if not normalized_address:
            raise ValueError("Robot address must be a non-empty string.")
        robot = self._robots.get(normalized_address)
        if robot is None:
            robot = Robot(
                sdk=self,
                address=normalized_address,
                name=normalized_address if name is None else str(name),
            )
            self._robots[normalized_address] = robot
        elif name is not None and str(name) != robot.name:
            raise RuntimeError(
                f"Robot '{normalized_address}' already exists with name '{robot.name}', "
                f"cannot recreate it as '{name}'."
            )
        return robot

    def clear_robots(self) -> None:
        self._robots.clear()


def create_standard_sdk(client_name: str) -> Sdk:
    """Create the small simulation-backed Spot SDK facade."""

    return Sdk(client_name)


__all__ = [
    "ImageClient",
    "ImageResponse",
    "ImageSource",
    "InMemoryLeaseBackend",
    "InMemoryRobotControlBackend",
    "Lease",
    "LeaseClient",
    "LeaseKeepAlive",
    "LeaseOwner",
    "LeaseResourceEntry",
    "ManipulationApiClient",
    "ManipulationApiFeedbackRequest",
    "ManipulationApiFeedbackResponse",
    "ManipulationApiResponse",
    "ManipulationFeedbackState",
    "Robot",
    "RobotStateClient",
    "Sdk",
    "TimeSyncEndpoint",
    "TimeSyncThread",
    "create_standard_sdk",
    "require_image_sources_ready",
    "require_robot_state_ready",
]
