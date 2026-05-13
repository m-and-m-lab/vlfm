"""Small command wrapper for the current Spot flat-terrain policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, Sequence

if TYPE_CHECKING:
    from .spot_policy import SpotFlatTerrainPolicy


class _SpotLocomotionPolicy(Protocol):
    """Current Spot policy surface used by the locomotion client."""

    control_dt: float

    def forward(self, dt: float, command: Sequence[float]) -> None: ...

    def stand(self) -> None: ...


@dataclass(slots=True)
class SpotLocomotionClient:
    """Minimal user-facing locomotion surface for ``spot_policy.py``."""

    policy: _SpotLocomotionPolicy | "SpotFlatTerrainPolicy"
    _command: tuple[float, float, float] = field(init=False, default=(0.0, 0.0, 0.0))

    @classmethod
    def create(
        cls,
        *,
        robot,
        policy_path: str | None = None,
        env_config_path: str | None = None,
    ) -> "SpotLocomotionClient":
        from .spot_policy import SpotFlatTerrainPolicy

        policy = SpotFlatTerrainPolicy(policy_path=policy_path, env_config_path=env_config_path)
        policy.initialize(robot)
        return cls(policy=policy)

    @classmethod
    def default_control_dt(cls) -> float:
        from .spot_policy import SpotFlatTerrainPolicy

        return SpotFlatTerrainPolicy.default_control_dt()

    def command_velocity(
        self,
        v_x: float,
        v_y: float,
        v_rot: float,
    ) -> None:
        self._command = (float(v_x), float(v_y), float(v_rot))

    def stop(self) -> None:
        self.stand()

    def stand(self) -> None:
        self._command = (0.0, 0.0, 0.0)
        self.policy.stand()

    def step(self) -> None:
        self.policy.forward(self.control_dt, self._command)

    @property
    def control_dt(self) -> float:
        return float(self.policy.control_dt)

    @property
    def mode(self) -> str:
        return "stand" if self._command == (0.0, 0.0, 0.0) else "velocity"

    @property
    def current_command(self) -> tuple[float, float, float]:
        return self._command
