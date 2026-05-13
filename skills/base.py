"""Base skill interfaces and shared helpers for interactive-search skills."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from skills.core import SkillCommand, SkillStatus, SkillTriggerResult
from skills.cam_utils import capture_camera_payload


def _noop(_: str) -> None:
    return None


@dataclass(frozen=True)
class CameraCapture:
    """Captured frame data from a camera sensor."""

    rgba: Optional[Any] = None
    depth: Optional[Any] = None
    semantic_mask: Optional[Any] = None
    label_to_id: Optional[dict[str, int]] = None
    position: Optional[Any] = None
    quat_wxyz: Optional[Any] = None
    timestamp: Optional[float] = None


class BaseSkill:
    """Common skill interface with config loading and camera capture helpers."""

    def __init__(
        self,
        *,
        name: str,
        logger: Optional[Callable[[str], None]] = None,
        rgb_camera: Optional[Any] = None,
        depth_camera: Optional[Any] = None,
        active_statuses: Optional[Sequence[SkillStatus]] = None,
    ) -> None:
        self._name = name
        self._log = logger or _noop
        self._status = SkillStatus.IDLE
        self._active_statuses = tuple(active_statuses) if active_statuses is not None else (
            SkillStatus.EXECUTING,
            SkillStatus.GRASPING,
        )
        self._rgb_camera = rgb_camera
        self._depth_camera = depth_camera

    @property
    def name(self) -> str:
        return self._name

    @property
    def status(self) -> SkillStatus:
        return self._status

    @property
    def is_active(self) -> bool:
        return self._status in self._active_statuses

    def set_status(self, status: SkillStatus) -> None:
        self._status = status

    def log(self, message: str) -> None:
        self._log(message)

    def reset(self) -> None:
        self._status = SkillStatus.IDLE
        self.on_reset()

    def on_reset(self) -> None:
        pass

    def set_cameras(
        self,
        *,
        rgb_camera: Optional[Any] = None,
        depth_camera: Optional[Any] = None,
    ) -> None:
        if rgb_camera is not None:
            self._rgb_camera = rgb_camera
        if depth_camera is not None:
            self._depth_camera = depth_camera

    def capture_rgbd(self) -> dict[str, CameraCapture]:
        captures: dict[str, CameraCapture] = {}
        if self._rgb_camera is not None:
            captures["rgb"] = self._capture_camera(self._rgb_camera)
        if self._depth_camera is not None and self._depth_camera is not self._rgb_camera:
            captures["depth"] = self._capture_camera(self._depth_camera)
        return captures

    def _capture_camera(self, camera: Any) -> CameraCapture:
        payload = capture_camera_payload(camera, camera_axes="usd")
        label_metadata = payload.get("label_metadata") or {}
        label_to_id = {
            value["class"]: int(key)
            for key, value in label_metadata.items()
            if isinstance(value, dict) and value.get("class") is not None
        }

        return CameraCapture(
            rgba=payload.get("rgba"),
            depth=payload.get("depth"),
            semantic_mask=payload.get("segmentation"),
            label_to_id=label_to_id,
            position=payload.get("position"),
            quat_wxyz=payload.get("quat_wxyz"),
            timestamp=time.time(),
        )

    def trigger(self, *args, **kwargs) -> SkillTriggerResult:
        raise NotImplementedError(f"{self.__class__.__name__}.trigger is not implemented.")

    def step(self, *args, **kwargs) -> SkillCommand:
        raise NotImplementedError(f"{self.__class__.__name__}.step is not implemented.")

    @classmethod
    def load_config(cls, config_path: str | Path) -> dict[str, Any]:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("PyYAML is required to load YAML configs.") from exc
            return yaml.safe_load(path.read_text()) or {}
        if path.suffix == ".json":
            return json.loads(path.read_text())
        raise ValueError(f"Unsupported config extension: {path.suffix}")

    @classmethod
    def build_from_config(cls, config: Mapping[str, Any], **kwargs) -> "BaseSkill":
        raise NotImplementedError(f"{cls.__name__}.build_from_config is not implemented.")

    @classmethod
    def load_from_path(cls, config_path: str | Path, **kwargs) -> "BaseSkill":
        config = cls.load_config(config_path)
        return cls.build_from_config(config, **kwargs)
