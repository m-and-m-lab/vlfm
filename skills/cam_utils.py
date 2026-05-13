"""Skill-agnostic camera, depth, and point-cloud utilities."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import open3d as o3d
import yaml


@dataclass(frozen=True)
class ImageCrop:
    """Pixel crop expressed as removed pixels from each image boundary."""

    left: int = 0
    top: int = 0
    right: int = 0
    bottom: int = 0

    @property
    def is_empty(self) -> bool:
        return self.left == 0 and self.top == 0 and self.right == 0 and self.bottom == 0

    def resolve(self, *, width: int, height: int) -> tuple[int, int, int, int]:
        x0 = int(self.left)
        y0 = int(self.top)
        x1 = int(width) - int(self.right)
        y1 = int(height) - int(self.bottom)
        if x0 < 0 or y0 < 0 or x1 > int(width) or y1 > int(height) or x0 >= x1 or y0 >= y1:
            raise ValueError(f"Invalid image crop {self} for image size width={width}, height={height}.")
        return x0, y0, x1, y1


@dataclass(frozen=True)
class DepthFilterConfig:
    """Depth validity thresholds applied per camera before point-cloud projection."""

    min_depth_m: float = 0.0
    max_depth_m: float = 3.5


@dataclass(frozen=True)
class PointCloudWorldFilterConfig:
    """Reference-frame point-cloud filters applied after RGB-D projection."""

    min_z_m: Optional[float] = None
    bounds_min: Optional[tuple[float, float, float]] = None
    bounds_max: Optional[tuple[float, float, float]] = None
    voxel_size_m: Optional[float] = None

    @property
    def is_empty(self) -> bool:
        return (
            self.min_z_m is None
            and self.bounds_min is None
            and self.bounds_max is None
            and self.voxel_size_m is None
        )


@dataclass(frozen=True)
class PointCloudFusionFrameConfig:
    """Common frame for multi-camera point-cloud fusion."""

    mode: str = "camera"
    name: str = ""
    camera_name: Optional[str] = None
    left_camera_name: Optional[str] = None
    right_camera_name: Optional[str] = None


@dataclass(frozen=True)
class CameraRigCameraConfig:
    """One logical camera source in a reusable camera rig profile."""

    name: str
    prim_path: str
    data_types: tuple[str, ...] = ("distance_to_image_plane", "rgb")
    crop: ImageCrop = field(default_factory=ImageCrop)
    depth_filter: DepthFilterConfig = field(default_factory=DepthFilterConfig)


@dataclass(frozen=True)
class CameraRigConfig:
    """Renderer camera rig config used by benchmark scenes and perception utilities."""

    cameras: tuple[CameraRigCameraConfig, ...]
    grasp_camera_names: tuple[str, ...]
    depth_collision_camera_names: tuple[str, ...]

    @property
    def camera_names(self) -> tuple[str, ...]:
        return tuple(camera.name for camera in self.cameras)

    @property
    def cameras_by_name(self) -> dict[str, CameraRigCameraConfig]:
        return {camera.name: camera for camera in self.cameras}

    @property
    def crops_by_camera(self) -> dict[str, ImageCrop]:
        return {camera.name: camera.crop for camera in self.cameras if not camera.crop.is_empty}

    @property
    def prim_paths_by_camera(self) -> dict[str, str]:
        return {camera.name: camera.prim_path for camera in self.cameras}


@dataclass(frozen=True)
class CameraFrame:
    """Single synchronized RGB-D camera frame in a renderer-agnostic shape."""

    name: str
    depth_image: np.ndarray
    rgb_image: Optional[np.ndarray]
    intrinsics: np.ndarray
    position_w: np.ndarray
    quat_wxyz: np.ndarray
    segmentation: Optional[np.ndarray] = None
    segmentation_name: Optional[str] = None
    label_metadata: Mapping[int, Mapping[str, object]] = field(default_factory=dict)
    depth_valid_mask: Optional[np.ndarray] = None
    position_reference: Optional[np.ndarray] = None
    quat_reference: Optional[np.ndarray] = None
    reference_position_w: Optional[np.ndarray] = None
    reference_quat_wxyz: Optional[np.ndarray] = None


@dataclass(frozen=True)
class CameraPointCloud:
    """Raw point cloud derived from one camera frame."""

    camera_name: str
    points_camera: np.ndarray
    points_world: np.ndarray
    valid_mask: np.ndarray
    intrinsics: np.ndarray
    position_w: np.ndarray
    quat_wxyz: np.ndarray
    points_target: Optional[np.ndarray] = None


@dataclass(frozen=True)
class CameraPipelineConfig:
    """Configuration for reusable camera capture and point-cloud fusion."""

    crops_by_camera: Mapping[str, ImageCrop] = field(default_factory=dict)
    depth_filter: DepthFilterConfig = field(default_factory=DepthFilterConfig)
    pointcloud_filter: PointCloudWorldFilterConfig = field(default_factory=PointCloudWorldFilterConfig)
    include_rgb: bool = True
    include_pointcloud: bool = True
    total_points: int = 16384
    fusion_frame: Optional[PointCloudFusionFrameConfig] = None
    camera_prim_paths: Mapping[str, str] = field(default_factory=dict)
    reference_prim_path: Optional[str] = None
    rng_seed: int = 0


@dataclass(frozen=True)
class CameraPipelineResult:
    """Captured camera frames plus an optional fused point cloud."""

    camera_names: tuple[str, ...]
    frames: tuple[CameraFrame, ...]
    fused_pointcloud: Optional[object] = None


class TargetMaskLookupError(LookupError):
    """Raised when a requested target/object mask cannot be resolved from segmentation."""


DEFAULT_CAMERA_CROPS = {
    "hand": ImageCrop(left=38),
}

DEFAULT_SPOT_ARM_CAMERA_RIG_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "spot_arm_cameras.yaml"


def load_camera_rig_config(path: str | Path = DEFAULT_SPOT_ARM_CAMERA_RIG_CONFIG_PATH) -> CameraRigConfig:
    """Load a skill-agnostic named camera rig config from YAML."""

    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Camera rig config was not found: {config_path}")

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    root = _require_mapping(data, "camera rig root")
    defaults = _optional_mapping(root, "defaults")
    groups = _optional_mapping(root, "camera_groups")
    camera_data = _require_mapping(root.get("cameras"), "cameras")

    default_data_types = _parse_string_tuple(defaults.get("data_types", ("distance_to_image_plane", "rgb")), "defaults.data_types")
    default_depth_filter = _parse_depth_filter(defaults.get("depth_filter", {}), "defaults.depth_filter")

    cameras: list[CameraRigCameraConfig] = []
    for raw_name, raw_config in camera_data.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise TypeError(f"Camera names must be non-empty strings, got {raw_name!r}.")
        item = _require_mapping(raw_config, f"cameras.{raw_name}")
        prim_path = item.get("prim_path")
        if not isinstance(prim_path, str) or not prim_path.strip():
            raise TypeError(f"Expected cameras.{raw_name}.prim_path to be a non-empty string.")
        cameras.append(
            CameraRigCameraConfig(
                name=raw_name.strip(),
                prim_path=prim_path.strip(),
                data_types=_parse_string_tuple(item.get("data_types", default_data_types), f"cameras.{raw_name}.data_types"),
                crop=_parse_image_crop(item.get("crop", {}), f"cameras.{raw_name}.crop"),
                depth_filter=_parse_depth_filter(item.get("depth_filter", default_depth_filter), f"cameras.{raw_name}.depth_filter"),
            )
        )

    if not cameras:
        raise ValueError("Camera rig config must define at least one camera.")
    camera_names = tuple(camera.name for camera in cameras)
    if len(set(camera_names)) != len(camera_names):
        raise ValueError(f"Camera rig camera names must be unique, got {camera_names}.")

    grasp_names = _parse_string_tuple(groups.get("grasp", camera_names), "camera_groups.grasp")
    depth_names = _parse_string_tuple(groups.get("depth_collision", camera_names), "camera_groups.depth_collision")
    missing = sorted((set(grasp_names) | set(depth_names)) - set(camera_names))
    if missing:
        raise ValueError(f"Camera groups reference unknown camera names: {missing}.")

    return CameraRigConfig(
        cameras=tuple(cameras),
        grasp_camera_names=grasp_names,
        depth_collision_camera_names=depth_names,
    )


def to_numpy(value: Any) -> np.ndarray:
    """Convert torch/numpy/scalar inputs into a detached CPU numpy array."""

    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def normalize_prim_path(path: str) -> str:
    """Normalize a prim path for robust comparisons across camera metadata sources."""

    normalized = str(path or "").strip().strip("'").strip('"')
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    if normalized and not normalized.startswith("/"):
        normalized = f"/{normalized}"
    if normalized.endswith("/") and normalized != "/":
        normalized = normalized[:-1]
    return normalized


def squeeze_depth_image(depth_image: Any) -> np.ndarray:
    """Collapse Isaac tensor layouts into a float32 2-D depth image."""

    arr = to_numpy(depth_image).astype(np.float32, copy=False)
    if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[-1] == 1:
        arr = arr[0, ..., 0]
    elif arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D depth image, got shape={tuple(arr.shape)}.")
    return arr.astype(np.float32, copy=False)


def squeeze_rgb_image(rgb_image: Any) -> np.ndarray:
    """Collapse Isaac RGB tensor layouts into a uint8 HWC image."""

    arr = to_numpy(rgb_image)
    while arr.ndim > 3:
        if arr.shape[0] == 1:
            arr = arr[0]
            continue
        break
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim != 3:
        raise ValueError(f"Expected an RGB image with 3 dims, got shape={tuple(arr.shape)}.")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[-1] != 3:
        raise ValueError(f"Expected 1/3/4 RGB channels, got shape={tuple(arr.shape)}.")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32, copy=False)
        if arr.size and float(np.nanmax(arr)) <= 1.0 + 1e-6:
            arr = arr * 255.0
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return arr.copy()


def squeeze_segmentation(segmentation: Any) -> np.ndarray:
    """Collapse renderer segmentation layouts into a 2-D integer image."""

    arr = to_numpy(segmentation)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 4 and arr.dtype == np.uint8:
        raise ValueError("Received a colorized semantic segmentation image; raw integer IDs are required.")
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D segmentation image, got shape={tuple(arr.shape)}.")
    return arr.astype(np.int64, copy=False)


def crop_spatial_image(image: Any, crop: ImageCrop):
    """Crop an image-like tensor/array while preserving leading/channel axes."""

    if image is None or crop is None or crop.is_empty:
        return image
    if not hasattr(image, "shape"):
        raise TypeError(f"Cannot crop object without a shape attribute: {type(image)!r}.")

    shape = tuple(int(dim) for dim in image.shape)
    y_axis, x_axis = _infer_spatial_axes(shape, crop)
    x0, y0, x1, y1 = crop.resolve(width=shape[x_axis], height=shape[y_axis])

    slices = [slice(None)] * len(shape)
    slices[y_axis] = slice(y0, y1)
    slices[x_axis] = slice(x0, x1)
    return image[tuple(slices)]


def crop_camera_intrinsics(intrinsics: Any, crop: ImageCrop) -> np.ndarray:
    """Adjust camera intrinsics so cropped RGB-D pixels back-project correctly."""

    intrinsics_np = to_numpy(intrinsics).astype(np.float32, copy=True)
    if intrinsics_np.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 intrinsic matrix, got shape={tuple(intrinsics_np.shape)}.")
    if crop is None or crop.is_empty:
        return intrinsics_np
    intrinsics_np[0, 2] -= float(crop.left)
    intrinsics_np[1, 2] -= float(crop.top)
    return intrinsics_np


def valid_depth_mask(depth_image: Any, depth_filter: DepthFilterConfig | None = None) -> np.ndarray:
    """Return per-camera valid-depth mask using finite, positive, min, and max tests."""

    depth = squeeze_depth_image(depth_image)
    config = depth_filter or DepthFilterConfig()
    min_depth = float(config.min_depth_m)
    max_depth = float(config.max_depth_m)
    if min_depth < 0.0:
        raise ValueError(f"min_depth_m must be non-negative, got {min_depth}.")
    if max_depth <= min_depth:
        raise ValueError(f"max_depth_m must be greater than min_depth_m, got min={min_depth}, max={max_depth}.")
    return np.isfinite(depth) & (depth > min_depth) & (depth <= max_depth)


def get_rgb_image(camera: Any, crop: ImageCrop | None = None) -> np.ndarray:
    """Acquire one RGB image from a camera wrapper."""

    rgb = _extract_rgb_payload(camera)
    if rgb is None:
        raise ValueError("Camera has no RGB output.")
    rgb_np = squeeze_rgb_image(rgb)
    return crop_spatial_image(rgb_np, crop) if crop is not None else rgb_np


def get_depth_image(
    camera: Any,
    crop: ImageCrop | None = None,
    depth_filter: DepthFilterConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Acquire one depth image and return a thresholded depth image plus valid mask."""

    depth = _extract_depth_payload(camera)
    if depth is None:
        raise ValueError("Camera has no distance_to_image_plane depth output.")
    depth_np = squeeze_depth_image(depth)
    if crop is not None:
        depth_np = crop_spatial_image(depth_np, crop)
    mask = valid_depth_mask(depth_np, depth_filter)
    masked_depth = depth_np.copy()
    masked_depth[~mask] = 0.0
    return masked_depth.astype(np.float32, copy=False), mask.astype(bool, copy=False)


def capture_camera_frame(
    camera: Any,
    name: str,
    crop: ImageCrop | None = None,
    depth_filter: DepthFilterConfig | None = None,
    target_prim_path: str | None = None,
    camera_prim_path: str | None = None,
    reference_prim_path: str | None = None,
) -> CameraFrame:
    """Capture one normalized RGB-D frame from an IsaacLab-like camera wrapper."""

    depth_image, valid_mask = get_depth_image(camera, crop=crop, depth_filter=depth_filter)

    rgb_image = None
    rgb_payload = _extract_rgb_payload(camera)
    if rgb_payload is not None:
        rgb_image = squeeze_rgb_image(rgb_payload)
        if crop is not None:
            rgb_image = crop_spatial_image(rgb_image, crop)

    intrinsics = crop_camera_intrinsics(_extract_intrinsics(camera), crop or ImageCrop())
    position_w, quat_wxyz = _extract_camera_pose(camera)
    reference_pose = _resolve_camera_reference_pose(camera_prim_path, reference_prim_path)
    segmentation_name, segmentation, label_metadata = select_segmentation_candidate(camera, target_prim_path)
    if segmentation is not None:
        segmentation = squeeze_segmentation(segmentation)
        if crop is not None:
            segmentation = crop_spatial_image(segmentation, crop)

    if not np.any(valid_mask):
        raise ValueError(
            f"Camera '{name}' has no valid depth pixels after filtering "
            f"(min={float((depth_filter or DepthFilterConfig()).min_depth_m)}, "
            f"max={float((depth_filter or DepthFilterConfig()).max_depth_m)})."
        )

    return CameraFrame(
        name=str(name),
        depth_image=depth_image,
        rgb_image=rgb_image,
        intrinsics=intrinsics,
        position_w=position_w,
        quat_wxyz=quat_wxyz,
        segmentation=segmentation,
        segmentation_name=segmentation_name,
        label_metadata=label_metadata,
        depth_valid_mask=valid_mask,
        position_reference=None if reference_pose is None else reference_pose[0],
        quat_reference=None if reference_pose is None else reference_pose[1],
        reference_position_w=None if reference_pose is None else reference_pose[2],
        reference_quat_wxyz=None if reference_pose is None else reference_pose[3],
    )


def capture_camera_frames(
    camera_map: Mapping[str, object],
    camera_names: Sequence[str],
    crops_by_camera: Mapping[str, ImageCrop] | None = None,
    depth_filter: DepthFilterConfig | None = None,
    target_prim_path: str | None = None,
    camera_prim_paths: Mapping[str, str] | None = None,
    reference_prim_path: str | None = None,
) -> list[CameraFrame]:
    """Capture normalized frames from a named camera map."""

    frames: list[CameraFrame] = []
    crops = crops_by_camera or {}
    prim_paths = camera_prim_paths or {}
    for camera_name in camera_names:
        name = str(camera_name)
        camera = camera_map.get(name)
        if camera is None:
            raise ValueError(f"Camera '{name}' is not configured.")
        frames.append(
            capture_camera_frame(
                camera,
                name,
                crop=crops.get(name),
                depth_filter=depth_filter,
                target_prim_path=target_prim_path,
                camera_prim_path=prim_paths.get(name),
                reference_prim_path=reference_prim_path,
            )
        )
    return frames


def depth_to_pointcloud(
    depth_image: Any,
    intrinsics: Any,
    mask: np.ndarray | None = None,
    depth_filter: DepthFilterConfig | None = None,
) -> np.ndarray:
    """Back-project a depth image into the camera frame."""

    depth = squeeze_depth_image(depth_image)
    k = to_numpy(intrinsics).astype(np.float64, copy=False)
    if k.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 intrinsic matrix, got shape={tuple(k.shape)}.")
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    if fx == 0.0 or fy == 0.0:
        raise ValueError("Camera intrinsics must have non-zero fx and fy.")

    valid = valid_depth_mask(depth, depth_filter)
    if mask is not None:
        mask_np = np.asarray(mask, dtype=bool)
        if mask_np.shape != depth.shape:
            raise ValueError(f"Point-cloud mask shape {mask_np.shape} does not match depth shape {depth.shape}.")
        valid &= mask_np
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    depth_for_projection = np.ascontiguousarray(np.where(valid, depth, 0.0).astype(np.float32, copy=False))
    height, width = depth_for_projection.shape
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        int(width),
        int(height),
        fx,
        fy,
        float(k[0, 2]),
        float(k[1, 2]),
    )
    cloud = o3d.geometry.PointCloud.create_from_depth_image(
        o3d.geometry.Image(depth_for_projection),
        intrinsic,
        depth_scale=1.0,
        depth_trunc=float((depth_filter or DepthFilterConfig()).max_depth_m),
        stride=1,
        project_valid_depth_only=True,
    )
    return np.asarray(cloud.points, dtype=np.float32).reshape(-1, 3)


def capture_pointclouds(
    camera_map: Mapping[str, object],
    camera_names: Sequence[str],
    crops_by_camera: Mapping[str, ImageCrop] | None = None,
    depth_filter: DepthFilterConfig | None = None,
    target_frame: object | None = None,
) -> list[CameraPointCloud]:
    """Capture raw point clouds from one or more cameras."""

    frames = capture_camera_frames(
        camera_map,
        camera_names,
        crops_by_camera=crops_by_camera,
        depth_filter=depth_filter,
    )
    target_pose = _coerce_target_frame(target_frame)
    clouds: list[CameraPointCloud] = []
    for frame in frames:
        points_camera = depth_to_pointcloud(
            frame.depth_image,
            frame.intrinsics,
            mask=frame.depth_valid_mask,
            depth_filter=depth_filter,
        )
        points_world = transform_points_to_world(points_camera, frame.position_w, frame.quat_wxyz)
        points_target = None
        if target_pose is not None:
            target_position_w, target_quat_wxyz = target_pose
            points_target = transform_points_from_world(points_world, target_position_w, target_quat_wxyz)
        clouds.append(
            CameraPointCloud(
                camera_name=frame.name,
                points_camera=points_camera,
                points_world=points_world,
                valid_mask=np.asarray(frame.depth_valid_mask, dtype=bool),
                intrinsics=frame.intrinsics,
                position_w=frame.position_w,
                quat_wxyz=frame.quat_wxyz,
                points_target=points_target,
            )
        )
    return clouds


def run_camera_pipeline(
    camera_map: Mapping[str, object],
    camera_names: Sequence[str],
    config: CameraPipelineConfig | None = None,
) -> CameraPipelineResult:
    """Capture normalized camera frames and optionally return a fused point cloud."""

    cfg = config or CameraPipelineConfig()
    names = tuple(str(name) for name in camera_names)
    if not names:
        raise ValueError("camera_names must contain at least one camera name.")

    frames = capture_camera_frames(
        camera_map,
        names,
        crops_by_camera=cfg.crops_by_camera,
        depth_filter=cfg.depth_filter,
        camera_prim_paths=cfg.camera_prim_paths,
        reference_prim_path=cfg.reference_prim_path,
    )
    if not cfg.include_rgb:
        frames = [replace(frame, rgb_image=None) for frame in frames]

    fused = None
    if cfg.include_pointcloud:
        from helpers.pcd_utils import fuse_camera_frames

        fused = fuse_camera_frames(
            frames,
            camera_names=names,
            total_points=cfg.total_points,
            fusion_frame=cfg.fusion_frame,
            rng_seed=cfg.rng_seed,
            depth_filter=cfg.depth_filter,
            pointcloud_filter=cfg.pointcloud_filter,
        )

    return CameraPipelineResult(
        camera_names=names,
        frames=tuple(frames),
        fused_pointcloud=fused,
    )


def capture_camera_payload(camera: Any, *, camera_axes: str = "usd") -> dict[str, Any]:
    """Capture an SDK-friendly raw camera payload without normalizing image arrays."""

    output = _get_output(camera)
    depth = output.get("distance_to_image_plane")
    rgba = _extract_rgb_payload(camera)

    segmentation = None
    label_metadata = None
    segmentation_payload = output.get("semantic_segmentation")
    if isinstance(segmentation_payload, Mapping):
        segmentation = segmentation_payload.get("data")
        info = segmentation_payload.get("info") or {}
        raw_labels = info.get("idToLabels") or {}
        if isinstance(raw_labels, Mapping):
            label_metadata = {int(label_id): payload for label_id, payload in raw_labels.items() if str(label_id).isdigit()}

    try:
        position, quat_wxyz = _extract_camera_pose(camera, camera_axes=camera_axes)
    except ValueError:
        position, quat_wxyz = None, None

    return {
        "rgba": _to_python_payload(rgba),
        "depth": _to_python_payload(depth),
        "segmentation": _to_python_payload(segmentation),
        "label_metadata": label_metadata,
        "position": _to_python_payload(position),
        "quat_wxyz": _to_python_payload(quat_wxyz),
    }


def quat_wxyz_to_matrix(quat_wxyz: Any) -> np.ndarray:
    """Convert a quaternion in wxyz order to a 3x3 rotation matrix."""

    quat = to_numpy(quat_wxyz).astype(np.float64, copy=False).reshape(4)
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        raise ValueError("Quaternion norm must be non-zero.")
    w, x, y, z = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_wxyz(rotation_matrix: Any) -> np.ndarray:
    """Convert a 3x3 rotation matrix into wxyz quaternion order."""

    m = to_numpy(rotation_matrix).astype(np.float64, copy=False).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = np.asarray([w, x, y, z], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return quat.astype(np.float32)


def transform_points_to_world(points: Any, source_position_w: Any, source_quat_wxyz: Any) -> np.ndarray:
    """Transform points from a source frame into world frame."""

    points_np = to_numpy(points).astype(np.float64, copy=False).reshape(-1, 3)
    rot = quat_wxyz_to_matrix(source_quat_wxyz)
    trans = to_numpy(source_position_w).astype(np.float64, copy=False).reshape(3)
    return (points_np @ rot.T + trans).astype(np.float32, copy=False)


def transform_points_from_world(points_world: Any, target_position_w: Any, target_quat_wxyz: Any) -> np.ndarray:
    """Transform world-frame points into a target frame."""

    points = to_numpy(points_world).astype(np.float64, copy=False).reshape(-1, 3)
    rot = quat_wxyz_to_matrix(target_quat_wxyz)
    trans = to_numpy(target_position_w).astype(np.float64, copy=False).reshape(3)
    return ((points - trans) @ rot).astype(np.float32, copy=False)


def transform_points_between_frames(
    points: Any,
    source_position_w: Any,
    source_quat_wxyz: Any,
    target_position_w: Any,
    target_quat_wxyz: Any,
) -> np.ndarray:
    """Transform points directly from one world-posed frame into another."""

    points_world = transform_points_to_world(points, source_position_w, source_quat_wxyz)
    return transform_points_from_world(points_world, target_position_w, target_quat_wxyz)


def project_camera_points_to_pixels(points_cam: Any, intrinsics: Any) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-frame 3D points into image pixels using the given intrinsics."""

    points = to_numpy(points_cam).astype(np.float64, copy=False).reshape(-1, 3)
    k = to_numpy(intrinsics).astype(np.float64, copy=False).reshape(3, 3)
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > 1e-6)
    pixels = np.full((points.shape[0], 2), np.nan, dtype=np.float32)
    if np.any(valid):
        pixels[valid, 0] = points[valid, 0] * float(k[0, 0]) / z[valid] + float(k[0, 2])
        pixels[valid, 1] = points[valid, 1] * float(k[1, 1]) / z[valid] + float(k[1, 2])
    return pixels, valid


def select_segmentation_candidate(
    camera: Any,
    target_prim_path: str | None = None,
) -> tuple[Optional[str], Optional[object], Mapping[int, Mapping[str, object]]]:
    """Select the best raw segmentation output and normalized metadata."""

    cam_data = getattr(camera, "data", None)
    if cam_data is None:
        return None, None, {}

    fallback_name: Optional[str] = None
    fallback_segmentation = None
    fallback_metadata: Mapping[int, Mapping[str, object]] = {}
    for candidate_name, segmentation_output in _iter_segmentation_outputs(cam_data):
        segmentation = segmentation_output.get("data") if isinstance(segmentation_output, dict) else segmentation_output
        label_metadata = _extract_label_metadata(
            segmentation_output,
            cam_data,
            preferred_info_keys=_segmentation_info_key_aliases(candidate_name),
        )
        if fallback_name is None:
            fallback_name = candidate_name
            fallback_segmentation = segmentation
            fallback_metadata = label_metadata
        if not target_prim_path:
            return candidate_name, segmentation, label_metadata
        if segmentation is None or not label_metadata:
            continue
        if _segmentation_metadata_matches_target(label_metadata, target_prim_path):
            return candidate_name, segmentation, label_metadata

    return fallback_name, fallback_segmentation, fallback_metadata


def resolve_target_mask(
    segmentation: Any,
    label_metadata: Mapping[int, Mapping[str, object]],
    target_prim_path: str,
) -> np.ndarray:
    """Resolve a boolean segmentation mask for one target/object prim path."""

    target = normalize_prim_path(target_prim_path)
    if not target:
        raise ValueError("target_prim_path must be a non-empty prim path.")

    seg = squeeze_segmentation(segmentation)
    if not label_metadata:
        raise TargetMaskLookupError(
            f"Target '{target}' could not be resolved because segmentation metadata is unavailable."
        )

    matched_ids: list[int] = []
    for raw_id, metadata in label_metadata.items():
        try:
            label_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if _label_metadata_matches_target(metadata, target):
            matched_ids.append(label_id)

    if not matched_ids:
        raise TargetMaskLookupError(
            f"Target '{target}' is not present in the available segmentation metadata."
        )

    return np.isin(seg, np.asarray(matched_ids, dtype=seg.dtype))


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected {label} to be a mapping, got {type(value).__name__}.")
    return value


def _optional_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key not in value:
        return {}
    return _require_mapping(value[key], key)


def _parse_string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"Expected {label} to be a non-empty string sequence, got {value!r}.")
    parsed: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise TypeError(f"Expected every item in {label} to be a non-empty string, got {item!r}.")
        parsed.append(item.strip())
    return tuple(parsed)


def _parse_image_crop(value: object, label: str) -> ImageCrop:
    if isinstance(value, ImageCrop):
        return value
    mapping = _require_mapping(value or {}, label)
    return ImageCrop(
        left=_parse_nonnegative_int(mapping.get("left", 0), f"{label}.left"),
        top=_parse_nonnegative_int(mapping.get("top", 0), f"{label}.top"),
        right=_parse_nonnegative_int(mapping.get("right", 0), f"{label}.right"),
        bottom=_parse_nonnegative_int(mapping.get("bottom", 0), f"{label}.bottom"),
    )


def _parse_depth_filter(value: object, label: str) -> DepthFilterConfig:
    if isinstance(value, DepthFilterConfig):
        return value
    mapping = _require_mapping(value or {}, label)
    config = DepthFilterConfig(
        min_depth_m=float(mapping.get("min_depth_m", DepthFilterConfig().min_depth_m)),
        max_depth_m=float(mapping.get("max_depth_m", DepthFilterConfig().max_depth_m)),
    )
    if config.min_depth_m < 0.0:
        raise ValueError(f"{label}.min_depth_m must be non-negative, got {config.min_depth_m}.")
    if config.max_depth_m <= config.min_depth_m:
        raise ValueError(
            f"{label}.max_depth_m must be greater than min_depth_m, "
            f"got min={config.min_depth_m}, max={config.max_depth_m}."
        )
    return config


def _parse_nonnegative_int(value: object, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Expected {label} to be an integer, got {value!r}.") from exc
    if parsed < 0:
        raise ValueError(f"Expected {label} to be non-negative, got {parsed}.")
    return parsed


def _infer_spatial_axes(shape: tuple[int, ...], crop: ImageCrop) -> tuple[int, int]:
    if len(shape) < 2:
        raise ValueError(f"Expected image-like rank >= 2, got shape={shape}.")
    candidates: list[tuple[int, int]] = []
    if len(shape) == 2:
        candidates.append((0, 1))
    else:
        if shape[-1] <= 4:
            candidates.append((len(shape) - 3, len(shape) - 2))
        candidates.append((len(shape) - 2, len(shape) - 1))
    for y_axis, x_axis in candidates:
        if shape[y_axis] > crop.top + crop.bottom and shape[x_axis] > crop.left + crop.right:
            return y_axis, x_axis
    raise ValueError(f"Could not infer crop axes for shape={shape} and crop={crop}.")


def _get_output(camera: Any) -> Mapping[str, Any]:
    cam_data = getattr(camera, "data", None)
    output = getattr(cam_data, "output", None)
    if isinstance(output, Mapping):
        return output
    if hasattr(camera, "get_current_frame"):
        frame = camera.get_current_frame() or {}
        if isinstance(frame, Mapping):
            return frame
    return {}


def _extract_depth_payload(camera: Any):
    return _get_output(camera).get("distance_to_image_plane")


def _extract_rgb_payload(camera: Any):
    output = _get_output(camera)
    rgb = output.get("rgb")
    if isinstance(rgb, Mapping):
        rgb = rgb.get("data")
    if rgb is not None:
        return rgb
    if hasattr(camera, "get_rgba"):
        try:
            return camera.get_rgba()
        except Exception:
            return None
    return None


def _extract_intrinsics(camera: Any) -> np.ndarray:
    cam_data = getattr(camera, "data", None)
    intrinsics = getattr(cam_data, "intrinsic_matrices", None)
    if intrinsics is None:
        raise ValueError("Camera has no intrinsic_matrices data.")
    intrinsics_np = to_numpy(intrinsics)
    if intrinsics_np.ndim == 3:
        intrinsics_np = intrinsics_np[0]
    if intrinsics_np.shape != (3, 3):
        raise ValueError(f"Expected camera intrinsics shape (3, 3), got {tuple(intrinsics_np.shape)}.")
    return intrinsics_np.astype(np.float32, copy=False)


def _extract_camera_pose(camera: Any, *, camera_axes: str = "ros") -> tuple[np.ndarray, np.ndarray]:
    cam_data = getattr(camera, "data", None)
    if cam_data is not None:
        pos_w = getattr(cam_data, "pos_w", None)
        quat_w_ros = getattr(cam_data, "quat_w_ros", None)
        if pos_w is not None and quat_w_ros is not None:
            return (
                to_numpy(pos_w[0]).astype(np.float32, copy=False).reshape(3),
                to_numpy(quat_w_ros[0]).astype(np.float32, copy=False).reshape(4),
            )
    if hasattr(camera, "get_world_pose"):
        try:
            pos_w, quat_wxyz = camera.get_world_pose(camera_axes=camera_axes)
        except TypeError:
            pos_w, quat_wxyz = camera.get_world_pose()
        return (
            to_numpy(pos_w).astype(np.float32, copy=False).reshape(3),
            to_numpy(quat_wxyz).astype(np.float32, copy=False).reshape(4),
        )
    raise ValueError("Camera has no world pose data.")


def _resolve_camera_reference_pose(
    camera_prim_path: str | None,
    reference_prim_path: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if not camera_prim_path and not reference_prim_path:
        return None
    if not camera_prim_path or not reference_prim_path:
        raise ValueError("camera_prim_path and reference_prim_path must be configured together.")

    import isaaclab.sim as sim_utils

    camera_prim = _resolve_single_prim(camera_prim_path)
    reference_prim = _resolve_single_prim(reference_prim_path)
    position_reference, quat_reference_opengl = sim_utils.resolve_prim_pose(camera_prim, reference_prim)
    reference_position_w, reference_quat_wxyz = sim_utils.resolve_prim_pose(reference_prim)
    return (
        to_numpy(position_reference).astype(np.float32, copy=False).reshape(3),
        _camera_opengl_quat_to_ros(quat_reference_opengl),
        to_numpy(reference_position_w).astype(np.float32, copy=False).reshape(3),
        to_numpy(reference_quat_wxyz).astype(np.float32, copy=False).reshape(4),
    )


def _resolve_single_prim(prim_path_expr: str):
    import isaaclab.sim as sim_utils

    prim_expr = str(prim_path_expr).format(ENV_REGEX_NS="/World/envs/env_.*")
    prims = sim_utils.find_matching_prims(prim_expr)
    if len(prims) != 1:
        raise ValueError(f"Expected exactly one prim for {prim_path_expr!r}, found {len(prims)}.")
    return prims[0]


def _camera_opengl_quat_to_ros(quat_wxyz: Any) -> np.ndarray:
    rotation = quat_wxyz_to_matrix(quat_wxyz)
    rotation[:, 1] *= -1.0
    rotation[:, 2] *= -1.0
    return matrix_to_quat_wxyz(rotation)


def _to_python_payload(value: Any) -> Any:
    if value is None:
        return None
    arr = to_numpy(value)
    return arr.tolist() if isinstance(arr, np.ndarray) else arr


def _iter_segmentation_outputs(cam_data: Any):
    output = getattr(cam_data, "output", {}) or {}
    for candidate in (
        "instance_id_segmentation_fast",
        "instance_id_segmentation",
        "instance_segmentation_fast",
        "instance_segmentation",
        "semantic_segmentation",
    ):
        segmentation_output = output.get(candidate)
        if segmentation_output is not None:
            yield candidate, segmentation_output


def _segmentation_info_key_aliases(candidate_name: str) -> tuple[str, ...]:
    alias_map = {
        "instance_id_segmentation_fast": ("instance_id_segmentation_fast", "instanceIdSegmentationFast"),
        "instance_id_segmentation": ("instance_id_segmentation", "instanceIdSegmentation"),
        "instance_segmentation_fast": ("instance_segmentation_fast", "instanceSegmentationFast"),
        "instance_segmentation": ("instance_segmentation", "instanceSegmentation"),
        "semantic_segmentation": ("semantic_segmentation", "semanticSegmentation"),
    }
    return alias_map.get(candidate_name, (candidate_name,))


def _extract_label_metadata(
    segmentation_output: Any,
    cam_data: Any,
    preferred_info_keys: Optional[Sequence[str]] = None,
) -> dict[int, dict[str, object]]:
    info_candidates = []
    if isinstance(segmentation_output, Mapping):
        info_candidates.append(segmentation_output.get("info") or {})

    ordered_info_keys: list[str] = []
    for key in preferred_info_keys or ():
        if key not in ordered_info_keys:
            ordered_info_keys.append(str(key))
    for key in (
        "instance_id_segmentation_fast",
        "instanceIdSegmentationFast",
        "instance_id_segmentation",
        "instanceIdSegmentation",
        "instance_segmentation_fast",
        "instanceSegmentationFast",
        "instance_segmentation",
        "instanceSegmentation",
        "semantic_segmentation",
        "semanticSegmentation",
    ):
        if key not in ordered_info_keys:
            ordered_info_keys.append(key)

    def append_info_candidates(container) -> None:
        if isinstance(container, Mapping):
            for key in ordered_info_keys:
                info_candidates.append(container.get(key) or {})
            info_candidates.append(container)
        elif isinstance(container, (list, tuple)):
            for item in container:
                append_info_candidates(item)

    for attr_name in ("info", "infos", "output_info"):
        append_info_candidates(getattr(cam_data, attr_name, None))

    for info in info_candidates:
        labels = info.get("idToLabels") or info.get("id_to_labels") or {}
        semantics = info.get("idToSemantics") or info.get("id_to_semantics") or {}
        if isinstance(labels, str):
            try:
                labels = json.loads(labels)
            except json.JSONDecodeError:
                labels = {}
        if isinstance(semantics, str):
            try:
                semantics = json.loads(semantics)
            except json.JSONDecodeError:
                semantics = {}
        if not isinstance(labels, Mapping):
            labels = {}
        if not isinstance(semantics, Mapping):
            semantics = {}
        normalized: dict[int, dict[str, object]] = {}
        source_items = labels.items() if labels else semantics.items()
        for raw_key, raw_value in source_items:
            try:
                label_id = int(raw_key)
            except (TypeError, ValueError):
                continue
            payload = dict(raw_value) if isinstance(raw_value, Mapping) else {"label": raw_value}
            semantic_value = semantics.get(raw_key) or semantics.get(str(raw_key))
            if semantic_value is not None:
                payload.setdefault("semantics", semantic_value)
            normalized[label_id] = payload
        if normalized:
            return normalized
    return {}


def _segmentation_metadata_matches_target(
    label_metadata: Mapping[int, Mapping[str, object]],
    target_prim_path: str,
) -> bool:
    target_parts = _path_parts(target_prim_path)
    for metadata in label_metadata.values():
        for raw in _iter_metadata_strings(metadata):
            candidate_parts = _path_parts(raw)
            if _tuple_has_subsequence(candidate_parts, target_parts) or _tuple_has_subsequence(target_parts, candidate_parts):
                return True
            stripped_candidate = _strip_env_namespace(candidate_parts)
            stripped_target = _strip_env_namespace(target_parts)
            if _tuple_has_subsequence(stripped_candidate, stripped_target) or _tuple_has_subsequence(stripped_target, stripped_candidate):
                return True
    return False


def _label_metadata_matches_target(metadata: Mapping[str, object], target_prim_path: str) -> bool:
    target = normalize_prim_path(target_prim_path)
    if not target:
        return False

    candidates: list[str] = []
    for raw in _iter_metadata_strings(metadata):
        normalized = normalize_prim_path(raw)
        if normalized:
            candidates.append(normalized)

    if not candidates:
        return False

    target_parts = _path_parts(target)
    stripped_target = _strip_env_namespace(target_parts)
    for candidate in candidates:
        candidate_parts = _path_parts(candidate)
        stripped_candidate = _strip_env_namespace(candidate_parts)
        if _tuple_has_subsequence(candidate_parts, target_parts) or _tuple_has_subsequence(target_parts, candidate_parts):
            return True
        if _tuple_has_subsequence(stripped_candidate, stripped_target) or _tuple_has_subsequence(stripped_target, stripped_candidate):
            return True
    return False


def _iter_metadata_strings(value: Any):
    if value is None:
        return
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, Mapping):
        for nested in value.values():
            yield from _iter_metadata_strings(nested)
        return
    if isinstance(value, (list, tuple, set)):
        for nested in value:
            yield from _iter_metadata_strings(nested)


def _path_parts(path: str) -> tuple[str, ...]:
    normalized = normalize_prim_path(path)
    return tuple(part for part in normalized.split("/") if part)


def _strip_env_namespace(parts: Sequence[str]) -> tuple[str, ...]:
    stripped: list[str] = []
    idx = 0
    while idx < len(parts):
        if parts[idx] == "envs" and idx + 1 < len(parts) and parts[idx + 1].startswith("env_"):
            idx += 2
            continue
        stripped.append(parts[idx])
        idx += 1
    return tuple(stripped)


def _tuple_has_subsequence(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    if not needle or len(haystack) < len(needle):
        return False
    for start in range(len(haystack) - len(needle) + 1):
        if tuple(haystack[start : start + len(needle)]) == tuple(needle):
            return True
    return False


def _coerce_target_frame(target_frame: object | None) -> tuple[np.ndarray, np.ndarray] | None:
    if target_frame is None:
        return None
    if isinstance(target_frame, Mapping):
        return (
            to_numpy(target_frame["position_w"]).astype(np.float32, copy=False).reshape(3),
            to_numpy(target_frame["quat_wxyz"]).astype(np.float32, copy=False).reshape(4),
        )
    if isinstance(target_frame, CameraFrame):
        return target_frame.position_w, target_frame.quat_wxyz
    position = getattr(target_frame, "position_w", None)
    quat = getattr(target_frame, "quat_wxyz", None)
    if position is not None and quat is not None:
        return (
            to_numpy(position).astype(np.float32, copy=False).reshape(3),
            to_numpy(quat).astype(np.float32, copy=False).reshape(4),
        )
    if isinstance(target_frame, (list, tuple)) and len(target_frame) == 2:
        return (
            to_numpy(target_frame[0]).astype(np.float32, copy=False).reshape(3),
            to_numpy(target_frame[1]).astype(np.float32, copy=False).reshape(4),
        )
    raise TypeError("target_frame must be None, CameraFrame, mapping, pose object, or (position_w, quat_wxyz).")


def depth_preview_rgb(depth_image: Any) -> np.ndarray:
    """Render a compact RGB preview for one depth image."""

    depth = squeeze_depth_image(depth_image)
    valid = np.isfinite(depth) & (depth > 0.0)
    gray = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        values = depth[valid]
        low = float(np.percentile(values, 5.0))
        high = float(np.percentile(values, 95.0))
        if high <= low + 1e-6:
            high = low + 1e-3
        normalized = np.clip((depth - low) / (high - low), 0.0, 1.0)
        gray = np.clip((1.0 - normalized) * 255.0, 0.0, 255.0).astype(np.uint8)
        gray[~valid] = 0
    return np.repeat(gray[..., None], 3, axis=2)


__all__ = [
    "CameraFrame",
    "CameraPipelineConfig",
    "CameraPipelineResult",
    "CameraPointCloud",
    "CameraRigCameraConfig",
    "CameraRigConfig",
    "DEFAULT_CAMERA_CROPS",
    "DEFAULT_SPOT_ARM_CAMERA_RIG_CONFIG_PATH",
    "DepthFilterConfig",
    "ImageCrop",
    "PointCloudFusionFrameConfig",
    "PointCloudWorldFilterConfig",
    "capture_camera_frame",
    "capture_camera_frames",
    "capture_camera_payload",
    "capture_pointclouds",
    "crop_camera_intrinsics",
    "crop_spatial_image",
    "depth_preview_rgb",
    "depth_to_pointcloud",
    "get_depth_image",
    "get_rgb_image",
    "load_camera_rig_config",
    "matrix_to_quat_wxyz",
    "normalize_prim_path",
    "project_camera_points_to_pixels",
    "quat_wxyz_to_matrix",
    "resolve_target_mask",
    "run_camera_pipeline",
    "select_segmentation_candidate",
    "squeeze_depth_image",
    "squeeze_rgb_image",
    "squeeze_segmentation",
    "TargetMaskLookupError",
    "to_numpy",
    "transform_points_between_frames",
    "transform_points_from_world",
    "transform_points_to_world",
    "valid_depth_mask",
]