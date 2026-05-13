# Copyright (c) 2026 M&M Lab. All rights reserved.

"""IsaacSim policy mixin + concrete policy class.

`IsaacSimMixin` is the IsaacSim analogue of `vlfm.policy.reality_policies.RealityMixin`.
The only behavioural difference is that the init-phase sweep emits **body-yaw
targets** in episodic frame instead of arm-joint angles. The packet structure
emitted by `act()` (the same ``{linear, angular, arm_yaw, info}`` dict) is
identical, so the IsaacSim env wrapper can interpret the init-phase action as a
target body-yaw without any change to the parent policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from torch import Tensor

from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.policy.base_objectnav_policy import VLFMConfig
from vlfm.policy.itm_policy import ITMPolicyV2

# Eight evenly-spaced absolute headings (radians, episodic frame). After the
# robot executes each one in sequence, it has spun a full revolution in place
# and the value/object maps are seeded from every direction. Chosen to match
# the count of reality's arm-yaw sweep (`INITIAL_ARM_YAWS`, 8 entries).
INITIAL_BODY_YAWS: List[float] = np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]).tolist()


class IsaacSimMixin:
    """Mixin layered onto ``ITMPolicyV2`` for IsaacSim deployment.

    See `vlfm.policy.reality_policies.RealityMixin` for the analogous reality
    version — this is a near-mirror with two changes:

    1. ``_initial_yaws`` is seeded with body-yaw targets (episodic frame)
       rather than arm-joint angles. The values themselves carry no special
       meaning beyond "what the env should make the robot face next".
    2. No ZoeDepth instantiation — the hand camera in IsaacSim is a real RGB-D
       sensor; ``_infer_depth`` still exists as a fallback but downstream code
       prefers the real depth when present.
    """

    _stop_action: Tensor = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    _load_yolo: bool = False
    _non_coco_caption: str = (
        "chair . table . tv . laptop . microwave . toaster . sink . refrigerator . book"
        " . clock . vase . scissors . teddy bear . hair drier . toothbrush ."
    )
    _initial_yaws: List[float] = INITIAL_BODY_YAWS.copy()
    _observations_cache: Dict[str, Any] = {}
    _policy_info: Dict[str, Any] = {}
    _done_initializing: bool = False

    def __init__(self: Union["IsaacSimMixin", ITMPolicyV2], *args: Any, **kwargs: Any) -> None:
        super().__init__(sync_explored_areas=True, *args, **kwargs)  # type: ignore[misc]
        # Reality uses ZoeDepth to estimate hand-camera depth. In IsaacSim the
        # hand camera is a real RGB-D sensor, so we lazily load ZoeDepth only
        # if the env actually hands us a placeholder ones-array AND the user
        # has not pre-populated `_observations_cache["object_map_rgbd"]` with
        # real depth.
        self._depth_model = None
        self._object_map.use_dbscan = False  # type: ignore[attr-defined]

    @classmethod
    def from_config(cls, config: DictConfig, *args_unused: Any, **kwargs_unused: Any) -> Any:
        policy_config: VLFMConfig = config.policy
        kwargs = {k: policy_config[k] for k in VLFMConfig.kwaarg_names}  # type: ignore[attr-defined]
        return cls(**kwargs)

    def act(
        self: Union["IsaacSimMixin", ITMPolicyV2],
        observations: Dict[str, Any],
        rnn_hidden_states: Union[Tensor, Any],
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        """Run one policy step and return the env-shaped action dict.

        The structure of this method mirrors `RealityMixin.act` exactly,
        including the (surprising) index assignment ``angular = action[0][0]``,
        ``linear = action[0][1]``. Matching reality is important: the pointnav
        ResNet checkpoint that ships with VLFM was trained to emit values in
        that order, and the obstacle/value maps were tuned against that
        convention.
        """
        if observations["objectgoal"] not in self._non_coco_caption:
            self._non_coco_caption = observations["objectgoal"] + " . " + self._non_coco_caption
        parent_cls: ITMPolicyV2 = super()  # type: ignore[assignment]
        action: Tensor = parent_cls.act(
            observations, rnn_hidden_states, prev_actions, masks, deterministic
        )[0]

        if self._done_initializing:
            action_dict = {
                "angular": action[0][0].item(),
                "linear": action[0][1].item(),
                "arm_yaw": -1,
                "info": self._policy_info,
            }
        else:
            action_dict = {
                "angular": 0,
                "linear": 0,
                "arm_yaw": action[0][0].item(),  # body-yaw target (episodic radians)
                "info": self._policy_info,
            }

        if "rho_theta" in self._policy_info:
            action_dict["rho_theta"] = self._policy_info["rho_theta"]

        # Flip only after the last sweep target has been emitted. The next
        # `_pre_step` (which gates on `_did_reset`) won't run `_reset` again
        # until the env explicitly resets with masks[0] == 0.
        self._done_initializing = len(self._initial_yaws) == 0

        return action_dict

    def get_action(self, observations: Dict[str, Any], masks: Tensor, deterministic: bool = True) -> Dict[str, Any]:
        return self.act(observations, None, None, masks, deterministic=deterministic)

    def _reset(self: Union["IsaacSimMixin", ITMPolicyV2]) -> None:
        parent_cls: ITMPolicyV2 = super()  # type: ignore[assignment]
        parent_cls._reset()
        self._initial_yaws = INITIAL_BODY_YAWS.copy()
        self._done_initializing = False

    def _initialize(self) -> Tensor:
        """Pop the next body-yaw target and return it shaped as a (1, 1) tensor.

        The shape ``(1, 1)`` matches reality's ``_initialize`` so that
        downstream `_pre_step` plumbing (which reads ``action[0][0]``) works
        unchanged.
        """
        yaw = self._initial_yaws.pop(0)
        return torch.tensor([[yaw]], dtype=torch.float32)

    def _cache_observations(
        self: Union["IsaacSimMixin", ITMPolicyV2], observations: Dict[str, Any]
    ) -> None:
        """Cache obstacle / value / object maps from the per-step obs dict.

        Identical to `RealityMixin._cache_observations` because the obs-dict
        contract is identical. Pulled in verbatim so the policy doesn't have
        to know which backend produced the observations.
        """
        if len(self._observations_cache) > 0:
            return

        self._obstacle_map: ObstacleMap
        for obs_map_data in observations["obstacle_map_depths"][:-1]:
            depth, tf, min_depth, max_depth, fx, fy, topdown_fov = obs_map_data
            self._obstacle_map.update_map(
                depth, tf, min_depth, max_depth, fx, fy, topdown_fov, explore=False
            )

        _, tf, min_depth, max_depth, fx, fy, topdown_fov = observations["obstacle_map_depths"][-1]
        self._obstacle_map.update_map(
            None, tf, min_depth, max_depth, fx, fy, topdown_fov, explore=True, update_obstacles=False
        )

        self._obstacle_map.update_agent_traj(
            observations["robot_xy"], observations["robot_heading"]
        )
        frontiers = self._obstacle_map.frontiers

        height, width = observations["nav_depth"].shape
        nav_depth = torch.from_numpy(observations["nav_depth"])
        nav_depth = nav_depth.reshape(1, height, width, 1).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._observations_cache = {
            "frontier_sensor": frontiers,
            "nav_depth": nav_depth,
            "robot_xy": observations["robot_xy"],
            "robot_heading": observations["robot_heading"],
            "object_map_rgbd": observations["object_map_rgbd"],
            "value_map_rgbd": observations["value_map_rgbd"],
        }

    def _infer_depth(self, rgb: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
        """Monocular depth fallback. Loaded on first use to keep startup cheap
        when the IsaacSim hand camera already provides real depth.
        """
        if self._depth_model is None:
            self._depth_model = torch.hub.load(
                "isl-org/ZoeDepth", "ZoeD_NK", config_mode="eval", pretrained=True
            ).to("cuda" if torch.cuda.is_available() else "cpu")
        img_pil = Image.fromarray(rgb)
        with torch.inference_mode():
            depth = self._depth_model.infer_pil(img_pil)
        return (np.clip(depth, min_depth, max_depth)) / (max_depth - min_depth)


@dataclass
class IsaacSimConfig(DictConfig):
    policy: VLFMConfig = VLFMConfig()


class IsaacSimITMPolicyV2(IsaacSimMixin, ITMPolicyV2):
    """Concrete policy used by `run_isaacsim_objnav.py`. Body of the class is
    intentionally empty — the mixin supplies all IsaacSim-specific behaviour."""

    pass
