"""Camera-name constants for the IsaacSim Spot rig.

Names mirror the keys in `skills/config/spot_arm_cameras.yaml`. They are NOT the
same as `vlfm.reality.robots.camera_ids.SpotCamIds` (which use Spot SDK source
names like "frontleft_depth"). VLFM doesn't care about the names themselves —
only that the env's `_get_camera_obs` consistently maps the same string to the
same physical camera.
"""


class IsaacSimCamIds:
    FRONTLEFT: str = "frontleft"
    FRONTRIGHT: str = "frontright"
    HAND: str = "hand"


POINT_CLOUD_CAMS = [
    IsaacSimCamIds.FRONTLEFT,
    IsaacSimCamIds.FRONTRIGHT,
]

VALUE_MAP_CAMS = [
    IsaacSimCamIds.HAND,
]

ALL_CAMS = list(dict.fromkeys(POINT_CLOUD_CAMS + VALUE_MAP_CAMS))
