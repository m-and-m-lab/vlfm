"""Locomotion skill surface for Spot policy deployment on Isaac Lab."""

from .api import SpotLocomotionClient

__all__ = [
    "SpotLocomotionClient",
]

try:
    from .spot_policy import SpotFlatTerrainPolicy
except (ImportError, ModuleNotFoundError):
    pass
else:
    __all__.append("SpotFlatTerrainPolicy")
