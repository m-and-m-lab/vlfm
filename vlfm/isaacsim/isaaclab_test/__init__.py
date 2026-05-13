"""IsaacLab smoke harnesses for the VLFM IsaacSim package.

The two scripts in this directory (`smoke_camera_parity.py`, `smoke_env_loop.py`)
each construct a minimal IsaacLab scene, wire `IsaaclabSpot` to it, and
exercise one slice of the VLFM IsaacSim contract — see plan Section 7 smokes
2 and 3.

Each script must be invoked directly (not via `python -m`) because `AppLauncher`
needs to start the SimulationApp before any `isaaclab.*` imports. The shared
helper module `_scene` is intentionally imported AFTER `AppLauncher` fires in
each script.
"""
