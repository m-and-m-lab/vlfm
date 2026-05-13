"""IsaacSim deployment path for VLFM.

This subpackage runs the existing VLFM policy stack (`vlfm.policy.itm_policy`,
`vlfm.mapping.*`) inside IsaacSim, using the `skills/` package as the IsaacLab
adapter for sensor capture and locomotion commands. It is the IsaacSim analogue
of `vlfm.reality`.

The sensor + action contract matches `vlfm.reality.objectnav_env.ObjectNavEnv` —
same observation-dict keys, same action-dict keys — so the policy layer is
backend-agnostic.
"""
