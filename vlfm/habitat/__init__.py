"""Quarantined Habitat-only modules.

This subpackage isolates all code that depends on `habitat` or `habitat_baselines`
so that the reality (Spot) and IsaacSim runtimes can import the rest of `vlfm`
without those dependencies installed.

Importing anything under `vlfm.habitat.*` requires `pip install -e '.[habitat]'`.
The reality and isaacsim entrypoints must not import from this subpackage.
"""
