"""CuRobo robot-config loader with environment-variable expansion.

CuRobo does not expand ``$VAR`` / ``${VAR}`` itself, so YAML configs that
reference environment variables (e.g. ``${VLFM_ROOT}/spot_model/...``) must be
preprocessed before being handed to ``RobotConfig.from_dict`` or equivalent.
This module is the single entry point for that substitution.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml


_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def load_curobo_robot_config(
    path: str | Path,
    *,
    extra_env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Load a CuRobo robot YAML and expand ``$VAR`` / ``${VAR}`` in every string.

    Args:
        path: Path to the YAML config (e.g. ``spot_model/configuration/spot_arm_curobo.yaml``).
        extra_env: Optional overrides merged on top of ``os.environ`` for the
            substitution pass. Useful when the caller wants to supply
            ``VLFM_ROOT`` inline rather than exporting it.

    Returns:
        The fully expanded config dict, ready to pass to curobo.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        KeyError: If any referenced env var is unset (in both ``os.environ``
            and ``extra_env``). The error names every unresolved variable so
            callers can fix them up-front instead of debugging curobo's
            downstream "file not found" failures.
    """

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"CuRobo config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    env: dict[str, str] = dict(os.environ)
    if extra_env:
        env.update(extra_env)

    return _expand(raw, env=env, where=str(config_path))


def _expand(value: Any, *, env: Mapping[str, str], where: str) -> Any:
    if isinstance(value, str):
        return _expand_string(value, env=env, where=where)
    if isinstance(value, dict):
        return {key: _expand(item, env=env, where=where) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item, env=env, where=where) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand(item, env=env, where=where) for item in value)
    return value


def _expand_string(value: str, *, env: Mapping[str, str], where: str) -> str:
    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        if name not in env:
            missing.append(name)
            return match.group(0)
        return env[name]

    expanded = _ENV_VAR_PATTERN.sub(replace, value)
    if missing:
        unique = sorted(dict.fromkeys(missing))
        raise KeyError(
            f"Unresolved environment variable(s) {unique} while loading {where}: {value!r}. "
            "Export the variable(s) or pass them via extra_env before loading the curobo config."
        )
    return expanded
