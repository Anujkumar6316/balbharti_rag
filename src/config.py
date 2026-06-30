"""
config.py — Load YAML config once, expose as a singleton.

We use a module-level singleton because:
1. Pi is serial-kiosk mode — no concurrency concerns.
2. Config is read once at startup, never mutated.
3. Avoids passing config through every function signature.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

_CONFIG: Dict[str, Any] | None = None
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | os.PathLike | None = None) -> Dict[str, Any]:
    """Load YAML config. Subsequent calls return cached value.

    Args:
        path: Optional override. Defaults to <project_root>/config.yaml.
    """
    global _CONFIG
    if _CONFIG is not None and path is None:
        return _CONFIG

    cfg_path = Path(path) if path else _PROJECT_ROOT / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        _CONFIG = yaml.safe_load(f)

    # Resolve relative paths against project root
    def _resolve(node: Any) -> Any:
        if isinstance(node, str) and (
            node.startswith("./") or node.startswith("../")
        ):
            return str((_PROJECT_ROOT / node).resolve())
        if isinstance(node, dict):
            return {k: _resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_resolve(x) for x in node]
        return node

    _CONFIG = _resolve(_CONFIG)
    _CONFIG["_project_root"] = str(_PROJECT_ROOT)
    return _CONFIG


def get_config() -> Dict[str, Any]:
    """Return loaded config; raises if load_config() was never called."""
    if _CONFIG is None:
        return load_config()
    return _CONFIG


def project_root() -> Path:
    """Return absolute project root path."""
    return _PROJECT_ROOT
