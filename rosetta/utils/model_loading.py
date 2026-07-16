"""Resolve C2C model identifiers from the unified local model root."""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_HOST_MODEL_ROOT = Path("/home/lijunsi/projects/KVcache/models/c2c")
DEFAULT_CONTAINER_MODEL_ROOT = Path("/models/c2c")

MODEL_DIRECTORY_ALIASES = {
    "qwen2.5-math-1.5b": "Qwen2.5-Math-1.5B-Instruct",
}


def model_matches(
    model_name_or_path: str | os.PathLike[str], directory_name: str
) -> bool:
    """Return whether an ID or path refers to the named model directory."""
    return Path(model_name_or_path).name.lower() == directory_name.lower()


def model_root(model_root_path: str | os.PathLike[str] | None = None) -> Path:
    """Return the configured unified model root without requiring it to exist."""
    if model_root_path is not None:
        return Path(model_root_path).expanduser()
    configured = os.environ.get("C2C_MODEL_ROOT")
    if configured:
        return Path(configured).expanduser()
    if DEFAULT_CONTAINER_MODEL_ROOT.is_dir():
        return DEFAULT_CONTAINER_MODEL_ROOT
    return DEFAULT_HOST_MODEL_ROOT


def local_model_path(
    model_name_or_path: str | os.PathLike[str],
    model_root_path: str | os.PathLike[str] | None = None,
) -> Path | None:
    """Return a readable local model directory for an ID or legacy path."""
    original = Path(model_name_or_path).expanduser()
    if original.is_dir():
        return original

    root = model_root(model_root_path)
    basename = original.name
    directory_name = MODEL_DIRECTORY_ALIASES.get(basename.lower(), basename)
    candidate = root / directory_name
    if candidate.is_dir():
        return candidate

    if not root.is_dir():
        return None
    lowered = directory_name.lower()
    for child in root.iterdir():
        if child.name.lower() == lowered and child.is_dir():
            return child
    return None


def resolve_model_path(
    model_name_or_path: str | os.PathLike[str],
    model_root_path: str | os.PathLike[str] | None = None,
) -> str:
    """Prefer a unified local model directory, otherwise preserve the input."""
    local_path = local_model_path(model_name_or_path, model_root_path)
    if local_path is None:
        return os.fspath(model_name_or_path)
    resolved = str(local_path)
    print(f"[c2c-model] local={resolved}", flush=True)
    return resolved
