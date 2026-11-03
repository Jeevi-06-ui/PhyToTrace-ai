"""Filesystem path helpers for PhytoTrace AI."""

from pathlib import Path
from typing import Union


def get_repo_root() -> Path:
    """Return the absolute Path to the repository root directory."""
    # This file is at src/phytotrace/utils/paths.py -> 3 levels up to repo root
    return Path(__file__).resolve().parent.parent.parent.parent


def resolve_path(relative_or_absolute: Union[str, Path]) -> Path:
    """Resolve a path relative to the repository root if it is not already absolute."""
    p = Path(relative_or_absolute)
    if p.is_absolute():
        return p
    return (get_repo_root() / p).resolve()

