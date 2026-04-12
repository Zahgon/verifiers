"""Utilities for detecting verifiers and environment version/commit info."""

from __future__ import annotations

import importlib.metadata
import logging
import subprocess
from pathlib import Path

from verifiers.types import VersionInfo

logger = logging.getLogger(__name__)


def get_commit_for_path(path: Path) -> str | None:
    """
    Get the git commit hash for a file or directory path.

    Walks up the directory tree to find a git repository and returns the
    HEAD commit hash.
    """
    pass


def get_package_source_path(package_name: str) -> Path | None:
    """Get the source directory for an installed package."""
    pass


def get_vf_version() -> str:
    """Return the verifiers framework version."""
    pass


def get_vf_commit() -> str | None:
    """Return the git commit hash of the verifiers package, or None."""
    pass


def get_env_version(env_id: str) -> str | None:
    """Return the installed version of an environment package, or None."""
    pass


def get_env_commit(env_id: str) -> str | None:
    """Return the git commit hash of an environment package, or None."""
    pass


def get_version_info(env_id: str) -> VersionInfo:
    """Get version and commit info for the verifiers framework and an environment."""
    pass
