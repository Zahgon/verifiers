"""Prime-hosted command plugin contract."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
from functools import lru_cache
from typing import Sequence

PRIME_PLUGIN_API_VERSION = 1
WORKSPACE_ENV_DIR = "environments"


def _venv_python(venv_root: Path) -> Path:
    pass


@lru_cache(maxsize=32)
def _python_can_import_module(
    python_executable: str, module_name: str, cwd: str
) -> bool:
    pass


def _resolve_workspace_python(cwd: Path | None = None) -> str:
    pass


def _find_workspace_root(start: Path) -> Path | None:
    pass


def _current_cwd() -> Path:
    pass


def _resolve_dir_arg(value: str, cwd: Path) -> str:
    pass


def _normalize_or_append_dir_option(
    args: Sequence[str] | None,
    *,
    long_flag: str,
    short_flag: str,
    fallback_value: str | None,
    cwd: Path,
) -> list[str]:
    pass


@dataclass(frozen=True)
class PrimeCLIPlugin:
    """Declarative command surface consumed by prime-cli."""

    api_version: int = PRIME_PLUGIN_API_VERSION
    eval_module: str = "verifiers.cli.commands.eval"
    gepa_module: str = "verifiers.cli.commands.gepa"
    install_module: str = "verifiers.cli.commands.install"
    init_module: str = "verifiers.cli.commands.init"
    setup_module: str = "verifiers.cli.commands.setup"
    build_module: str = "verifiers.cli.commands.build"

    def build_module_command(
        self, module_name: str, args: Sequence[str] | None = None
    ) -> list[str]:
        pass


def get_plugin() -> PrimeCLIPlugin:
    """Return the prime plugin definition."""
    pass
