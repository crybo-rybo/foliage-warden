"""Locate canonical contracts in a checkout or the installed package."""

from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any

_CONFIG_NAME = "simulation-safe.example.json"
_REQUIRED_SCENARIOS = tuple(f"{index:02d}-" for index in range(1, 15))
_REQUIRED_SCHEMAS = (
    "action-audit.schema.json",
    "common.schema.json",
    "event-record.schema.json",
    "runtime-config.schema.json",
    "scenario.schema.json",
)
_materialized_resources: tempfile.TemporaryDirectory[str] | None = None


def _is_contract_root(root: Path) -> bool:
    scenarios = root / "scenarios"
    names = tuple(path.name for path in sorted(scenarios.glob("*.json")))
    return (
        len(names) == len(_REQUIRED_SCENARIOS)
        and all(
            name.startswith(prefix) for name, prefix in zip(names, _REQUIRED_SCENARIOS)
        )
        and all((root / "schemas" / name).is_file() for name in _REQUIRED_SCHEMAS)
        and (root / "config" / _CONFIG_NAME).is_file()
    )


def checkout_contract_root() -> Path | None:
    """Return the repository root when running from an editable checkout."""

    candidate = Path(__file__).resolve().parents[3]
    return candidate if _is_contract_root(candidate) else None


def _copy_traversable(source: Any, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        target = destination / child.name
        if child.is_dir():
            _copy_traversable(child, target)
        elif child.is_file():
            target.write_bytes(child.read_bytes())


@lru_cache(maxsize=1)
def packaged_contract_root() -> Path:
    """Return bundled contracts, materializing non-filesystem resources if needed."""

    resource = importlib_resources.files("foliage_warden_sim").joinpath("resources")
    try:
        candidate = Path(os.fspath(resource))
    except TypeError:
        global _materialized_resources
        _materialized_resources = tempfile.TemporaryDirectory(
            prefix="foliage-warden-simulator-contracts-"
        )
        candidate = Path(_materialized_resources.name)
        _copy_traversable(resource, candidate)
    if not _is_contract_root(candidate):
        raise FileNotFoundError(
            "installed foliage-warden-simulator package lacks its canonical contracts"
        )
    return candidate.resolve()


@lru_cache(maxsize=1)
def default_contract_root() -> Path:
    """Prefer checkout contracts so editable runs use the repository source of truth."""

    return checkout_contract_root() or packaged_contract_root()


def default_scenario_dir() -> Path:
    return default_contract_root() / "scenarios"


def contract_root_for(scenario_path: Path) -> Path:
    """Identify the trusted root that contains a scenario path."""

    resolved = scenario_path.resolve()
    checkout = checkout_contract_root()
    if checkout is not None and resolved.is_relative_to(checkout):
        return checkout
    packaged = packaged_contract_root()
    if resolved.is_relative_to(packaged):
        return packaged
    return checkout or packaged
