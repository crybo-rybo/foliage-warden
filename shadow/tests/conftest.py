from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repository() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def config_path(repository: Path) -> Path:
    return repository / "config" / "simulation-safe.example.json"


@pytest.fixture(scope="session")
def runtime_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text(encoding="utf-8"))
