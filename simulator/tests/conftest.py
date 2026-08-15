from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
SCENARIOS = REPOSITORY / "scenarios"
SCHEMAS = REPOSITORY / "schemas"
CONFIG = REPOSITORY / "config" / "simulation-safe.example.json"


@pytest.fixture
def repository() -> Path:
    return REPOSITORY


@pytest.fixture
def schemas() -> Path:
    return SCHEMAS


@pytest.fixture
def config() -> Path:
    return CONFIG


@pytest.fixture
def scenario_dir() -> Path:
    return SCENARIOS


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path
