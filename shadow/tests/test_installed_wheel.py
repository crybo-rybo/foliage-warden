from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from support import predictions_for, series

from foliage_warden_shadow.contracts import stable_json


def _run(command: list[str], *, cwd: Path, environment: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"command failed ({result.returncode}): {' '.join(command)}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result.stdout


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(stable_json(record) + "\n" for record in records),
        encoding="utf-8",
    )


@pytest.mark.skipif(shutil.which("uv") is None, reason="clean-wheel smoke requires uv")
def test_clean_installed_wheel_uses_packaged_safe_config_by_default(
    tmp_path: Path,
    repository: Path,
) -> None:
    uv = shutil.which("uv")
    assert uv is not None
    distribution_dir = tmp_path / "dist"
    for project in ("python", "simulator", "shadow"):
        _run(
            [
                uv,
                "build",
                "--wheel",
                "--out-dir",
                str(distribution_dir),
                str(repository / project),
            ],
            cwd=tmp_path,
        )
    wheels = sorted(distribution_dir.glob("*.whl"))
    assert len(wheels) == 3

    environment_dir = tmp_path / "venv"
    _run([uv, "venv", "--python", sys.executable, str(environment_dir)], cwd=tmp_path)
    python = environment_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    shadow = environment_dir / (
        "Scripts/foliage-warden-shadow.exe" if os.name == "nt" else "bin/foliage-warden-shadow"
    )
    _run(
        [uv, "pip", "install", "--python", str(python), *(str(path) for path in wheels)],
        cwd=tmp_path,
    )

    outside = tmp_path / "outside-repository"
    outside.mkdir()
    perceptions = series(4)
    predictions = predictions_for(perceptions, "EATING")
    perception_path = outside / "perception.jsonl"
    behavior_path = outside / "behavior.jsonl"
    scenario_path = outside / "scenario.json"
    summary_path = outside / "summary.json"
    _write_jsonl(perception_path, perceptions)
    _write_jsonl(behavior_path, predictions)

    clean_environment = os.environ.copy()
    clean_environment.pop("PYTHONPATH", None)
    clean_environment["PYTHONNOUSERSITE"] = "1"
    installed_paths = json.loads(
        _run(
            [
                python,
                "-c",
                (
                    "import json, foliage_warden_eval, foliage_warden_shadow, "
                    "foliage_warden_sim; "
                    "print(json.dumps([foliage_warden_eval.__file__, "
                    "foliage_warden_shadow.__file__, foliage_warden_sim.__file__]))"
                ),
            ],
            cwd=outside,
            environment=clean_environment,
        )
    )
    assert all(not Path(path).resolve().is_relative_to(repository) for path in installed_paths)

    output = _run(
        [
            shadow,
            str(perception_path),
            str(behavior_path),
            "--scenario-out",
            str(scenario_path),
            "--summary",
            str(summary_path),
        ],
        cwd=outside,
        environment=clean_environment,
    )

    summary = json.loads(output)
    assert summary == json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["passed"] is True
    assert summary["mode"] == "OBSERVE_ONLY"
    assert summary["actuator"] == {
        "backend": "MOCK",
        "physical_effect_possible": False,
    }
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    config_path = Path(scenario["config_ref"])
    assert config_path.is_file()
    assert not config_path.resolve().is_relative_to(repository)
