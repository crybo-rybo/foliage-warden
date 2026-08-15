from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

EXPECTED_TRACE_SHA256 = (
    "698a610a5a7e6449120c5131b20cc1e8ad737f86ccd86f686764c952b8be4e00"
)


def _run(
    command: list[str], *, cwd: Path, environment: dict[str, str] | None = None
) -> str:
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


@pytest.mark.skipif(shutil.which("uv") is None, reason="clean-wheel smoke requires uv")
def test_clean_installed_wheel_runs_all_scenarios_with_exact_trace(
    tmp_path: Path,
    repository: Path,
) -> None:
    uv = shutil.which("uv")
    assert uv is not None
    project = repository / "simulator"
    distribution_dir = tmp_path / "dist"
    _run([uv, "build", "--out-dir", str(distribution_dir), str(project)], cwd=tmp_path)
    wheels = list(distribution_dir.glob("*.whl"))
    assert len(wheels) == 1

    environment_dir = tmp_path / "venv"
    _run([uv, "venv", "--python", sys.executable, str(environment_dir)], cwd=tmp_path)
    python = environment_dir / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    simulator = environment_dir / (
        "Scripts/foliage-warden-sim.exe"
        if os.name == "nt"
        else "bin/foliage-warden-sim"
    )
    _run([uv, "pip", "install", "--python", str(python), str(wheels[0])], cwd=tmp_path)

    outside = tmp_path / "outside-repository"
    outside.mkdir()
    clean_environment = os.environ.copy()
    clean_environment.pop("PYTHONPATH", None)
    clean_environment["PYTHONNOUSERSITE"] = "1"
    installed_path = _run(
        [python, "-c", "import foliage_warden_sim; print(foliage_warden_sim.__file__)"],
        cwd=outside,
        environment=clean_environment,
    ).strip()
    assert not Path(installed_path).resolve().is_relative_to(repository)

    source_trace = tmp_path / "source-trace.json"
    source_output = _run(
        [
            sys.executable,
            "-m",
            "foliage_warden_sim.cli",
            "--all",
            "--trace",
            str(source_trace),
        ],
        cwd=outside,
    )
    installed_trace = tmp_path / "installed-trace.json"
    installed_output = _run(
        [simulator, "--all", "--trace", str(installed_trace)],
        cwd=outside,
        environment=clean_environment,
    )

    source_summary = json.loads(source_output)
    installed_summary = json.loads(installed_output)
    assert source_summary == installed_summary
    assert installed_summary["passed"] is True
    assert installed_summary["scenario_count"] == 14
    assert installed_summary["trace_sha256"] == EXPECTED_TRACE_SHA256
    assert installed_trace.read_bytes() == source_trace.read_bytes()
    assert (
        hashlib.sha256(installed_trace.read_bytes()).hexdigest()
        == EXPECTED_TRACE_SHA256
    )
