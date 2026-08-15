from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _run(command: list[str], *, cwd: Path, environment: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"command failed ({result.returncode}): {' '.join(command)}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result.stdout


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 10),
    reason="clean assembler wheel smoke is pinned to the Python 3.10 CI lane",
)
@pytest.mark.skipif(shutil.which("uv") is None, reason="clean-wheel smoke requires uv")
def test_clean_installed_wheels_import_and_expose_cli_help(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    assert uv is not None
    repository = Path(__file__).resolve().parents[2]
    distribution_dir = tmp_path / "dist"
    for project_name in ("python", "simulator", "shadow", "assembler"):
        _run(
            [
                uv,
                "build",
                "--wheel",
                "--out-dir",
                str(distribution_dir),
                str(repository / project_name),
            ],
            cwd=tmp_path,
        )
    wheels = sorted(distribution_dir.glob("*.whl"))
    assert len(wheels) == 4

    environment_dir = tmp_path / "venv"
    _run([uv, "venv", "--python", sys.executable, str(environment_dir)], cwd=tmp_path)
    python = environment_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    assembler = environment_dir / (
        "Scripts/foliage-warden-assemble.exe" if os.name == "nt" else "bin/foliage-warden-assemble"
    )
    _run(
        [uv, "pip", "install", "--python", str(python), *(str(path) for path in wheels)],
        cwd=tmp_path,
    )

    outside = tmp_path / "outside-repository"
    outside.mkdir()
    clean_environment = os.environ.copy()
    clean_environment.pop("PYTHONPATH", None)
    clean_environment["PYTHONNOUSERSITE"] = "1"
    installed_paths = json.loads(
        _run(
            [
                python,
                "-c",
                (
                    "import json, foliage_warden_assembler, foliage_warden_eval, "
                    "foliage_warden_shadow, foliage_warden_sim; "
                    "print(json.dumps([foliage_warden_assembler.__file__, "
                    "foliage_warden_eval.__file__, foliage_warden_shadow.__file__, "
                    "foliage_warden_sim.__file__]))"
                ),
            ],
            cwd=outside,
            environment=clean_environment,
        )
    )
    assert all(not Path(path).resolve().is_relative_to(repository) for path in installed_paths)

    help_text = _run(
        [assembler, "--help"],
        cwd=outside,
        environment=clean_environment,
    )
    assert "usage: foliage-warden-assemble" in help_text
    assert "Offline-only assembly" in help_text
