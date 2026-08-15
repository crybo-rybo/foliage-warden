from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

EXPECTED_TRACE = (
    b'{"after_arm":"MONITORING","burst_count":1,"duplicate_suppressed":true,'
    b'"person_interlock":true,"person_state":"MONITORING","startup":"DISARMED"}\n'
)


@pytest.mark.skipif(
    shutil.which("cmake") is None or shutil.which("c++") is None,
    reason="C++ conformance probe needs cmake and a C++ compiler",
)
def test_cpp_core_overlap_probe_is_byte_stable(
    tmp_path: Path, repository: Path
) -> None:
    source = repository / "simulator" / "cpp_conformance"
    build = tmp_path / "build"
    subprocess.run(
        ["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build), "--parallel", "2"],
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(
        [str(build / "foliage_warden_cpp_conformance")],
        check=True,
        capture_output=True,
    )
    assert completed.stdout == EXPECTED_TRACE
