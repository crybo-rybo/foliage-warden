#!/usr/bin/env python3
"""Emit a read-only JSON inventory of a prospective Jetson runtime target."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any


def command_output(*command: str, timeout: float = 5.0) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def first_line(*command: str) -> str | None:
    output = command_output(*command)
    return output.splitlines()[0] if output else None


def parse_os_release() -> dict[str, str]:
    values: dict[str, str] = {}
    path = Path("/etc/os-release")
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw_line or raw_line.startswith("#"):
            continue
        key, value = raw_line.split("=", 1)
        values[key] = value.strip('"')
    return values


def memory_kib() -> dict[str, int]:
    values: dict[str, int] = {}
    path = Path("/proc/meminfo")
    if not path.exists():
        return values
    wanted = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, _, remainder = line.partition(":")
        if key in wanted:
            values[key] = int(remainder.strip().split()[0])
    return values


def video_devices() -> list[dict[str, str]]:
    devices: list[dict[str, str]] = []
    for node in sorted(Path("/sys/class/video4linux").glob("video*")):
        try:
            name = (node / "name").read_text(encoding="utf-8").strip()
        except OSError:
            name = "unknown"
        devices.append({"device": f"/dev/{node.name}", "name": name})
    return devices


def gstreamer_plugins() -> dict[str, bool]:
    inspect = shutil.which("gst-inspect-1.0")
    names = [
        "nvarguscamerasrc",
        "nvv4l2camerasrc",
        "nvv4l2decoder",
        "nvvidconv",
        "nvinfer",
        "nvtracker",
        "nvdsanalytics",
    ]
    if inspect is None:
        return {name: False for name in names}
    return {
        name: subprocess.run(
            [inspect, name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).returncode
        == 0
        for name in names
    }


def python_modules() -> dict[str, str | None]:
    import importlib.util

    versions: dict[str, str | None] = {}
    for name in ("cv2", "numpy", "torch", "torchvision", "tensorrt", "onnx", "onnxruntime"):
        if importlib.util.find_spec(name) is None:
            versions[name] = None
            continue
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "present"))
        except Exception as exc:  # The failure type is more useful than aborting the probe.
            versions[name] = f"import-error:{type(exc).__name__}"
    return versions


def inventory() -> dict[str, Any]:
    root_usage = shutil.disk_usage("/")
    l4t_path = Path("/etc/nv_tegra_release")
    l4t = l4t_path.read_text(encoding="utf-8").strip() if l4t_path.exists() else None
    nvcc = Path("/usr/local/cuda/bin/nvcc")
    trtexec = Path("/usr/src/tensorrt/bin/trtexec")
    return {
        "schema_version": 1,
        "platform": {
            "hostname": platform.node(),
            "machine": platform.machine(),
            "kernel": platform.release(),
            "os_release": parse_os_release(),
            "l4t_release": l4t,
            "cpu_count": os.cpu_count(),
            "memory_kib": memory_kib(),
            "root_disk_bytes": {
                "total": root_usage.total,
                "used": root_usage.used,
                "free": root_usage.free,
            },
        },
        "runtime": {
            "cuda_compiler": first_line(str(nvcc), "--version") if nvcc.exists() else None,
            "nvidia_smi": first_line("nvidia-smi"),
            "tensorrt_exec": str(trtexec) if trtexec.exists() else shutil.which("trtexec"),
            "gstreamer": first_line("gst-launch-1.0", "--version"),
            "cmake": first_line("cmake", "--version"),
            "compiler": first_line("g++", "--version"),
            "docker": first_line("docker", "--version"),
            "python": platform.python_version(),
            "python_modules": python_modules(),
            "gstreamer_plugins": gstreamer_plugins(),
        },
        "devices": {
            "video": video_devices(),
            "power_mode": command_output("nvpmodel", "-q"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    args = parser.parse_args()
    print(json.dumps(inventory(), indent=None if args.compact else 2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

