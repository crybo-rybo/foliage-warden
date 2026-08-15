"""Loading and verification for pinned detector artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ModelError

DEFAULT_MODEL_ID = "yolox_s_opencv_zoo"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    description: str
    filename: str
    input_width: int
    input_height: int
    input_color: str
    person_class_id: int
    cat_class_id: int
    sha256: str
    source_revision: str
    url: str

    @property
    def relevant_classes(self) -> dict[int, str]:
        return {self.person_class_id: "person", self.cat_class_id: "cat"}


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ModelError(f"{context} must be a JSON object")
    return value


def _string(data: dict[str, Any], key: str, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ModelError(f"{context}.{key} must be a non-empty string")
    return value


def _positive_int(data: dict[str, Any], key: str, context: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelError(f"{context}.{key} must be a positive integer")
    return value


def load_model_spec(path: str | Path, model_id: str = DEFAULT_MODEL_ID) -> ModelSpec:
    """Load the subset of registry metadata required by this detector."""

    registry_path = Path(path)
    try:
        raw = json.loads(registry_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ModelError(
            f"model registry not found at {registry_path}; pass --registry or run from "
            "the repository root"
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise ModelError(f"cannot read model registry {registry_path}: {error}") from error

    root = _mapping(raw, "registry")
    models = _mapping(root.get("models"), "registry.models")
    if model_id not in models:
        choices = ", ".join(sorted(models)) or "<none>"
        raise ModelError(f"model {model_id!r} is not in {registry_path}; available: {choices}")
    model = _mapping(models[model_id], f"registry.models.{model_id}")
    input_config = _mapping(model.get("input"), f"registry.models.{model_id}.input")
    classes = _mapping(
        model.get("relevant_classes"), f"registry.models.{model_id}.relevant_classes"
    )
    person_class_id = classes.get("person")
    cat_class_id = classes.get("cat")
    if not isinstance(person_class_id, int) or not isinstance(cat_class_id, int):
        raise ModelError("model relevant_classes must contain integer person and cat IDs")

    digest = _string(model, "sha256", f"registry.models.{model_id}").lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ModelError(f"registry.models.{model_id}.sha256 is not a SHA-256 digest")

    return ModelSpec(
        model_id=model_id,
        description=_string(model, "description", f"registry.models.{model_id}"),
        filename=_string(model, "filename", f"registry.models.{model_id}"),
        input_width=_positive_int(input_config, "width", f"registry.models.{model_id}.input"),
        input_height=_positive_int(input_config, "height", f"registry.models.{model_id}.input"),
        input_color=_string(input_config, "color", f"registry.models.{model_id}.input"),
        person_class_id=person_class_id,
        cat_class_id=cat_class_id,
        sha256=digest,
        source_revision=_string(model, "source_revision", f"registry.models.{model_id}"),
        url=_string(model, "url", f"registry.models.{model_id}"),
    )


def default_registry_path() -> Path:
    """Resolve the in-repository registry without assuming an install layout."""

    candidates = [
        Path.cwd() / "models" / "registry.json",
        Path(__file__).resolve().parents[3] / "models" / "registry.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def resolve_and_verify_model(
    registry_path: str | Path,
    spec: ModelSpec,
    model_override: str | Path | None = None,
) -> Path:
    """Resolve a model path and reject missing or unpinned bytes."""

    model_path = (
        Path(model_override)
        if model_override is not None
        else Path(registry_path).parent / spec.filename
    )
    if not model_path.is_file():
        raise ModelError(
            f"model artifact not found at {model_path}. Fetch the pinned artifact with "
            f"`uv run python tools/fetch_model.py {spec.model_id}` from the repository root, "
            "or pass --model."
        )

    digest = hashlib.sha256()
    try:
        with model_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ModelError(f"cannot read model artifact {model_path}: {error}") from error
    actual = digest.hexdigest()
    if actual != spec.sha256:
        raise ModelError(
            f"model checksum mismatch for {model_path}: expected {spec.sha256}, got {actual}; "
            "refusing to run unpinned model bytes"
        )
    return model_path
