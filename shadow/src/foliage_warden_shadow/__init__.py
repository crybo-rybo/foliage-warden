"""Observe-only fusion and deterministic reference-policy replay."""

from .contracts import (
    BEHAVIOR_LABELS,
    BehaviorPrediction,
    ContractError,
    PerceptionObservation,
    parse_behavior_stream,
    parse_perception_stream,
)
from .fusion import FusionOptions, FusionResult, assemble_scenario, fuse_streams
from .runner import ShadowRun, execute_shadow

__all__ = [
    "BEHAVIOR_LABELS",
    "BehaviorPrediction",
    "ContractError",
    "FusionOptions",
    "FusionResult",
    "PerceptionObservation",
    "ShadowRun",
    "assemble_scenario",
    "execute_shadow",
    "fuse_streams",
    "parse_behavior_stream",
    "parse_perception_stream",
]
