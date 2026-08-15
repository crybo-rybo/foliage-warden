"""Leakage-safe, deterministic dataset partitioning."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .schemas import DatasetItem, JsonValue

DEFAULT_RATIOS: dict[str, float] = {"test": 0.15, "train": 0.7, "validation": 0.15}


@dataclass(frozen=True, slots=True)
class SplitResult:
    item_assignments: dict[str, str]
    session_assignments: dict[str, str]
    group_assignments: dict[str, str]
    ratios: dict[str, float]
    seed: str

    def assert_no_leakage(self, items: Iterable[DatasetItem]) -> None:
        sessions: dict[str, set[str]] = {}
        groups: dict[str, set[str]] = {}
        for item in items:
            split = self.item_assignments[item.item_id]
            sessions.setdefault(item.session_id, set()).add(split)
            groups.setdefault(item.split_group, set()).add(split)
        leaking_sessions = sorted(key for key, values in sessions.items() if len(values) > 1)
        leaking_groups = sorted(key for key, values in groups.items() if len(values) > 1)
        if leaking_sessions or leaking_groups:
            raise AssertionError(
                f"split leakage detected; sessions={leaking_sessions}, groups={leaking_groups}"
            )

    def to_dict(self) -> dict[str, JsonValue]:
        counts = {split: 0 for split in self.ratios}
        for split in self.item_assignments.values():
            counts[split] += 1
        session_counts = {split: 0 for split in self.ratios}
        for split in self.session_assignments.values():
            session_counts[split] += 1
        group_counts = {split: 0 for split in self.ratios}
        for split in self.group_assignments.values():
            group_counts[split] += 1
        return {
            "counts": {
                "groups": dict(sorted(group_counts.items())),
                "items": dict(sorted(counts.items())),
                "sessions": dict(sorted(session_counts.items())),
            },
            "group_assignments": dict(sorted(self.group_assignments.items())),
            "item_assignments": dict(sorted(self.item_assignments.items())),
            "ratios": dict(sorted(self.ratios.items())),
            "seed": self.seed,
            "session_assignments": dict(sorted(self.session_assignments.items())),
        }


def _validate_ratios(ratios: Mapping[str, float]) -> dict[str, float]:
    if not ratios:
        raise ValueError("ratios must not be empty")
    normalized: dict[str, float] = {}
    for name, ratio in ratios.items():
        if not isinstance(name, str) or not name:
            raise ValueError("split names must be non-empty strings")
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or ratio < 0.0:
            raise ValueError(f"ratio for {name!r} must be a non-negative number")
        normalized[name] = float(ratio)
    total = sum(normalized.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"split ratios must sum to 1.0, got {total}")
    if not any(ratio > 0.0 for ratio in normalized.values()):
        raise ValueError("at least one split ratio must be positive")
    return dict(sorted(normalized.items()))


def _allocation(total: int, ratios: Mapping[str, float]) -> dict[str, int]:
    raw = {name: total * ratio for name, ratio in ratios.items()}
    allocated = {name: math.floor(value) for name, value in raw.items()}
    remaining = total - sum(allocated.values())
    # Largest remainder apportionment makes counts as close to requested ratios
    # as possible. Split name is the deterministic tie-breaker.
    order = sorted(ratios, key=lambda name: (-(raw[name] - allocated[name]), name))
    for name in order[:remaining]:
        allocated[name] += 1
    return allocated


def split_by_session(
    items: Iterable[DatasetItem],
    *,
    ratios: Mapping[str, float] = DEFAULT_RATIOS,
    seed: str | int = 0,
) -> SplitResult:
    """Assign complete sessions (and optional cross-session groups) together.

    Set ``group_id`` to a recording day, animal, camera setup, or household when
    that broader unit must also remain isolated across train/validation/test.
    """

    item_list = list(items)
    normalized_ratios = _validate_ratios(ratios)
    seed_text = str(seed)
    item_ids: set[str] = set()
    session_groups: dict[str, str] = {}
    for item in item_list:
        if item.item_id in item_ids:
            raise ValueError(f"duplicate dataset item_id {item.item_id!r}")
        item_ids.add(item.item_id)
        previous_group = session_groups.get(item.session_id)
        if previous_group is not None and previous_group != item.split_group:
            raise ValueError(
                f"session {item.session_id!r} has conflicting groups "
                f"{previous_group!r} and {item.split_group!r}"
            )
        session_groups[item.session_id] = item.split_group

    groups = sorted(
        set(session_groups.values()),
        key=lambda group: (
            hashlib.sha256(f"{seed_text}\0{group}".encode()).digest(),
            group,
        ),
    )
    allocation = _allocation(len(groups), normalized_ratios)
    group_assignments: dict[str, str] = {}
    offset = 0
    for split in normalized_ratios:
        split_groups = groups[offset : offset + allocation[split]]
        group_assignments.update((group, split) for group in split_groups)
        offset += allocation[split]
    session_assignments = {
        session_id: group_assignments[group]
        for session_id, group in session_groups.items()
    }
    item_assignments = {
        item.item_id: session_assignments[item.session_id]
        for item in item_list
    }
    result = SplitResult(
        item_assignments=item_assignments,
        session_assignments=session_assignments,
        group_assignments=group_assignments,
        ratios=normalized_ratios,
        seed=seed_text,
    )
    result.assert_no_leakage(item_list)
    return result

