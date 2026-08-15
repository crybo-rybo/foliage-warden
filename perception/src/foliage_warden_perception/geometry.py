"""Normalized scene geometry and box/zone overlap evidence."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .errors import ZoneError
from .types import NormalizedBox, _wire_float

_EPSILON = 1e-10


class ZoneType(str, Enum):
    APPROACH = "approach"
    FOLIAGE = "foliage"
    SOIL = "soil"
    NO_FIRE = "no_fire"


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.x) or not math.isfinite(self.y):
            raise ZoneError("zone coordinates must be finite")
        if not 0.0 <= self.x <= 1.0 or not 0.0 <= self.y <= 1.0:
            raise ZoneError("zone coordinates must be within normalized [0, 1] image space")


Triangle = tuple[Point, Point, Point]


@dataclass(frozen=True, slots=True)
class Zone:
    zone_id: str
    zone_type: ZoneType
    points: tuple[Point, ...]
    plant_id: str | None = None
    _triangles: tuple[Triangle, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.zone_id:
            raise ZoneError("zone id must be a non-empty string")
        if len(self.points) < 3:
            raise ZoneError(f"zone {self.zone_id!r} must contain at least three points")
        if len(set(self.points)) != len(self.points):
            raise ZoneError(f"zone {self.zone_id!r} contains a duplicate point")
        if not _is_simple_polygon(self.points):
            raise ZoneError(f"zone {self.zone_id!r} polygon self-intersects")
        if abs(_signed_area(self.points)) <= _EPSILON:
            raise ZoneError(f"zone {self.zone_id!r} polygon has zero area")
        object.__setattr__(self, "_triangles", _triangulate(self.points, self.zone_id))

    def overlap_ratio(self, box: NormalizedBox) -> float:
        intersection_area = sum(
            abs(_signed_area(_clip_polygon_to_box(triangle, box)))
            for triangle in self._triangles
        )
        return min(1.0, max(0.0, intersection_area / box.area))


@dataclass(frozen=True, slots=True)
class ZoneOverlap:
    zone_id: str
    zone_type: ZoneType
    ratio: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "overlap": _wire_float(self.ratio),
            "zone_id": self.zone_id,
            "zone_type": self.zone_type.value,
        }


@dataclass(frozen=True, slots=True)
class RegionEvidence:
    approach_overlap: float
    foliage_overlap: float
    soil_overlap: float
    no_fire_overlap: float
    zone_id: str | None
    overlaps: tuple[ZoneOverlap, ...]

    @property
    def no_fire_intersection(self) -> bool:
        return self.no_fire_overlap > _EPSILON

    def policy_dict(self) -> dict[str, float]:
        return {
            "approach_overlap": _wire_float(self.approach_overlap),
            "foliage_overlap": _wire_float(self.foliage_overlap),
            "motion_score": 0.0,
            "soil_overlap": _wire_float(self.soil_overlap),
        }


def _cross(a: Point, b: Point, c: Point) -> float:
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)


def _signed_area(points: tuple[Point, ...] | list[Point]) -> float:
    if len(points) < 3:
        return 0.0
    doubled = sum(
        point.x * points[(index + 1) % len(points)].y
        - points[(index + 1) % len(points)].x * point.y
        for index, point in enumerate(points)
    )
    return doubled / 2.0


def _on_segment(a: Point, b: Point, point: Point) -> bool:
    return (
        min(a.x, b.x) - _EPSILON <= point.x <= max(a.x, b.x) + _EPSILON
        and min(a.y, b.y) - _EPSILON <= point.y <= max(a.y, b.y) + _EPSILON
        and abs(_cross(a, b, point)) <= _EPSILON
    )


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    ab_c = _cross(a, b, c)
    ab_d = _cross(a, b, d)
    cd_a = _cross(c, d, a)
    cd_b = _cross(c, d, b)
    if (
        (ab_c > _EPSILON and ab_d < -_EPSILON)
        or (ab_c < -_EPSILON and ab_d > _EPSILON)
    ) and (
        (cd_a > _EPSILON and cd_b < -_EPSILON)
        or (cd_a < -_EPSILON and cd_b > _EPSILON)
    ):
        return True
    return (
        (abs(ab_c) <= _EPSILON and _on_segment(a, b, c))
        or (abs(ab_d) <= _EPSILON and _on_segment(a, b, d))
        or (abs(cd_a) <= _EPSILON and _on_segment(c, d, a))
        or (abs(cd_b) <= _EPSILON and _on_segment(c, d, b))
    )


def _is_simple_polygon(points: tuple[Point, ...]) -> bool:
    edge_count = len(points)
    for first in range(edge_count):
        a = points[first]
        b = points[(first + 1) % edge_count]
        if a == b:
            return False
        for second in range(first + 1, edge_count):
            if second in (first, (first + 1) % edge_count):
                continue
            if first == 0 and second == edge_count - 1:
                continue
            c = points[second]
            d = points[(second + 1) % edge_count]
            if _segments_intersect(a, b, c, d):
                return False
    return True


def _point_in_triangle(point: Point, a: Point, b: Point, c: Point) -> bool:
    first = _cross(a, b, point)
    second = _cross(b, c, point)
    third = _cross(c, a, point)
    return first >= -_EPSILON and second >= -_EPSILON and third >= -_EPSILON


def _triangulate(points: tuple[Point, ...], zone_id: str) -> tuple[Triangle, ...]:
    oriented = list(points if _signed_area(points) > 0.0 else reversed(points))
    remaining = list(range(len(oriented)))
    triangles: list[Triangle] = []
    while len(remaining) > 3:
        ear_found = False
        for position, current in enumerate(remaining):
            previous = remaining[position - 1]
            following = remaining[(position + 1) % len(remaining)]
            a, b, c = oriented[previous], oriented[current], oriented[following]
            if _cross(a, b, c) <= _EPSILON:
                continue
            if any(
                _point_in_triangle(oriented[candidate], a, b, c)
                for candidate in remaining
                if candidate not in (previous, current, following)
            ):
                continue
            triangles.append((a, b, c))
            del remaining[position]
            ear_found = True
            break
        if not ear_found:
            raise ZoneError(f"zone {zone_id!r} could not be triangulated")
    triangles.append(tuple(oriented[index] for index in remaining))  # type: ignore[arg-type]
    return tuple(triangles)


def _clip_boundary(
    points: list[Point],
    *,
    inside: Any,
    intersect: Any,
) -> list[Point]:
    if not points:
        return []
    output: list[Point] = []
    previous = points[-1]
    previous_inside = inside(previous)
    for current in points:
        current_inside = inside(current)
        if current_inside:
            if not previous_inside:
                output.append(intersect(previous, current))
            output.append(current)
        elif previous_inside:
            output.append(intersect(previous, current))
        previous = current
        previous_inside = current_inside
    return output


def _vertical_intersection(a: Point, b: Point, x: float) -> Point:
    ratio = (x - a.x) / (b.x - a.x)
    return Point(x, a.y + ratio * (b.y - a.y))


def _horizontal_intersection(a: Point, b: Point, y: float) -> Point:
    ratio = (y - a.y) / (b.y - a.y)
    return Point(a.x + ratio * (b.x - a.x), y)


def _clip_polygon_to_box(points: Triangle, box: NormalizedBox) -> list[Point]:
    clipped = list(points)
    clipped = _clip_boundary(
        clipped,
        inside=lambda point: point.x >= box.x - _EPSILON,
        intersect=lambda a, b: _vertical_intersection(a, b, box.x),
    )
    clipped = _clip_boundary(
        clipped,
        inside=lambda point: point.x <= box.x2 + _EPSILON,
        intersect=lambda a, b: _vertical_intersection(a, b, box.x2),
    )
    clipped = _clip_boundary(
        clipped,
        inside=lambda point: point.y >= box.y - _EPSILON,
        intersect=lambda a, b: _horizontal_intersection(a, b, box.y),
    )
    return _clip_boundary(
        clipped,
        inside=lambda point: point.y <= box.y2 + _EPSILON,
        intersect=lambda a, b: _horizontal_intersection(a, b, box.y2),
    )


def evidence_for_box(box: NormalizedBox, zones: tuple[Zone, ...]) -> RegionEvidence:
    overlaps = tuple(
        ZoneOverlap(zone.zone_id, zone.zone_type, zone.overlap_ratio(box))
        for zone in sorted(zones, key=lambda item: item.zone_id)
    )

    def maximum(zone_type: ZoneType) -> float:
        return max((item.ratio for item in overlaps if item.zone_type is zone_type), default=0.0)

    approach = [
        item
        for item in overlaps
        if item.zone_type is ZoneType.APPROACH and item.ratio > _EPSILON
    ]
    approach.sort(key=lambda item: (-item.ratio, item.zone_id))
    return RegionEvidence(
        approach_overlap=maximum(ZoneType.APPROACH),
        foliage_overlap=maximum(ZoneType.FOLIAGE),
        soil_overlap=maximum(ZoneType.SOIL),
        no_fire_overlap=maximum(ZoneType.NO_FIRE),
        zone_id=approach[0].zone_id if approach else None,
        overlaps=overlaps,
    )


def load_zones(path: str | Path) -> tuple[Zone, ...]:
    """Load zones from a calibration export, scene object, or runtime config."""

    zone_path = Path(path)
    try:
        raw = json.loads(zone_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ZoneError(f"zone configuration not found at {zone_path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise ZoneError(f"cannot read zone configuration {zone_path}: {error}") from error
    if not isinstance(raw, dict):
        raise ZoneError("zone configuration root must be a JSON object")
    scene = raw.get("scene", raw)
    if not isinstance(scene, dict):
        raise ZoneError("scene must be a JSON object")
    raw_zones = scene.get("zones")
    if not isinstance(raw_zones, list):
        raise ZoneError("scene.zones must be an array")

    zones: list[Zone] = []
    seen_ids: set[str] = set()
    for index, raw_zone in enumerate(raw_zones):
        context = f"scene.zones[{index}]"
        if not isinstance(raw_zone, dict):
            raise ZoneError(f"{context} must be a JSON object")
        zone_id = raw_zone.get("id")
        if not isinstance(zone_id, str) or not zone_id:
            raise ZoneError(f"{context}.id must be a non-empty string")
        if zone_id in seen_ids:
            raise ZoneError(f"duplicate zone id {zone_id!r}")
        seen_ids.add(zone_id)
        try:
            zone_type = ZoneType(raw_zone.get("type"))
        except ValueError as error:
            choices = ", ".join(item.value for item in ZoneType)
            raise ZoneError(f"{context}.type must be one of: {choices}") from error
        raw_points = raw_zone.get("points")
        if not isinstance(raw_points, list):
            raise ZoneError(f"{context}.points must be an array")
        points: list[Point] = []
        for point_index, raw_point in enumerate(raw_points):
            if not isinstance(raw_point, dict):
                raise ZoneError(f"{context}.points[{point_index}] must be an object")
            x, y = raw_point.get("x"), raw_point.get("y")
            if (
                isinstance(x, bool)
                or isinstance(y, bool)
                or not isinstance(x, (int, float))
                or not isinstance(y, (int, float))
            ):
                raise ZoneError(f"{context}.points[{point_index}] needs numeric x and y")
            points.append(Point(float(x), float(y)))
        plant_id = raw_zone.get("plant_id")
        if plant_id is not None and (not isinstance(plant_id, str) or not plant_id):
            raise ZoneError(f"{context}.plant_id must be a non-empty string when present")
        zones.append(Zone(zone_id, zone_type, tuple(points), plant_id))
    return tuple(sorted(zones, key=lambda zone: zone.zone_id))
