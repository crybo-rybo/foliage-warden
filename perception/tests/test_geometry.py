from __future__ import annotations

import json

import pytest

from foliage_warden_perception.errors import ZoneError
from foliage_warden_perception.geometry import (
    Point,
    Zone,
    ZoneType,
    evidence_for_box,
    load_zones,
)
from foliage_warden_perception.types import NormalizedBox


def test_rectangle_overlap_is_fraction_of_detection_box() -> None:
    zone = Zone(
        "left",
        ZoneType.FOLIAGE,
        (Point(0, 0), Point(0.5, 0), Point(0.5, 1), Point(0, 1)),
    )
    box = NormalizedBox(0.25, 0.25, 0.5, 0.5)

    assert zone.overlap_ratio(box) == pytest.approx(0.5)


def test_concave_polygon_is_triangulated_without_filling_its_notch() -> None:
    zone = Zone(
        "l-shape",
        ZoneType.APPROACH,
        (
            Point(0, 0),
            Point(1, 0),
            Point(1, 0.5),
            Point(0.5, 0.5),
            Point(0.5, 1),
            Point(0, 1),
        ),
    )

    assert zone.overlap_ratio(NormalizedBox(0, 0, 1, 1)) == pytest.approx(0.75)
    assert zone.overlap_ratio(NormalizedBox(0.75, 0.75, 0.2, 0.2)) == 0.0


def test_evidence_has_deterministic_zone_order_and_approach_tie_break() -> None:
    square = (Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1))
    zones = (
        Zone("z-approach", ZoneType.APPROACH, square),
        Zone("a-approach", ZoneType.APPROACH, square),
        Zone("soil", ZoneType.SOIL, square),
        Zone("blocked", ZoneType.NO_FIRE, square),
    )

    evidence = evidence_for_box(NormalizedBox(0.2, 0.2, 0.2, 0.2), zones)

    assert evidence.zone_id == "a-approach"
    assert evidence.approach_overlap == 1.0
    assert evidence.foliage_overlap == 0.0
    assert evidence.soil_overlap == 1.0
    assert evidence.no_fire_intersection
    assert [overlap.zone_id for overlap in evidence.overlaps] == [
        "a-approach",
        "blocked",
        "soil",
        "z-approach",
    ]


def test_load_zones_accepts_full_runtime_config(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scene": {
                    "zones": [
                        {
                            "id": "pot-1",
                            "type": "soil",
                            "plant_id": "plant-1",
                            "points": [
                                {"x": 0.1, "y": 0.1},
                                {"x": 0.9, "y": 0.1},
                                {"x": 0.5, "y": 0.9},
                            ],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    [zone] = load_zones(path)

    assert zone.zone_id == "pot-1"
    assert zone.zone_type is ZoneType.SOIL
    assert zone.plant_id == "plant-1"


def test_load_zones_rejects_self_intersection_and_duplicate_ids(tmp_path) -> None:
    bow_tie = {
        "zones": [
            {
                "id": "bad",
                "type": "approach",
                "points": [
                    {"x": 0, "y": 0},
                    {"x": 1, "y": 1},
                    {"x": 0, "y": 1},
                    {"x": 1, "y": 0},
                ],
            }
        ]
    }
    path = tmp_path / "zones.json"
    path.write_text(json.dumps(bow_tie), encoding="utf-8")
    with pytest.raises(ZoneError, match="self-intersects"):
        load_zones(path)

    triangle = {
        "id": "same",
        "type": "soil",
        "points": [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 0, "y": 1}],
    }
    path.write_text(json.dumps({"zones": [triangle, triangle]}), encoding="utf-8")
    with pytest.raises(ZoneError, match="duplicate zone id"):
        load_zones(path)
