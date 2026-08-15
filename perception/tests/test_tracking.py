from __future__ import annotations

import pytest

from foliage_warden_perception.tracking import IouTracker
from foliage_warden_perception.types import Detection, NormalizedBox, ObjectClass


def _detection(
    object_class: ObjectClass = ObjectClass.CAT,
    box: NormalizedBox | None = None,
    prediction_index: int = 0,
) -> Detection:
    return Detection(
        object_class=object_class,
        class_id=15 if object_class is ObjectClass.CAT else 0,
        confidence=0.9,
        bbox=box if box is not None else NormalizedBox(0.1, 0.1, 0.2, 0.2),
        prediction_index=prediction_index,
    )


def test_track_id_age_and_quality_mature_deterministically() -> None:
    tracker = IouTracker(iou_threshold=0.3)

    [first] = tracker.update([_detection()], frame_index=0, timestamp_ms=0)
    [second] = tracker.update([_detection()], frame_index=1, timestamp_ms=33)
    [third] = tracker.update([_detection()], frame_index=2, timestamp_ms=67)

    assert first.track_id == second.track_id == third.track_id == "cat-000001"
    assert [first.age_frames, second.age_frames, third.age_frames] == [1, 2, 3]
    assert [first.age_ms, second.age_ms, third.age_ms] == [0, 33, 67]
    assert first.quality == pytest.approx(1 / 3)
    assert second.quality == pytest.approx(2 / 3)
    assert third.quality == 1.0


def test_track_can_reacquire_with_quality_penalty_then_expires() -> None:
    tracker = IouTracker(iou_threshold=0.3, max_missed_frames=1)
    [first] = tracker.update([_detection()], frame_index=0, timestamp_ms=0)
    assert tracker.update([], frame_index=1, timestamp_ms=33) == []
    [reacquired] = tracker.update([_detection()], frame_index=2, timestamp_ms=67)
    assert reacquired.track_id == first.track_id
    assert reacquired.quality == pytest.approx(4 / 9)

    tracker.update([], frame_index=3, timestamp_ms=100)
    tracker.update([], frame_index=4, timestamp_ms=133)
    [replacement] = tracker.update([_detection()], frame_index=5, timestamp_ms=167)
    assert replacement.track_id == "cat-000002"


def test_tracker_never_associates_different_classes() -> None:
    tracker = IouTracker(iou_threshold=0.0)
    [cat] = tracker.update([_detection()], frame_index=0, timestamp_ms=0)
    [person] = tracker.update(
        [_detection(ObjectClass.PERSON)],
        frame_index=1,
        timestamp_ms=33,
    )

    assert cat.track_id == "cat-000001"
    assert person.track_id == "person-000002"


def test_competing_matches_are_marked_ambiguous_with_stable_assignment() -> None:
    tracker = IouTracker(iou_threshold=0.1)
    tracker.update(
        [
            _detection(box=NormalizedBox(0.1, 0.1, 0.2, 0.2), prediction_index=0),
            _detection(box=NormalizedBox(0.2, 0.1, 0.2, 0.2), prediction_index=1),
        ],
        frame_index=0,
        timestamp_ms=0,
    )

    [result] = tracker.update(
        [_detection(box=NormalizedBox(0.15, 0.1, 0.2, 0.2))],
        frame_index=1,
        timestamp_ms=33,
    )

    assert result.track_id == "cat-000001"
    assert result.ambiguous


def test_tracker_rejects_non_monotonic_updates() -> None:
    tracker = IouTracker()
    tracker.update([], frame_index=1, timestamp_ms=10)
    with pytest.raises(ValueError, match="strictly increasing"):
        tracker.update([], frame_index=1, timestamp_ms=11)
