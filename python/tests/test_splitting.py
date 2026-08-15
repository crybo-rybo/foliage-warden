from __future__ import annotations

import pytest
from foliage_warden_eval.schemas import DatasetItem
from foliage_warden_eval.split_cli import main
from foliage_warden_eval.splitting import split_by_session


def test_split_keeps_sessions_and_broader_groups_together() -> None:
    items = [
        DatasetItem("a1", "session-a", group_id="day-1"),
        DatasetItem("a2", "session-a", group_id="day-1"),
        DatasetItem("b1", "session-b", group_id="day-1"),
        DatasetItem("c1", "session-c", group_id="day-2"),
        DatasetItem("d1", "session-d", group_id="day-3"),
    ]
    result = split_by_session(items, seed="fixture")
    result.assert_no_leakage(items)
    assert result.session_assignments["session-a"] == result.session_assignments["session-b"]
    assert result.item_assignments["a1"] == result.item_assignments["a2"]
    assert result.to_dict() == split_by_session(reversed(items), seed="fixture").to_dict()


def test_split_apportions_whole_groups() -> None:
    items = [DatasetItem(f"item-{index}", f"session-{index}") for index in range(20)]
    result = split_by_session(items, ratios={"train": 0.7, "validation": 0.15, "test": 0.15})
    counts = result.to_dict()["counts"]["groups"]
    assert counts == {"test": 3, "train": 14, "validation": 3}


def test_split_rejects_conflicting_group_for_one_session() -> None:
    with pytest.raises(ValueError, match="conflicting groups"):
        split_by_session(
            [
                DatasetItem("one", "session", group_id="day-1"),
                DatasetItem("two", "session", group_id="day-2"),
            ]
        )


def test_split_rejects_invalid_ratios_and_duplicate_items() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        split_by_session([], ratios={"train": 0.8, "test": 0.3})
    with pytest.raises(ValueError, match="duplicate dataset"):
        split_by_session([DatasetItem("same", "a"), DatasetItem("same", "b")])


def test_split_cli_writes_assignments(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "split.json"
    manifest.write_text(
        '{"item_id":"one","session_id":"session-one"}\n'
        '{"item_id":"two","session_id":"session-two"}\n',
        encoding="utf-8",
    )
    assert main([str(manifest), "--output", str(output), "--seed", "test"]) == 0
    assert '"session-one"' in output.read_text(encoding="utf-8")
