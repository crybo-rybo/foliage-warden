from __future__ import annotations

from pathlib import Path

from foliage_warden_sim.resources import (
    checkout_contract_root,
    default_contract_root,
    packaged_contract_root,
)


def _files(root: Path, directory: str) -> dict[str, bytes]:
    base = root / directory
    return {
        str(path.relative_to(base)): path.read_bytes()
        for path in sorted(base.rglob("*"))
        if path.is_file() and path.suffix == ".json"
    }


def test_packaged_contracts_are_byte_identical_to_checkout(repository: Path) -> None:
    assert checkout_contract_root() == repository
    assert default_contract_root() == repository
    packaged = packaged_contract_root()
    assert packaged != repository
    for directory in ("scenarios", "schemas", "config"):
        assert _files(packaged, directory) == _files(repository, directory)
