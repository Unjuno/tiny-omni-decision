from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_physicalai_fresh_action_split import (
    build_fresh_manifest,
    validate_fresh_manifest,
)
from scripts.build_physicalai_stack_platforms_split import build_manifest
from scripts.download_physicalai_stack_platforms_pilot import download_pilot


def _asset_inventory(count: int, kind: str) -> list[dict[str, object]]:
    result = []
    for episode in range(count):
        payload = f"{kind}-{episode}".encode()
        digest = hashlib.sha256(payload).hexdigest()
        if kind == "video":
            path = (
                "panda-stack-platforms/videos/chunk-000/"
                "observation.images.world_camera/"
                f"episode_{episode:06d}.mp4"
            )
        else:
            path = f"panda-stack-platforms/data/chunk-000/episode_{episode:06d}.parquet"
        result.append({"path": path, "size": len(payload), "sha256": digest})
    return result


def _source_manifest() -> dict[str, object]:
    return build_manifest(
        _asset_inventory(40, "video"),
        _asset_inventory(40, "parquet"),
        total_episodes=40,
        seed=17,
        pilot_train_groups=4,
        pilot_validation_groups=3,
    )


def test_fresh_split_is_deterministic_and_disjoint_from_prior_exposure():
    source = _source_manifest()
    first = build_fresh_manifest(
        source,
        train_groups=8,
        validation_groups=4,
        reserved_groups=4,
    )
    second = build_fresh_manifest(
        source,
        train_groups=8,
        validation_groups=4,
        reserved_groups=4,
    )

    assert first["split_hashes"] == second["split_hashes"]
    assert first["episode_id_order_hashes"] == second["episode_id_order_hashes"]
    prior = set(source["pilot_selection"]["train_episode_indices"])
    prior.update(source["pilot_selection"]["validation_episode_indices"])
    selected = {
        name: set(first["pilot_selection"][f"{name}_episode_indices"])
        for name in ("train", "validation")
    }
    selected["reserved"] = set(first["reserved_development_episode_indices"])
    assert all(not (ids & prior) for ids in selected.values())
    assert selected["train"].isdisjoint(selected["validation"])
    assert selected["train"].isdisjoint(selected["reserved"])
    assert selected["validation"].isdisjoint(selected["reserved"])

    original = {row["episode_index"]: row["split"] for row in source["episode_assignments"]}
    fresh_rows = {row["episode_index"]: row for row in first["episode_assignments"]}
    for ids in selected.values():
        assert all(original[episode] == "train" for episode in ids)
    assert all(
        fresh_rows[episode]["source_split"] == "train"
        for ids in selected.values()
        for episode in ids
    )
    assert first["split_counts"] == {
        "train": 8,
        "validation": 4,
        "reserved_holdout": 4,
        "excluded": 24,
    }
    assert first["split_policy"]["source_validation_partition_groups_excluded"] == 6
    assert first["split_policy"]["source_reserved_partition_groups_protected"] == 6
    assert first["split_policy"]["source_pilot_validation_episode_ids_exposed_before"] == 3
    assert first["split_policy"]["reserved_payload_loaded"] is False


def test_fresh_split_rejects_prior_id_overlap_and_insufficient_groups():
    source = _source_manifest()
    bad = copy.deepcopy(source)
    bad["pilot_selection"]["validation_episode_indices"][0] = bad[
        "pilot_selection"
    ]["train_episode_indices"][0]
    with pytest.raises(ValueError, match="overlap or repeat"):
        build_fresh_manifest(bad, train_groups=1, validation_groups=1, reserved_groups=1)

    with pytest.raises(ValueError, match="only .* fresh source-train groups"):
        build_fresh_manifest(source, train_groups=30, validation_groups=10, reserved_groups=2)


def test_fresh_manifest_preserves_original_partition_labels_for_audit():
    source = _source_manifest()
    result = build_fresh_manifest(
        source,
        train_groups=8,
        validation_groups=4,
        reserved_groups=4,
    )
    by_id = {row["episode_index"]: row for row in result["episode_assignments"]}
    for row in source["episode_assignments"]:
        copied = by_id[row["episode_index"]]
        assert copied["source_split"] == row["split"]
    assert all(
        by_id[episode]["split"] == "reserved_holdout"
        for episode in result["reserved_development_episode_indices"]
    )


def test_fresh_manifest_validator_fails_closed_on_hash_or_reserve_mutation():
    result = build_fresh_manifest(
        _source_manifest(),
        train_groups=8,
        validation_groups=4,
        reserved_groups=4,
    )
    bad_hash = copy.deepcopy(result)
    bad_hash["split_hashes"]["train"] = "0" * 64
    with pytest.raises(ValueError, match="split hash mismatch"):
        validate_fresh_manifest(bad_hash)

    opened_reserve = copy.deepcopy(result)
    opened_reserve["split_policy"]["reserved_payload_loaded"] = True
    with pytest.raises(ValueError, match="reserved payload"):
        validate_fresh_manifest(opened_reserve)


def test_fresh_selection_downloads_only_new_train_and_validation(tmp_path, monkeypatch):
    videos = _asset_inventory(40, "video")
    parquets = _asset_inventory(40, "parquet")
    payloads = {
        item["path"]: f"{kind}-{episode}".encode()
        for kind, items in (("video", videos), ("parquet", parquets))
        for episode, item in enumerate(items)
    }
    source = build_manifest(
        videos,
        parquets,
        total_episodes=40,
        seed=17,
        pilot_train_groups=4,
        pilot_validation_groups=3,
    )
    manifest = build_fresh_manifest(
        source,
        train_groups=8,
        validation_groups=4,
        reserved_groups=4,
    )
    manifest_path = tmp_path / "fresh-split.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    class Response:
        def __init__(self, content: bytes):
            self.content = content

        def raise_for_status(self):
            return None

    def fake_get(url: str, *, headers: dict[str, str], timeout: tuple[int, int]):
        del headers
        del timeout
        path = "panda-stack-platforms/" + url.split("/panda-stack-platforms/", 1)[
            1
        ].split("?", 1)[0]
        return Response(payloads[path])

    monkeypatch.setattr(
        "scripts.download_physicalai_stack_platforms_pilot.requests.get", fake_get
    )
    output_dir = tmp_path / "download"
    result = download_pilot(manifest_path, output_dir, workers=2)
    receipt = json.loads(Path(result["receipt_path"]).read_text(encoding="utf-8"))

    expected_train = set(manifest["pilot_selection"]["train_episode_indices"])
    expected_validation = set(manifest["pilot_selection"]["validation_episode_indices"])
    expected_reserved = set(manifest["reserved_development_episode_indices"])
    fetched = {item["episode"] for item in receipt["files"]}
    assert result["files"] == 2 * (8 + 4)
    receipt_train = {
        item["episode"] for item in receipt["files"] if item["split"] == "train"
    }
    assert receipt_train == expected_train
    assert {
        item["episode"] for item in receipt["files"] if item["split"] == "validation"
    } == expected_validation
    assert fetched.isdisjoint(expected_reserved)
    assert receipt["reserved_holdout_payload_downloaded"] is False
