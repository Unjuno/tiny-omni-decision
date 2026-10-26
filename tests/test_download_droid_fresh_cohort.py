import json
from pathlib import Path

import pytest

from scripts.download_droid_fresh_cohort import validate_manifest

MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "manifests/pretrained_reuse/droid-fresh-next-action-seed17-20261026.json"
)


def test_fresh_cohort_manifest_is_pinned_and_disjoint_from_previous_shards():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    selected = {row["index"] for row in manifest["selected_shards"]}
    assert selected.isdisjoint(manifest["excluded_previous_shards"])
    assert manifest["expected_episode_count"] == 428


def test_fresh_cohort_manifest_rejects_previously_used_shard():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["selected_shards"][0]["index"] = 110
    with pytest.raises(ValueError, match="frozen nine-shard selection"):
        validate_manifest(manifest)


def test_fresh_cohort_manifest_rejects_size_or_object_tampering():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["selected_shards"][0]["size"] += 1
    with pytest.raises(ValueError, match="object sizes"):
        validate_manifest(manifest)
