from __future__ import annotations

import json

import pytest

from scripts.fetch_bsd10k_allowed_audio import _allowed_ids


def test_old_by_license_requires_explicit_opt_in(tmp_path):
    manifest = {
        "source_record_id": "19868804",
        "schema": "tiny-omni-decision/bsd10k-group-split-v1",
        "metadata_zip_sha256": "a" * 64,
        "examples": [
            {
                "sound_id": "1",
                "license": "CC-BY-3.0",
                "uploader_group_sha256": "b" * 64,
                "split": "train",
            }
        ],
    }
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="non-allowlisted"):
        _allowed_ids(manifest_path, "train")
    assert _allowed_ids(manifest_path, "train", {"CC-BY-3.0"}) == {"1"}
