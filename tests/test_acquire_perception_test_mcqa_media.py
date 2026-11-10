from __future__ import annotations

from pathlib import Path

import pytest

from scripts.acquire_perception_test_mcqa_media import acquire_media


def test_refuses_modified_preflight_before_creating_output(tmp_path):
    preflight = tmp_path / "preflight.json"
    preflight.write_text("{}", encoding="utf-8")
    output = Path("D:/CodexArtifacts/test-new-output")
    with pytest.raises(ValueError, match="preflight report hash mismatch"):
        acquire_media(
            corpus_dir=tmp_path / "missing-corpus",
            preflight_path=preflight,
            expected_preflight_sha256="0" * 64,
            output_root=output,
            min_free_bytes=0,
        )
    assert not output.exists()


def test_refuses_existing_output_without_touching_it(tmp_path):
    output = Path("D:/CodexArtifacts/test-existing-output")
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        acquire_media(
            corpus_dir=tmp_path / "missing-corpus",
            preflight_path=tmp_path / "missing-preflight",
            expected_preflight_sha256="0" * 64,
            output_root=output,
            min_free_bytes=0,
        )
    assert marker.read_text(encoding="utf-8") == "preserve"
