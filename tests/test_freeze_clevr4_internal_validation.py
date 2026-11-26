from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_clevr4_content_overlap import sha256_file
from scripts.freeze_clevr4_internal_validation import freeze_pool


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    labels = tmp_path / "labels.jsonl"
    media = tmp_path / "media.jsonl"
    near = tmp_path / "near.json"
    old_images = tmp_path / "old"
    old_images.mkdir()
    rows = [{"id": f"CLEVR_{index}", "source_split": "train", "color": "red"} for index in range(6)]
    media_rows = [{"id": row["id"], "media_sha256": str(index)} for index, row in enumerate(rows)]
    _write_jsonl(labels, rows)
    _write_jsonl(media, media_rows)
    (old_images / "CLEVR_1.png").write_bytes(b"prior")
    near.write_text(
        json.dumps(
            {
                "candidate_manifest_sha256": sha256_file(media),
                "near_pairs": [{"candidate_id": "CLEVR_2"}],
            }
        ),
        encoding="utf-8",
    )
    return labels, media, near, old_images, tmp_path / "frozen"


def test_freeze_quarantines_old_id_and_near_pair_and_is_deterministic(tmp_path: Path) -> None:
    labels, media, near, old_images, output = _fixture(tmp_path)
    receipt = freeze_pool(
        labels_path=labels,
        media_manifest_path=media,
        prior_image_dirs=[old_images],
        near_pair_result_path=near,
        output=output,
        selector_count=2,
    )

    assert receipt["quarantined_count"] == 2
    assert receipt["quarantined_by_prior_id_only"] == 1
    assert receipt["quarantined_by_near_phash_only"] == 1
    selector = [
        json.loads(line)["id"]
        for line in (output / "selector-labels.jsonl").read_text().splitlines()
    ]
    report = [
        json.loads(line)["id"] for line in (output / "report-labels.jsonl").read_text().splitlines()
    ]
    assert len(selector) == 2
    assert len(report) == 2
    assert set(selector).isdisjoint(report)
    assert not ({"CLEVR_1", "CLEVR_2"} & (set(selector) | set(report)))
    assert (
        receipt["selector_id_order_sha256"]
        == hashlib.sha256(("\n".join(selector) + "\n").encode()).hexdigest()
    )


def test_freeze_rejects_near_audit_for_different_manifest(tmp_path: Path) -> None:
    labels, media, near, old_images, output = _fixture(tmp_path)
    media.write_text(media.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="different media manifest"):
        freeze_pool(
            labels_path=labels,
            media_manifest_path=media,
            prior_image_dirs=[old_images],
            near_pair_result_path=near,
            output=output,
            selector_count=2,
        )


def test_freeze_requires_selector_and_report_rows(tmp_path: Path) -> None:
    labels, media, near, old_images, output = _fixture(tmp_path)
    with pytest.raises(ValueError, match="leave nonempty"):
        freeze_pool(
            labels_path=labels,
            media_manifest_path=media,
            prior_image_dirs=[old_images],
            near_pair_result_path=near,
            output=output,
            selector_count=4,
        )
