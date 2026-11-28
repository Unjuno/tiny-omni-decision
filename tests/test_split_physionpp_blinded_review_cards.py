import csv
import hashlib
import json

import pytest
from PIL import Image

from scripts.split_physionpp_blinded_review_cards import (
    ANNOTATION_COLUMNS,
    _card_regions,
    build_review_cards,
)


def test_card_regions_split_each_paired_row_without_overlap():
    regions = _card_regions((2048, 2070 * 3))

    assert regions == [
        (0, 0, 2048, 2070),
        (0, 2070, 2048, 4140),
        (0, 4140, 2048, 6210),
    ]


@pytest.mark.parametrize("size", [(2047, 2070), (2048, 2069), (2048, 2071)])
def test_card_regions_reject_unexpected_sheet_dimensions(size):
    with pytest.raises(ValueError, match="unexpected contact-sheet dimensions"):
        _card_regions(size)


def test_build_review_cards_keeps_blind_package_without_mapping(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    image_hashes = {}
    for index in range(75):
        path = source / f"sheet-{index:03d}.png"
        Image.new("RGB", (2048, 2070), (index, 0, 0)).save(path)
        image_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (source / "review-pack-report.json").write_text(
        json.dumps(
            {
                "status": "complete_train_only_blinded_review_pack",
                "labels_and_sample_ids_in_images": False,
                "pair_count": 75,
                "frames_per_clip": 32,
                "image_sha256": image_hashes,
            }
        ),
        encoding="utf-8",
    )
    template = source / "annotation-template.csv"
    with template.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=ANNOTATION_COLUMNS)
        writer.writeheader()
        writer.writerows({"review_id": f"review-{i:03d}"} for i in range(1, 76))

    output = tmp_path / "cards"
    result = build_review_cards(source, output, seed=17)
    manifest = json.loads((output / "review-card-manifest.json").read_text())

    assert result["pair_count"] == 75
    assert result["source_to_card_mapping_saved"] is False
    assert len(list((output / "cards").glob("*.png"))) == 75
    assert len(manifest["cards_sha256"]) == 75
    assert len(list(csv.DictReader((output / "annotation-template.csv").open()))) == 75
    assert not (output / "answer-key.jsonl").exists()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_review_cards(source, output, seed=17)
