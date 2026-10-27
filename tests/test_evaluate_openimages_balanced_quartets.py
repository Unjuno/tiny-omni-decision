import hashlib

import pytest

from scripts.evaluate_openimages_balanced_quartets import prepare


def test_prepare_pins_every_image_and_page(tmp_path):
    groups, receipts, media = [], [], []
    for group_index in range(15):
        rows = []
        for label in "abcd":
            image_id = f"{group_index:02d}{label}"
            content = image_id.encode()
            (tmp_path / image_id).write_bytes(content)
            rows.append({"image_id": image_id, "target": label})
            receipts.append({"ImageID": image_id, "page_license_and_attribution_match": True})
            media.append({"image_id": image_id, "status": "saved", "local_path": image_id,
                          "media_sha256": hashlib.sha256(content).hexdigest()})
        groups.append({"options": list("abcd"), "rows": rows})
    rows, paths = prepare(groups, receipts, media, {x: x for x in "abcd"}, tmp_path)
    assert len(rows) == len(paths) == 60
    paths[0].write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        prepare(groups, receipts, media, {x: x for x in "abcd"}, tmp_path)
