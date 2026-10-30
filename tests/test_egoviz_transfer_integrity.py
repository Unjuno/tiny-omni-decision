import pytest

from scripts.egoviz_transfer_integrity import validate_isolation


def test_rejects_each_kind_of_source_overlap():
    pilot = [{"split": split, "group_id": f"{split}{i}",
              "chunk_id": f"{split}-chunk{i}", "sample_id": f"{split}-sample{i}"}
             for split in ("train", "development") for i in range(8)]
    current = [{"split": "train", "group_id": "new", "chunk_id": "new-chunk",
                "sample_id": "new-sample"}]
    assert validate_isolation(pilot, current)["group_overlap"] == 0
    for field in ("group_id", "chunk_id", "sample_id"):
        contaminated = [dict(current[0], **{field: pilot[0][field]})]
        with pytest.raises(ValueError, match=field):
            validate_isolation(pilot, contaminated)
