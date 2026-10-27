import pytest

pytest.importorskip("torch")
from scripts.acquire_openimages_quartet_media import page_matched_quartets  # noqa: E402


def test_page_matching_requires_all_four_and_no_substitution():
    group = {"options": list("abcd"), "rows": [
        {"image_id": label, "target": label} for label in "abcd"]}
    receipts = [{"ImageID": label, "page_license_and_attribution_match": label != "d"}
                for label in "abcd"]
    assert page_matched_quartets([group], receipts) == []
    receipts[-1]["page_license_and_attribution_match"] = True
    assert page_matched_quartets([group], receipts) == [group]


def test_rejects_duplicate_image_and_missing_target():
    group = {"options": list("abcd"), "rows": [
        {"image_id": label, "target": label} for label in "abcd"]}
    receipts = [{"ImageID": label, "page_license_and_attribution_match": True} for label in "abcd"]
    with pytest.raises(ValueError, match="duplicate"):
        page_matched_quartets([group, group], receipts)
    group["rows"][0]["target"] = "b"
    with pytest.raises(ValueError, match="symmetric"):
        page_matched_quartets([group], receipts)
