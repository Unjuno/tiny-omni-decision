import base64
import hashlib

import pytest

pytest.importorskip("torch")
from scripts.retry_openimages_quartet_media import prepare_retries, verify_original  # noqa: E402


def test_retry_preserves_photo_and_original_md5():
    rows = [{"image_id": "0123456789abcdef", "target_label": "a", "status": "failed_HTTPError",
             "url": "https://live.staticflickr.com/1/123_x.jpg", "error": "HTTP Error 410: Gone"},
            {"image_id": "fedcba9876543210", "target_label": "b", "status": "failed_HTTPError",
             "url": "https://live.staticflickr.com/2/456_o.jpg",
             "error": "HTTP Error 429: Too Many Requests"}]
    digest = base64.b64encode(b"x" * 16).decode()
    pages = [{"ImageID": rows[0]["image_id"], "Thumbnail300KURL": rows[0]["url"],
              "OriginalURL": "https://live.staticflickr.com/1/123_o.jpg", "OriginalMD5": digest,
              "OriginalLandingURL": "https://flickr.com/photos/u/123",
              "landing_page_check_status": "metadata_confirmed_on_landing_page",
              "page_license_and_attribution_match": True},
             {"ImageID": rows[1]["image_id"], "Thumbnail300KURL": None,
              "OriginalURL": rows[1]["url"], "OriginalMD5": digest,
              "OriginalLandingURL": "https://flickr.com/photos/u/456",
              "landing_page_check_status": "metadata_confirmed_on_landing_page",
              "page_license_and_attribution_match": True}]
    retries = prepare_retries(rows, pages)
    assert retries[0]["url"] == pages[0]["OriginalURL"]
    assert retries[1]["url"] == rows[1]["url"]
    pages[0]["OriginalURL"] = "https://live.staticflickr.com/1/999_o.jpg"
    with pytest.raises(ValueError, match="identity"):
        prepare_retries(rows, pages)


def test_original_bytes_must_match_metadata(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"bytes")
    expected = base64.b64encode(hashlib.md5(b"bytes", usedforsecurity=False).digest()).decode()
    assert verify_original(image, expected)
    image.write_bytes(b"different")
    assert not verify_original(image, expected)
