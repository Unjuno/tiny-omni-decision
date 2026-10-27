import pytest

pytest.importorskip("torch")
from scripts.acquire_openimages_train_candidate_media import (  # noqa: E402
    compare_images,
    prepare_records,
)


def test_prepare_requires_matching_page_and_photo_identity():
    candidate = {"image_id": "0123456789abcdef", "target_label": "x"}
    receipt = {"ImageID": candidate["image_id"],
               "landing_page_check_status": "metadata_confirmed_on_landing_page",
               "page_license_and_attribution_match": True,
               "OriginalLandingURL": "https://www.flickr.com/photos/a/123",
               "OriginalURL": "https://live.staticflickr.com/1/123_abc.jpg"}
    assert len(prepare_records([candidate], [receipt])) == 1
    with pytest.raises(ValueError, match="identity"):
        prepare_records([candidate], [receipt | {"OriginalURL": "https://example.com/123_x.jpg"}])
    rejected = receipt | {"landing_page_check_status": "http_error_404"}
    assert prepare_records([candidate], [rejected]) == []


def test_detects_exact_and_near_but_skips_failed_media():
    train = [{"status": "saved", "image_id": "a", "normalized_rgb_sha256": "x",
              "perceptual_hash_64": "0000000000000000"}]
    validation = [{"status": "saved", "image_id": "b", "normalized_rgb_sha256": "x",
                   "perceptual_hash_64": "0000000000000001"}]
    assert compare_images(train, validation)[0]["phash_distance"] == 1
    assert compare_images([{"status": "failed"}], validation) == []
