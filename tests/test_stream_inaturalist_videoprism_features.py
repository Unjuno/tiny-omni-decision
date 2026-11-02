import pytest

from scripts.stream_inaturalist_videoprism_features import _reject_media_overlap


def _media_row(photo_id="new-photo", phash="ffffffffffffffff"):
    return {
        "photo_id": photo_id,
        "split": "train",
        "download_sha256": "raw-new",
        "media_sha256": "media-new",
        "normalized_rgb_sha256": "rgb-new",
        "perceptual_hash_64": phash,
    }


def test_media_overlap_gate_tracks_exact_hashes_and_perceptual_hashes():
    seen_exact = set()
    seen_phashes = []

    _reject_media_overlap(
        _media_row(), {"old-raw", "old-media", "old-rgb"}, {0}, seen_exact, seen_phashes
    )

    assert seen_exact == {"raw-new", "media-new", "rgb-new"}
    assert seen_phashes == [(0xFFFFFFFFFFFFFFFF, "train", "new-photo")]
    with pytest.raises(ValueError, match="exact media hash overlap"):
        _reject_media_overlap(
            _media_row("second")
            | {
                "download_sha256": "raw-second",
                "media_sha256": "media-new",
                "normalized_rgb_sha256": "rgb-second",
                "perceptual_hash_64": "ffffffffffffffff",
            },
            set(),
            set(),
            seen_exact,
            seen_phashes,
        )


@pytest.mark.parametrize("phash", ["0000000000000000", "000000000000000f"])
def test_media_overlap_gate_rejects_near_duplicates_without_filtering(phash):
    with pytest.raises(ValueError, match="perceptual hash distance <=4"):
        _reject_media_overlap(_media_row(phash=phash), set(), {0}, set(), [])
