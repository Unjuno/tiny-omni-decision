from scripts.audit_greatest_hits_av_match_inputs import cross_split_overlap_count


def test_raw_hash_overlap_is_checked_within_modality():
    rows = [
        {"modality": "audio", "split": "train", "sha256": "same"},
        {"modality": "video", "split": "validation", "sha256": "same"},
        {"modality": "audio", "split": "validation", "sha256": "same"},
    ]

    assert cross_split_overlap_count(rows, "sha256") == 1
