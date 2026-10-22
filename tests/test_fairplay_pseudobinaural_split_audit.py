import pytest


def _metadata():
    def entries(train, val):
        return {
            "train": {"rows": len(train), "ids": train},
            "val": {"rows": len(val), "ids": val},
        }

    fair = entries(
        ["000001.wav", "000002.wav", "000003.wav", "000004.wav"],
        ["000005.wav", "000006.wav"],
    )
    pseudo = {
        "1": entries(
            ["000001.mp3", "000002.mp3", "000005.mp3", "000006.mp3"],
            ["000003.mp3", "000004.mp3"],
        ),
        "2": entries(
            ["000001.mp3", "000003.mp3", "000005.mp3", "000006.mp3"],
            ["000002.mp3", "000004.mp3"],
        ),
        "3": entries(
            ["000001.mp3", "000002.mp3", "000004.mp3", "000006.mp3"],
            ["000003.mp3", "000005.mp3"],
        ),
        "4": entries(
            ["000001.mp3", "000002.mp3", "000003.mp3", "000006.mp3"],
            ["000004.mp3", "000005.mp3"],
        ),
        "5": entries(
            ["000001.mp3", "000003.mp3", "000004.mp3", "000005.mp3"],
            ["000002.mp3", "000006.mp3"],
        ),
    }
    return {
        "sources": {
            "fair_play": {"original_random_splits_train_validation_only": {"1": fair}},
            "pseudobinaural": {
                "train_validation_only": pseudo,
                "author_states_five_splits_non-overlapping_by_scene": True,
            },
        },
        "audit_scope": {
            "test_split_content_read": False,
            "media_bytes_downloaded": False,
            "models_loaded": False,
            "features_extracted": False,
            "training_or_evaluation": False,
        },
    }


def test_split_audit_normalizes_ids_and_compares_random_validation_to_group_split():
    from scripts.audit_fairplay_pseudobinaural_splits import audit_splits

    report = audit_splits(
        _metadata(),
        {"split": "PseudoBinaural split1 train only", "ids": ["000001", "000002"]},
    )

    split1 = report["pseudobinaural_splits"]["1"]
    assert split1["train_count"] == 4
    assert split1["validation_count"] == 2
    assert split1["train_validation_overlap_count"] == 0
    assert split1["preflight_ids_all_in_train"] is True
    assert split1["preflight_ids_in_validation"] == []
    assert report["original_random_split1_vs_pseudobinaural_split1"][
        "original_val_in_pseudo_train_count"
    ] == 2
    assert report["original_random_split1_vs_pseudobinaural_split1"][
        "shared_asset_id_count"
    ] == 6
    assert report["test_content_read"] is False
    assert report["validation_group_gate"] == {
        "required_independent_groups": 200,
        "validation_record_count": 2,
        "independent_scene_group_count": None,
        "maximum_possible_independent_scene_groups": 2,
        "passes_minimum_group_count": False,
    }


def test_split_audit_reports_cross_split_reuse_without_calling_it_independent_folds():
    from scripts.audit_fairplay_pseudobinaural_splits import audit_splits

    report = audit_splits(
        _metadata(),
        {"split": "PseudoBinaural split1 train only", "ids": ["000001"]},
    )

    assert report["cross_split_policy"] == "overlapping split variants; not independent folds"
    assert report["pseudobinaural_splits"]["1"]["other_splits_overlap_ids"] > 0


def test_split_audit_fails_if_train_and_validation_share_a_normalized_asset_id():
    from scripts.audit_fairplay_pseudobinaural_splits import audit_splits

    metadata = _metadata()
    metadata["sources"]["pseudobinaural"]["train_validation_only"]["1"]["val"][
        "ids"
    ].append("000001.wav")
    metadata["sources"]["pseudobinaural"]["train_validation_only"]["1"]["val"][
        "rows"
    ] += 1

    with pytest.raises(ValueError, match="train/validation asset overlap"):
        audit_splits(
            metadata,
            {"split": "PseudoBinaural split1 train only", "ids": ["000001"]},
        )


def test_split_audit_rejects_metadata_that_says_test_content_was_read():
    from scripts.audit_fairplay_pseudobinaural_splits import audit_splits

    metadata = _metadata()
    metadata["audit_scope"]["test_split_content_read"] = True

    with pytest.raises(ValueError, match="test split content must remain unread"):
        audit_splits(
            metadata,
            {"split": "PseudoBinaural split1 train only", "ids": ["000001"]},
        )
