from scripts.physicalai_pilot_receipts import split_episode_ids


def test_split_episode_ids_reads_acquisition_receipt_selection() -> None:
    receipt = {
        "selection": {
            "train_episode_indices": [8510, 3736],
            "validation_episode_indices": [5491],
        },
        "files": [],
    }

    assert split_episode_ids(receipt) == ({8510, 3736}, {5491})


def test_split_episode_ids_keeps_legacy_flat_receipts_compatible() -> None:
    receipt = {"train_ids": [1, 2], "validation_ids": [3]}

    assert split_episode_ids(receipt) == ({1, 2}, {3})
