import numpy as np
import pytest

from scripts.grid_path_a_data import OPTIONS


def _row(speaker, code, words):
    return {
        "speaker": speaker,
        "code": code,
        "targets": {slot: word for slot, word in zip(OPTIONS, words, strict=True)},
    }


def test_corrected_grid_split_returns_stable_ids_and_native_targets():
    from scripts.grid_path_a_data import validate_direct_split
    train = [_row("s10", "lwwc5a", [choices[0] for choices in OPTIONS.values()])]
    validation = [_row("s30", "bbbx2a", [choices[-1] for choices in OPTIONS.values()])]

    result = validate_direct_split(train, validation)

    assert result.train_ids == ["s10_lwwc5a"]
    assert result.validation_ids == ["s30_bbbx2a"]
    assert result.train_targets.shape == (1, 6)
    assert result.validation_targets.shape == (1, 6)


def test_corrected_grid_split_rejects_speaker_overlap():
    from scripts.grid_path_a_data import validate_direct_split
    words = [choices[0] for choices in OPTIONS.values()]
    other_words = [choices[-1] for choices in OPTIONS.values()]
    with pytest.raises(ValueError, match="speaker"):
        validate_direct_split([_row("s10", "lwwc5a", words)],
                              [_row("s10", "bbbx2a", other_words)])


def test_corrected_grid_split_rejects_repeated_asset_and_normalized_target():
    from scripts.grid_path_a_data import validate_direct_split
    words = [choices[0] for choices in OPTIONS.values()]
    with pytest.raises(ValueError, match="asset"):
        validate_direct_split([_row("s10", "lwwc5a", words)],
                              [_row("s10", "lwwc5a", words)])
    with pytest.raises(ValueError, match="target"):
        validate_direct_split([_row("s10", "lwwc5a", words)],
                              [_row("s30", "bbbx2a", words)])


def test_feature_bank_requires_exact_id_order_finite_fp32_and_dimensions():
    from scripts.grid_path_a_data import validate_feature_bank

    expected = ["s10_lwwc5a", "s30_bbbx2a"]
    bank = np.zeros((2, 384), dtype=np.float32)
    validate_feature_bank(expected, expected, bank, width=384, name="audio")
    with pytest.raises(ValueError, match="order"):
        validate_feature_bank(expected[::-1], expected, bank, width=384, name="audio")
    with pytest.raises(ValueError, match="finite"):
        validate_feature_bank(expected, expected, bank * np.nan, width=384, name="audio")
    with pytest.raises(ValueError, match="shape"):
        validate_feature_bank(expected, expected, bank[:, :8], width=384, name="audio")


def test_balanced_quartet_selection_is_deterministic_and_asset_disjoint():
    from scripts.grid_path_a_data import select_balanced_quartets

    codes = ["aa0000", "aa0011", "aa0101", "aa0110"]
    rows = [_row("s10", code, [choices[0] for choices in OPTIONS.values()]) for code in codes]

    first = select_balanced_quartets(rows, split="train", speakers=["s10"], per_speaker=1)
    second = select_balanced_quartets(rows, split="train", speakers=["s10"], per_speaker=1)

    assert first == second
    assert first[0]["codes"] == codes
    assert first[0]["split"] == "train"
    assert set(first[0]["targets"]) == set(codes)


def test_balanced_quartet_selection_excludes_previously_observed_media():
    from scripts.grid_path_a_data import select_balanced_quartets

    codes = ["aa0000", "aa0011", "aa0101", "aa0110"]
    rows = [_row("s30", code, [choices[0] for choices in OPTIONS.values()]) for code in codes]

    with pytest.raises(ValueError, match="only 0/1"):
        select_balanced_quartets(
            rows,
            split="validation",
            speakers=["s30"],
            excluded_codes={("s30", codes[0])},
            per_speaker=1,
        )
