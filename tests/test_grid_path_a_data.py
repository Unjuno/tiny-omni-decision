import numpy as np
import pytest


def _row(speaker, code, words):
    from scripts.lombard_readout_core import OPTIONS

    return {
        "speaker": speaker,
        "code": code,
        "targets": {slot: word for slot, word in zip(OPTIONS, words, strict=True)},
    }


def test_corrected_grid_split_returns_stable_ids_and_native_targets():
    from scripts.grid_path_a_data import validate_direct_split
    from scripts.lombard_readout_core import OPTIONS

    train = [_row("s10", "lwwc5a", [choices[0] for choices in OPTIONS.values()])]
    validation = [_row("s30", "bbbx2a", [choices[-1] for choices in OPTIONS.values()])]

    result = validate_direct_split(train, validation)

    assert result.train_ids == ["s10_lwwc5a"]
    assert result.validation_ids == ["s30_bbbx2a"]
    assert result.train_targets.shape == (1, 6)
    assert result.validation_targets.shape == (1, 6)


def test_corrected_grid_split_rejects_speaker_overlap():
    from scripts.grid_path_a_data import validate_direct_split
    from scripts.lombard_readout_core import OPTIONS

    words = [choices[0] for choices in OPTIONS.values()]
    other_words = [choices[-1] for choices in OPTIONS.values()]
    with pytest.raises(ValueError, match="speaker"):
        validate_direct_split([_row("s10", "lwwc5a", words)],
                              [_row("s10", "bbbx2a", other_words)])


def test_corrected_grid_split_rejects_repeated_asset_and_normalized_target():
    from scripts.grid_path_a_data import validate_direct_split
    from scripts.lombard_readout_core import OPTIONS

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
