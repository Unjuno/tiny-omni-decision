import pytest


def test_reference_wording_distinguishes_recomputation_from_stored_logits():
    from scripts.correct_native_av_publication import correct_reference

    original = "All selected native CPU video frame features and raw logits match the original\nnative CPU bank exactly."
    corrected = correct_reference(original, 2e-6)
    assert "same-head recomputation" in corrected
    assert "2e-06" in corrected
    assert "raw logits match the original" not in corrected
    with pytest.raises(ValueError):
        correct_reference("unexpected source text", 2e-6)
