import pytest

from scripts.train_soundactions_temporal_readout import should_stop, validation_improved


def test_validation_checkpoint_rule_keeps_first_exact_tie():
    assert validation_improved(0.5, 0.6)
    assert not validation_improved(0.5, 0.5)
    assert not validation_improved(0.6, 0.5)


def test_early_stop_uses_full_configured_patience():
    assert not should_stop(100, 1, patience=100)
    assert should_stop(101, 1, patience=100)
    with pytest.raises(ValueError, match="positive"):
        should_stop(0, 1, patience=100)
