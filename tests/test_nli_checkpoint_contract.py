import pytest

from scripts.evaluate_nli_condition_probe import checkpoint_contract


def test_native_entailment_indices_are_pinned():
    assert checkpoint_contract("minilm2")[2] == 1
    assert checkpoint_contract("deberta_xsmall")[2] == 0
    assert len(checkpoint_contract("deberta_xsmall")[0]) == 64
    with pytest.raises(ValueError):
        checkpoint_contract("unknown")
