import pytest

from scripts.train_banking77_minilm_l6_readout import verify_model_snapshot


def test_pinned_model_snapshot_fails_closed_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Pinned model file is missing"):
        verify_model_snapshot(tmp_path)


def test_pinned_model_snapshot_fails_closed_on_hash_mismatch(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Pinned MiniLM-L6 file hash mismatch"):
        verify_model_snapshot(tmp_path)
