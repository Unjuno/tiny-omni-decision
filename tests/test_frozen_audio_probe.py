import hashlib
import struct
import wave
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch

from scripts.train_frozen_audio_probe import (
    LABELS,
    _option_order_stability,
    _validate_splits,
)
from scripts.train_frozen_text_probe import CandidateScorer

SOURCE = "google/speech_commands@a751309c0fd613e8a5d30d77900f30e8b42bc2da"


def _sample_rows(tmp_path: Path, split: str, speaker_prefix: str) -> list[dict[str, object]]:
    rows = []
    for index, label in enumerate(LABELS):
        relative = Path("media") / split / f"{index}.wav"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(struct.pack("<h", index + (1 if split == "train" else 20)) * 16000)
        rows.append(
            {
                "id": f"{split}-{index}",
                "split": split,
                "source": SOURCE,
                "speaker_group_sha256": hashlib.sha256(
                    f"{speaker_prefix}-{index}".encode()
                ).hexdigest(),
                "media_path": relative.as_posix(),
                "media_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "target": label,
            }
        )
    return rows


def test_audio_probe_accepts_balanced_speaker_disjoint_splits(tmp_path: Path):
    train = _sample_rows(tmp_path, "train", "train-speaker")
    validation = _sample_rows(tmp_path, "validation", "validation-speaker")
    _validate_splits(tmp_path, train, validation)


def test_audio_probe_rejects_speaker_overlap(tmp_path: Path):
    train = _sample_rows(tmp_path, "train", "train-speaker")
    validation = _sample_rows(tmp_path, "validation", "validation-speaker")
    validation[0]["speaker_group_sha256"] = train[0]["speaker_group_sha256"]
    with pytest.raises(ValueError, match="speaker groups overlap"):
        _validate_splits(tmp_path, train, validation)


def test_audio_probe_rejects_media_hash_mismatch(tmp_path: Path):
    train = _sample_rows(tmp_path, "train", "train-speaker")
    validation = _sample_rows(tmp_path, "validation", "validation-speaker")
    train[0]["media_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _validate_splits(tmp_path, train, validation)


def test_audio_candidate_scores_are_stable_under_option_reordering():
    torch.manual_seed(17)
    scorer = CandidateScorer(hidden_size=4 * 8).eval()
    results = _option_order_stability(
        scorer,
        torch.randn(3, 8),
        torch.randn(10, 8),
        [0, 4, 9],
        torch.device("cpu"),
    )
    assert set(results) == {"reverse", "rotate", "fixed_shuffle"}
    assert all(
        result["max_abs_logit_delta_after_option_id_alignment"] <= 1e-6
        for result in results.values()
    )
    assert all(result["accuracy_after_reordering"] >= 0 for result in results.values())
