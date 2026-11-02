from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from scripts.verify_avatar_fusion_runtime import run_verification, sha256_file
from tiny_omni_decision.avatar_fusion_runtime import AvatarAVDecisionRuntime


def _runtime():
    torch.manual_seed(11)
    return AvatarAVDecisionRuntime(
        audio_mean=np.zeros(384, dtype=np.float32),
        audio_std=np.ones(384, dtype=np.float32),
        video_mean=np.zeros(768, dtype=np.float32),
        video_std=np.ones(768, dtype=np.float32),
    )


def _inputs():
    rng = np.random.default_rng(17)
    return (
        rng.normal(size=384).astype(np.float32),
        rng.normal(size=(4, 768)).astype(np.float32),
        ["clip-a", "clip-b", "clip-c", "clip-d"],
        ["Candidate A", "Candidate B", "Candidate C", "Candidate D"],
    )


def test_runtime_returns_finite_option_id_probability_json_and_is_permutation_equivariant():
    runtime = _runtime()
    audio, videos, option_ids, option_texts = _inputs()

    original = runtime.decide(audio, videos, option_ids, option_texts, request_id="q-1")
    permutation = [2, 0, 3, 1]
    permuted = runtime.decide(
        audio,
        videos[permutation],
        [option_ids[i] for i in permutation],
        [option_texts[i] for i in permutation],
        request_id="q-1",
    )

    assert original["schema_version"] == 1
    assert original["selected_option_id"] in option_ids
    assert [item["id"] for item in original["options"]] == option_ids
    probabilities = [item["probability"] for item in original["options"]]
    assert np.isfinite(probabilities).all()
    assert sum(probabilities) == pytest.approx(1.0)
    original_by_id = {item["id"]: item["probability"] for item in original["options"]}
    permuted_by_id = {item["id"]: item["probability"] for item in permuted["options"]}
    assert permuted_by_id == pytest.approx(original_by_id, abs=1e-7)


def test_runtime_abstains_when_required_audio_is_missing():
    runtime = _runtime()
    _, videos, option_ids, option_texts = _inputs()

    result = runtime.decide(None, videos, option_ids, option_texts, request_id="q-missing")

    assert result["status"] == "abstain"
    assert result["selected_option_id"] is None
    assert all(item["probability"] is None for item in result["options"])


def test_runtime_rejects_nonfinite_and_wrong_width_features():
    runtime = _runtime()
    audio, videos, option_ids, option_texts = _inputs()

    with pytest.raises(ValueError, match="384"):
        runtime.decide(audio[:12], videos, option_ids, option_texts, request_id="q-bad")
    audio[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        runtime.decide(audio, videos, option_ids, option_texts, request_id="q-nan")


def test_runtime_load_is_hash_checked_frozen_and_exactly_parameter_counted(tmp_path):
    runtime = _runtime()
    checkpoint = tmp_path / "readout.pt"
    normalization = tmp_path / "normalization.json"
    torch.save(runtime.model.state_dict(), checkpoint)
    normalization.write_text(
        json.dumps(
            {
                "audio_mean": [0.0] * 384,
                "audio_std": [1.0] * 384,
                "video_mean": [0.0] * 768,
                "video_std": [1.0] * 768,
            }
        ),
        encoding="utf-8",
    )
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    normalization_sha = hashlib.sha256(normalization.read_bytes()).hexdigest()

    loaded = AvatarAVDecisionRuntime.load(
        checkpoint,
        normalization,
        expected_checkpoint_sha256=checkpoint_sha,
        expected_normalization_sha256=normalization_sha,
    )

    assert loaded.parameter_count == 73_857
    assert all(not parameter.requires_grad for parameter in loaded.model.parameters())
    audio, videos, option_ids, option_texts = _inputs()
    loaded_result = loaded.decide(audio, videos, option_ids, option_texts, request_id="q-1")
    original_result = runtime.decide(audio, videos, option_ids, option_texts, request_id="q-1")
    assert loaded_result == original_result
    with pytest.raises(ValueError, match="SHA-256"):
        AvatarAVDecisionRuntime.load(
            checkpoint,
            normalization,
            expected_checkpoint_sha256="0" * 64,
            expected_normalization_sha256=normalization_sha,
        )


def test_train_only_verifier_checks_hashes_reloads_and_writes_json(tmp_path):
    runtime = _runtime()
    checkpoint = tmp_path / "readout.pt"
    normalization = tmp_path / "normalization.json"
    features_path = tmp_path / "features.npz"
    manifest_path = tmp_path / "manifest.json"
    output = tmp_path / "result"
    torch.save(runtime.model.state_dict(), checkpoint)
    normalization.write_text(
        json.dumps(
            {
                "audio_mean": [0.0] * 384,
                "audio_std": [1.0] * 384,
                "video_mean": [0.0] * 768,
                "video_std": [1.0] * 768,
            }
        ),
        encoding="utf-8",
    )
    audio, video, option_ids, option_texts = _inputs()
    np.savez_compressed(
        features_path,
        audio=np.stack([audio]),
        video=np.stack([video]),
    )
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "example_id": "train-1",
                    "option_ids": option_ids,
                    "option_texts": option_texts,
                    "target_option_id": option_ids[0],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = run_verification(
        checkpoint_path=checkpoint,
        normalization_path=normalization,
        features_path=features_path,
        manifest_path=manifest_path,
        output_dir=output,
        expected_checkpoint_sha256=sha256_file(checkpoint),
        expected_normalization_sha256=sha256_file(normalization),
        expected_features_sha256=sha256_file(features_path),
        expected_manifest_sha256=sha256_file(manifest_path),
    )

    assert result["status"] == "PASS"
    assert result["classification"] == "TRAIN_ONLY_RUNTIME_INTEGRATION_NOT_QUALITY_EVALUATION"
    assert result["sample_count"] == 1
    assert result["encoder_calls"] == 0
    assert result["runtime_parameter_count"] == 73_857
    assert result["option_reorder_max_probability_delta"] <= 1e-6
    assert result["missing_audio_abstained"] is True
    assert result["missing_video_abstained"] is True
    assert (output / "predictions.jsonl").is_file()
    with pytest.raises(FileExistsError, match="overwrite"):
        run_verification(
            checkpoint_path=checkpoint,
            normalization_path=normalization,
            features_path=features_path,
            manifest_path=manifest_path,
            output_dir=output,
            expected_checkpoint_sha256=sha256_file(checkpoint),
            expected_normalization_sha256=sha256_file(normalization),
            expected_features_sha256=sha256_file(features_path),
            expected_manifest_sha256=sha256_file(manifest_path),
        )
