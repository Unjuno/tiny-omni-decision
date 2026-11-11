import json
from types import SimpleNamespace

import pytest

from scripts.cohort_gate import CohortUseBlocked


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_audio_training_blocks_unknown_history_before_media_or_model_load(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from scripts import train_frozen_audio_probe as module

    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    _write_jsonl(sample_dir / "train.jsonl", [{"speaker_group_sha256": "speaker-a"}])
    _write_jsonl(sample_dir / "validation.jsonl", [{"speaker_group_sha256": "speaker-b"}])
    output = tmp_path / "output"
    monkeypatch.setattr(
        module,
        "_args",
        lambda: SimpleNamespace(
            sample_dir=sample_dir,
            output_dir=output,
            cohort_ledger=module.DEFAULT_LEDGER,
        ),
    )
    monkeypatch.setattr(
        module, "_validate_splits", lambda *args: pytest.fail("media validation ran")
    )
    monkeypatch.setattr(module, "_load_encoder", lambda *args: pytest.fail("model loaded"))

    with pytest.raises(CohortUseBlocked, match="BLOCKED_UNKNOWN_SOURCE_OR_REVISION"):
        module.main()
    assert not output.exists()


def test_audio_multiquery_blocks_unknown_history_before_hash_cache_or_model_load(
    tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from scripts import train_frozen_audio_multiquery as module

    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    _write_jsonl(
        sample_dir / "train.jsonl",
        [{"speaker_group_sha256": "speaker-a"}],
    )
    _write_jsonl(
        sample_dir / "validation.jsonl",
        [{"speaker_group_sha256": "speaker-b"}],
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        json.dumps(
            {
                "device": "cpu",
                "source": {
                    "sample_dir": str(sample_dir),
                    "feature_cache": str(tmp_path / "cache.pt"),
                },
                "encoder": {"text_model_path": str(tmp_path / "text-model")},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    monkeypatch.setattr(
        module,
        "_args",
        lambda: SimpleNamespace(
            config=config_path,
            output_dir=output,
            device=None,
            preflight_only=False,
            cohort_ledger=module.DEFAULT_LEDGER,
        ),
    )
    monkeypatch.setattr(module, "_hash", lambda *args: pytest.fail("artifact hashed"))
    monkeypatch.setattr(module, "_validate_splits", lambda *args: pytest.fail("media read"))
    monkeypatch.setattr(
        module, "_count_model_parameters", lambda *args: pytest.fail("model loaded")
    )
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("cache loaded"))

    with pytest.raises(CohortUseBlocked, match="BLOCKED_UNKNOWN_SOURCE_OR_REVISION"):
        module.main()
    assert not output.exists()


def test_image_training_blocks_unknown_history_before_media_or_model_load(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from scripts import train_frozen_image_probe as module

    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    _write_jsonl(sample_dir / "train-labels.jsonl", [{"id": "image-a"}])
    _write_jsonl(sample_dir / "val-labels.jsonl", [{"id": "image-b"}])
    output = tmp_path / "output"
    monkeypatch.setattr(
        module,
        "_args",
        lambda: SimpleNamespace(
            sample_dir=sample_dir, output_dir=output, cohort_ledger=module.DEFAULT_LEDGER
        ),
    )
    monkeypatch.setattr(module, "_load_vjepa", lambda *args: pytest.fail("model loaded"))

    with pytest.raises(CohortUseBlocked, match="BLOCKED_UNKNOWN_SOURCE_OR_REVISION"):
        module.main()
    assert not output.exists()


def test_video_training_blocks_unknown_history_before_media_or_model_load(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("av")
    from scripts import train_frozen_video_probe as module

    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    _write_jsonl(sample_dir / "train.jsonl", [{"scene_index": 1}])
    _write_jsonl(sample_dir / "validation.jsonl", [{"scene_index": 2}])
    output = tmp_path / "output"
    monkeypatch.setattr(
        module,
        "_args",
        lambda: SimpleNamespace(
            sample_dir=sample_dir,
            output_dir=output,
            cohort_ledger=module.DEFAULT_LEDGER,
        ),
    )
    monkeypatch.setattr(
        module, "_validate_splits", lambda *args: pytest.fail("media validation ran")
    )
    monkeypatch.setattr(module, "_load_vjepa", lambda *args: pytest.fail("model loaded"))

    with pytest.raises(CohortUseBlocked, match="BLOCKED_UNKNOWN_SOURCE_OR_REVISION"):
        module.main()
    assert not output.exists()


def test_text_training_blocks_unknown_history_before_tokenizer_or_model_load(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from scripts import train_frozen_text_probe as module

    train_manifest = tmp_path / "train.yaml"
    validation_manifest = tmp_path / "validation.yaml"
    manifest_base = {
        "schema_version": 1,
        "dataset_id": "example/text-source",
        "revision": "a" * 40,
        "split": "train",
        "modalities": ["text"],
        "upstream_url": "https://example.invalid/dataset",
        "license": "MIT",
        "usage": "training",
        "trust_status": "trusted",
    }
    train_manifest.write_text(json.dumps(manifest_base), encoding="utf-8")
    validation_manifest.write_text(
        json.dumps({**manifest_base, "split": "validation", "usage": "evaluation"}),
        encoding="utf-8",
    )
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    train_path.write_text("metadata only", encoding="utf-8")
    validation_path.write_text("metadata only", encoding="utf-8")
    output = tmp_path / "output"
    examples = {
        str(train_path): [
            SimpleNamespace(
                source_record_id="state-a:q1",
                state="s1",
                question="q1",
                options=["a", "b"],
            )
        ],
        str(validation_path): [
            SimpleNamespace(
                source_record_id="state-b:q1",
                state="s2",
                question="q2",
                options=["a", "b"],
            )
        ],
    }
    monkeypatch.setattr(
        module,
        "_args",
        lambda: SimpleNamespace(
            manifest=train_manifest,
            validation_manifest=validation_manifest,
            adapter="generic",
            train=train_path,
            validation=validation_path,
            preserve_noul_criteria=False,
            output_dir=output,
            cohort_ledger=module.DEFAULT_LEDGER,
            seed=17,
            model=tmp_path / "missing-model",
            revision="b" * 40,
            context_policy="legacy",
            option_max_length=256,
            feature_batch_size=1,
            batch_questions=1,
            epochs=1,
        ),
    )
    monkeypatch.setattr(
        module,
        "_load_examples",
        lambda path, *args, **kwargs: examples[str(path)],
    )
    monkeypatch.setattr(
        module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("tokenizer loaded"),
    )

    with pytest.raises(CohortUseBlocked, match="BLOCKED_UNKNOWN_SOURCE_OR_REVISION"):
        module.main()
    assert not output.exists()
