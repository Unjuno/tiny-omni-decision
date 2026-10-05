from __future__ import annotations

import hashlib
import io
import wave
import zipfile
from pathlib import Path

from tiny_omni_decision import media
from tiny_omni_decision.schema import (
    DatasetManifest,
    DecisionExample,
    LicenseProvenance,
    MediaRef,
)


class _SizedBytesIO(io.BytesIO):
    def __init__(self, value: bytes) -> None:
        super().__init__(value)
        self.size = len(value)


def _clevr4_example(example_id: str, filename: str) -> DecisionExample:
    return DecisionExample(
        id=example_id,
        modality="image",
        state="A scene",
        question="Which option is correct?",
        options=["A", "B"],
        target="A",
        media=[
            MediaRef(
                kind="image",
                uri=f"source-ref://sgvaze/clevr4@{'a' * 40}/images/{filename}",
                license="CC-BY-4.0",
            )
        ],
        source="sgvaze/clevr4",
        source_revision="a" * 40,
        source_record_id=example_id,
        split="train",
        provenance=LicenseProvenance(
            license="CC-BY-4.0",
            commercial_use=True,
            derivative_model_training_allowed=True,
            redistribution_allowed=False,
            media_redistribution_allowed=True,
            trust_status="trusted",
        ),
    )


def _clevrer_example(example_id: str, filename: str) -> DecisionExample:
    return DecisionExample(
        id=example_id,
        modality="video",
        state="A scene",
        question="Which option is correct?",
        options=["A", "B"],
        target="A",
        media=[
            MediaRef(
                kind="video",
                uri=f"source-ref://CLEVRER/official-2020/videos/train/{filename}",
                license="CC-BY-4.0",
            )
        ],
        source="CLEVRER",
        source_revision="a" * 40,
        source_record_id=example_id,
        split="train",
        provenance=LicenseProvenance(
            license="CC-BY-4.0",
            commercial_use=True,
            derivative_model_training_allowed=True,
            redistribution_allowed=False,
            media_redistribution_allowed=True,
            trust_status="trusted",
        ),
    )


def _speech_commands_example(split: str, filename: str) -> DecisionExample:
    return DecisionExample(
        id=f"speech-{split}-{filename}",
        modality="audio",
        state="A one-second keyword recording",
        question="Which spoken English keyword was uttered?",
        options=["yes", "no"],
        target="yes",
        media=[
            MediaRef(
                kind="audio",
                uri=f"source-ref://google/speech_commands/v0.02/{split}/yes/{filename}",
                license="CC-BY-4.0",
            )
        ],
        source="google/speech_commands",
        source_revision="a" * 40,
        source_record_id=f"yes/{filename}",
        split=split,
        provenance=LicenseProvenance(
            license="CC-BY-4.0",
            commercial_use=True,
            derivative_model_training_allowed=True,
            redistribution_allowed=True,
            media_redistribution_allowed=True,
            trust_status="trusted",
        ),
    )


def _speech_commands_manifest(split: str, parquet_file: str) -> DatasetManifest:
    return DatasetManifest(
        dataset_id="google/speech_commands",
        revision="a" * 40,
        subset="v0.02",
        split=split,
        modalities=["audio"],
        upstream_url="https://huggingface.co/datasets/google/speech_commands",
        license="CC-BY-4.0",
        usage="evaluation",
        trust_status="trusted",
        notes={"parquet_files": [parquet_file]},
    )


def _wav_payload() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(16_000)
        audio_file.writeframes(b"\x00\x00" * 160)
    return buffer.getvalue()


def test_clevr4_media_are_extracted_in_archive_order(
    tmp_path: Path, monkeypatch
) -> None:
    archive_data = io.BytesIO()
    png_header = b"\x89PNG\r\n\x1a\n"
    with zipfile.ZipFile(archive_data, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("images/early.png", png_header + b"early")
        archive.writestr("images/late.png", png_header + b"late")
    payload = archive_data.getvalue()

    opened: list[str] = []
    original_open = zipfile.ZipFile.open

    def record_open(self, name, *args, **kwargs):
        opened.append(name.filename if isinstance(name, zipfile.ZipInfo) else name)
        return original_open(self, name, *args, **kwargs)

    monkeypatch.setattr(media, "_HttpRangeReader", lambda _url: _SizedBytesIO(payload))
    monkeypatch.setattr(zipfile.ZipFile, "open", record_open)
    examples = [
        _clevr4_example("late", "late.png"),
        _clevr4_example("early", "early.png"),
    ]

    media.materialize_clevr4_images(
        examples,
        data_root=tmp_path,
        archive_url="https://example.invalid/clevr4.zip",
    )

    assert opened == ["images/early.png", "images/late.png"]


def test_clevr4_media_can_be_materialized_from_verified_local_archive(
    tmp_path: Path, monkeypatch
) -> None:
    archive_path = tmp_path / "clevr4.zip"
    png_header = b"\x89PNG\r\n\x1a\n"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("images/requested.png", png_header + b"requested")
        archive.writestr("images/unrequested-reserve.png", png_header + b"reserve")

    def unexpected_http_reader(_url: str):
        raise AssertionError("verified local materialization must not read remote ranges")

    monkeypatch.setattr(media, "_HttpRangeReader", unexpected_http_reader)
    examples = [_clevr4_example("requested", "requested.png")]
    converted, metadata = media.materialize_clevr4_images(
        examples,
        data_root=tmp_path,
        archive_url="https://example.invalid/clevr4.zip",
        archive_path=archive_path,
    )

    assert (tmp_path / converted[0].media[0].path).is_file()
    assert metadata["images_materialized"] == 1
    assert metadata["images_newly_downloaded"] == 1
    assert "full archive hash verified" in str(metadata["archive_revision_or_checksum"])
    assert not (tmp_path / "raw/clevr4-10k/images/unrequested-reserve.png").exists()


def test_clevrer_media_are_extracted_in_archive_order(
    tmp_path: Path, monkeypatch
) -> None:
    archive_data = io.BytesIO()
    mp4_header = b"\x00\x00\x00\x18ftypisom"
    with zipfile.ZipFile(archive_data, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("train/videos/early.mp4", mp4_header + b"early")
        archive.writestr("train/videos/late.mp4", mp4_header + b"late")
    payload = archive_data.getvalue()

    opened: list[str] = []
    original_open = zipfile.ZipFile.open

    def record_open(self, name, *args, **kwargs):
        opened.append(name.filename if isinstance(name, zipfile.ZipInfo) else name)
        return original_open(self, name, *args, **kwargs)

    monkeypatch.setattr(media, "_HttpRangeReader", lambda _url: _SizedBytesIO(payload))
    monkeypatch.setattr(zipfile.ZipFile, "open", record_open)
    examples = [
        _clevrer_example("late", "late.mp4"),
        _clevrer_example("early", "early.mp4"),
    ]

    media.materialize_clevrer_videos(
        examples,
        data_root=tmp_path,
        archive_urls={"train": "https://example.invalid/clevrer.zip"},
    )

    assert opened == ["train/videos/early.mp4", "train/videos/late.mp4"]


def test_speech_commands_materialization_uses_manifest_for_each_split(
    tmp_path: Path,
) -> None:
    examples = [
        _speech_commands_example("validation", "validation.wav"),
        _speech_commands_example("test", "test.wav"),
    ]
    payloads = {example.split: _wav_payload() for example in examples}
    for example in examples:
        path = tmp_path / "raw" / "speech-commands" / example.split / example.source_record_id
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payloads[example.split])

    manifests = {
        "validation": _speech_commands_manifest(
            "validation", "v0.02/validation-00000-of-00001.parquet"
        ),
        "test": _speech_commands_manifest("test", "v0.02/test-00000-of-00001.parquet"),
    }
    materialized, metadata = media.materialize_speech_commands_audio(
        examples, manifest=manifests, data_root=tmp_path
    )

    assert [item.media[0].path for item in materialized] == [
        "raw/speech-commands/validation/yes/validation.wav",
        "raw/speech-commands/test/yes/test.wav",
    ]
    assert [item.media[0].sha256 for item in materialized] == [
        hashlib.sha256(payloads["validation"]).hexdigest(),
        hashlib.sha256(payloads["test"]).hexdigest(),
    ]
    assert metadata["source_manifests_by_split"] == {
        "validation": {
            "source_dataset": "google/speech_commands",
            "source_revision": "a" * 40,
            "parquet_files": ["v0.02/validation-00000-of-00001.parquet"],
        },
        "test": {
            "source_dataset": "google/speech_commands",
            "source_revision": "a" * 40,
            "parquet_files": ["v0.02/test-00000-of-00001.parquet"],
        },
    }
    assert metadata["audio_materialized"] == 2


def test_speech_commands_downloads_each_split_from_its_pinned_shard(
    tmp_path: Path, monkeypatch
) -> None:
    examples = [
        _speech_commands_example("validation", "validation.wav"),
        _speech_commands_example("test", "test.wav"),
    ]
    manifests = {
        "validation": _speech_commands_manifest(
            "validation", "v0.02/validation-00000-of-00001.parquet"
        ),
        "test": _speech_commands_manifest("test", "v0.02/test-00000-of-00001.parquet"),
    }
    payloads = {split: _wav_payload() for split in manifests}
    streamed_splits: list[str] = []

    def fake_iter_hub_rows(source_manifest: DatasetManifest):
        split = source_manifest.split
        streamed_splits.append(split)
        filename = f"{split}.wav"
        return iter(
            [
                {
                    "label": "yes",
                    "audio": {"path": f"yes/{filename}", "bytes": payloads[split]},
                }
            ]
        )

    monkeypatch.setattr(media, "iter_hub_rows", fake_iter_hub_rows)
    materialized, metadata = media.materialize_speech_commands_audio(
        examples, manifest=manifests, data_root=tmp_path
    )

    assert set(streamed_splits) == {"validation", "test"}
    assert metadata["audio_newly_downloaded"] == 2
    for example in materialized:
        assert example.media[0].sha256 == hashlib.sha256(payloads[example.split]).hexdigest()
        path = tmp_path / example.media[0].path
        assert path.read_bytes() == payloads[example.split]
