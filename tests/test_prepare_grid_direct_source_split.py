import io
import tarfile

import pytest

from scripts.prepare_grid_direct_source_split import (
    _read_alignment_codes,
    _validate_bundle_index,
    build_split_rows,
)


def test_build_split_rows_excludes_observed_content_and_keeps_source_ids_separate():
    speaker_codes = {
        "s2": {"bbaa1a", "bbab1p"},
        "s3": {"bbaa2a"},
        "s4": {"bbaa3a"},
    }

    rows, counts = build_split_rows(
        speaker_codes,
        {"train": ["s4", "s2"], "validation": ["s3"]},
        forbidden_codes={"bbab1p"},
    )

    assert rows["train"] == [
        {
            "speaker": "s2",
            "code": "bbaa1a",
            "targets": {
                "command": "bin",
                "color": "blue",
                "preposition": "at",
                "letter": "a",
                "number": "1",
                "adverb": "again",
            },
        },
        {
            "speaker": "s4",
            "code": "bbaa3a",
            "targets": {
                "command": "bin",
                "color": "blue",
                "preposition": "at",
                "letter": "a",
                "number": "3",
                "adverb": "again",
            },
        },
    ]
    assert [row["code"] for row in rows["validation"]] == ["bbaa2a"]
    assert counts == {"train": 1, "validation": 0}


def test_build_split_rows_rejects_duplicate_source_codes_before_exclusion():
    speaker_codes = {"s2": {"bbaa1a"}, "s3": {"bbaa1a"}}

    with pytest.raises(ValueError, match="duplicate source code"):
        build_split_rows(
            speaker_codes,
            {"train": ["s2"], "validation": ["s3"]},
            forbidden_codes={"bbaa1a"},
        )


def test_build_split_rows_rejects_speaker_overlap_between_partitions():
    with pytest.raises(ValueError, match="speaker overlap"):
        build_split_rows(
            {"s2": {"bbaa1a"}},
            {"train": ["s2"], "validation": ["s2"]},
            forbidden_codes=set(),
        )


def test_alignment_reader_rejects_words_that_disagree_with_the_source_code(tmp_path):
    path = tmp_path / "s2-align.tar"
    payload = b"0 0.1 bin\n0.1 0.2 blue\n0.2 0.3 at\n0.3 0.4 a\n0.4 0.5 two\n0.5 0.6 again\n"
    info = tarfile.TarInfo("bbaa1a.align")
    info.size = len(payload)
    with tarfile.open(path, "w") as archive:
        archive.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValueError, match="alignment and code disagree"):
        _read_alignment_codes(path)


def test_alignment_reader_accepts_a_word_sequence_matching_the_code(tmp_path):
    path = tmp_path / "s2-align.tar"
    payload = b"0 0.1 bin\n0.1 0.2 blue\n0.2 0.3 at\n0.3 0.4 a\n0.4 0.5 one\n0.5 0.6 again\n"
    info = tarfile.TarInfo("bbaa1a.align")
    info.size = len(payload)
    with tarfile.open(path, "w") as archive:
        archive.addfile(info, io.BytesIO(payload))

    assert _read_alignment_codes(path) == ({"bbaa1a"}, 1)


def test_bundle_index_requires_exact_speakers_and_metadata_revision():
    index = {
        "source_record": "doi:10.5281/zenodo.3625687",
        "zenodo_record_sha256": "pinned-hash",
        "speakers": {"s2": {"audio_tar": "a.tar"}},
    }
    record = {"files": [{"key": "s2.zip"}]}

    assert _validate_bundle_index(
        index, record, ["s2"], zenodo_sha256="pinned-hash"
    ) == {"s2.zip": {"key": "s2.zip"}}
    with pytest.raises(ValueError, match="different Zenodo metadata"):
        _validate_bundle_index(index, record, ["s2"], zenodo_sha256="other-hash")
