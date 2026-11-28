from __future__ import annotations

import io
import json
import pickle
import struct
import zipfile

import pytest

from scripts.audit_physionpp_official_train_labels import (
    RestrictedUnpickler,
    ZipMember,
    archive_member_for_row,
    audit_record,
    decode_member,
    load_resume_results,
    parse_central_directory,
    parse_eocd,
    sha256_file,
)


def _row(path: str = "raw/physionpp-readout/readout_data_v1/scenario/scene/0003_img.mp4") -> dict:
    return {"media": [{"kind": "video", "path": path}]}


def test_archive_member_uses_complete_relative_identity() -> None:
    assert archive_member_for_row(_row()) == "readout_data_v1/scenario/scene/0003.pkl"


@pytest.mark.parametrize(
    "path",
    [
        "raw/physionpp-readout/../../outside/0003_img.mp4",
        "raw/other/scenario/scene/0003_img.mp4",
        "raw/physionpp-readout/scenario/scene/0003_seg.mp4",
        r"raw\physionpp-readout\scenario\scene\0003_img.mp4",
    ],
)
def test_archive_member_rejects_unsafe_or_wrong_media_paths(path: str) -> None:
    with pytest.raises(ValueError):
        archive_member_for_row(_row(path))


def test_zip_directory_and_member_crc_are_verified() -> None:
    payload = b"official metadata bytes" * 17
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("readout_data_v1/scenario/scene/0003.pkl", payload)
    raw_zip = buffer.getvalue()
    entries, cd_size, cd_offset, eocd_offset = parse_eocd(raw_zip[-65536:], len(raw_zip))
    assert entries == 1
    assert cd_offset + cd_size == eocd_offset
    central = parse_central_directory(raw_zip[cd_offset : cd_offset + cd_size], entries)
    member = central["readout_data_v1/scenario/scene/0003.pkl"]
    local = raw_zip[member.local_offset : cd_offset]
    decoded = decode_member(member, local)
    assert decoded == payload


def test_zip_member_rejects_corrupt_payload() -> None:
    member = ZipMember("x.pkl", 0, 0, 1, 1, 0)
    header = struct.pack("<4s5H3I2H", b"PK\x03\x04", 20, 0, 0, 0, 0, 0, 1, 1, 5, 0)
    header += b"x.pkl" + b"z"
    with pytest.raises(ValueError, match="CRC"):
        decode_member(member, header)


def test_restricted_unpickler_blocks_unlisted_globals() -> None:
    payload = b"\x80\x04cposix\nsystem\n."
    with pytest.raises(pickle.UnpicklingError, match="blocked pickle global"):
        RestrictedUnpickler(io.BytesIO(payload)).load()


def test_restricted_unpickler_accepts_plain_data_containers() -> None:
    payload = pickle.dumps({"static": {"label": True}, "frames": {"0000": {}}})
    loaded = RestrictedUnpickler(io.BytesIO(payload)).load()
    assert loaded["static"]["label"] is True


def test_restricted_unpickler_allows_only_explicit_builtin_string_global() -> None:
    assert RestrictedUnpickler(io.BytesIO(pickle.dumps(str))).load() is str


def test_audit_record_preserves_all_positive_contact_frame_indices(monkeypatch) -> None:
    member_name = "readout_data_v1/scenario/scene/0003.pkl"
    metadata = {
        "static": {
            "does_target_contact_zone": True,
            "first_target_contact_zone_frame": 2,
            "target_id": 6,
            "zone_id": 1,
            "stimulus_name": "scene_0003",
        },
        "frames": {
            str(index): {
                "labels": {"target_contacting_zone": index in {2, 4}}
            }
            for index in range(6)
        },
    }
    monkeypatch.setattr(
        "scripts.audit_physionpp_official_train_labels.fetch_member",
        lambda _member, _etag: pickle.dumps(metadata),
    )
    row = {
        "id": "sample-1",
        "target": "Yes",
        "source_target": True,
    }

    result = audit_record(row, ZipMember(member_name, 0, 0, 0, 0, 0), "etag")

    assert result["positive_contact_frame_count"] == 2
    assert result["positive_contact_frames"] == [2, 4]
    assert result["first_contact_frame"] == 2


def test_resume_results_must_match_train_id_path_and_labels(tmp_path) -> None:
    row = _row()
    row.update({"id": "sample-1", "target": "No", "source_target": False})
    result = {
        "sample_id": row["id"],
        "archive_member": archive_member_for_row(row),
        "manifest_target": "No",
        "source_target": False,
        "official_target": "No",
        "frame_count": 10,
        "positive_contact_frame_count": 0,
        "positive_contact_frames": [],
        "label_match": True,
        "frame_timeline_match": True,
        "stimulus_identity_match": True,
    }
    resume_path = tmp_path / "partial.jsonl"
    resume_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
    digest = sha256_file(resume_path)
    assert load_resume_results(resume_path, [row], digest) == {"sample-1": result}
    result["official_target"] = "Yes"
    resume_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not verify"):
        load_resume_results(resume_path, [row], sha256_file(resume_path))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_resume_results(resume_path, [row], digest)
