from __future__ import annotations

import io
import zipfile

import pytest

from scripts.audit_tartanaviation_availability import (
    adsb_day_groups,
    audio_day_groups,
    normalized_listing_hash,
    parse_zip_central_names,
)


def test_audio_day_groups_normalize_airport_and_full_date() -> None:
    rows = [
        {"name": "kagc/2022/1/01-02-22_audio.zip", "bytes": 100},
        {"name": "kbtp/2022/1/01-02-22_audio.zip", "bytes": 200},
        {"name": "kagc/2022/1/not-a-date_audio.zip", "bytes": 300},
    ]

    assert audio_day_groups(rows) == {"kagc/2022-01-02", "kbtp/2022-01-02"}


def test_zip_central_directory_reads_names_without_member_payload() -> None:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("01-02-22/1.csv", b"private trajectory payload")
        zf.writestr("01-02-22/2.csv", b"another payload")
    raw = archive.getvalue()

    assert parse_zip_central_names(raw, object_size=len(raw)) == [
        "01-02-22/1.csv",
        "01-02-22/2.csv",
    ]


def test_zip_central_directory_rejects_truncated_tail() -> None:
    with pytest.raises(ValueError, match="end-of-central-directory"):
        parse_zip_central_names(b"not a zip", object_size=9)


def test_adsb_groups_intersect_using_full_airport_date_identity(monkeypatch) -> None:
    observed: list[str] = []

    def fake_fetch(row: dict[str, object]) -> tuple[list[str], int]:
        observed.append(str(row["name"]))
        if str(row["name"]).endswith("2021.zip"):
            return ["12-31-21/1.csv"], 8
        return ["01-01-22/1.csv"], 8

    monkeypatch.setattr(
        "scripts.audit_tartanaviation_availability.fetch_zip_central_names",
        fake_fetch,
    )
    groups, archives, byte_count = adsb_day_groups(
        [
            {"name": "kagc/raw/2021.zip", "bytes": 100},
            {"name": "kagc/raw/2022.zip", "bytes": 200},
            {"name": "kbtp/processed.zip", "bytes": 300},
        ]
    )

    assert groups == {"kagc/2021-12-31", "kagc/2022-01-01"}
    assert len(archives) == 2
    assert byte_count == 16
    assert len(observed) == 2


def test_normalized_listing_hash_is_order_independent() -> None:
    rows = [
        {"name": "b", "bytes": 2},
        {"name": "a", "bytes": 1},
    ]
    assert normalized_listing_hash(rows) == normalized_listing_hash(list(reversed(rows)))
