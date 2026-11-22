from __future__ import annotations

import pytest

from scripts.pretrained_reuse.screen_noaa_radar_asos_catalog import (
    parse_asos_index,
    parse_mrms_listing,
    run_screen,
)


def test_parse_asos_index_counts_unique_station_month_files() -> None:
    html = """<table>
    <tr><td><a href="asos-1min-pg2-KAAA-202405.dat">file</a></td>
    <td align="right">1024</td></tr>
    <tr><td><a href="asos-1min-pg2-KBBB-202405.dat">file</a></td>
    <td align="right">2048</td></tr>
    </table>"""

    rows = parse_asos_index(html, year=2024, month=5)

    assert [row["station_id"] for row in rows] == ["KAAA", "KBBB"]
    assert [row["listed_bytes"] for row in rows] == [1024, 2048]


def test_parse_asos_index_rejects_duplicate_station_files() -> None:
    html = """<table>
    <tr><td><a href="asos-1min-pg2-KAAA-202405.dat">file</a></td>
    <td align="right">1024</td></tr>
    <tr><td><a href="asos-1min-pg2-KAAA-202405.dat">file</a></td>
    <td align="right">1024</td></tr>
    </table>"""

    with pytest.raises(ValueError, match="duplicate station-month"):
        parse_asos_index(html, year=2024, month=5)


def test_parse_mrms_listing_sums_only_complete_unique_timestamp_list() -> None:
    xml = b"""<?xml version="1.0"?>
    <ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
      <IsTruncated>false</IsTruncated>
      <Contents><Key>CONUS/RadarOnly_QPE_01H_00.00/20240501/MRMS_RadarOnly_QPE_01H_00.00_20240501-000000.grib2.gz</Key><Size>10</Size></Contents>
      <Contents><Key>CONUS/RadarOnly_QPE_01H_00.00/20240501/MRMS_RadarOnly_QPE_01H_00.00_20240501-000200.grib2.gz</Key><Size>20</Size></Contents>
    </ListBucketResult>"""

    result = parse_mrms_listing(xml, date="2024-05-01")

    assert result["object_count"] == 2
    assert result["unique_timestamp_count"] == 2
    assert result["listed_compressed_payload_bytes"] == 30


def test_parse_mrms_listing_fails_closed_on_truncation() -> None:
    xml = b"""<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
    <IsTruncated>true</IsTruncated>
    <Contents><Key>anything</Key><Size>10</Size></Contents>
    </ListBucketResult>"""

    with pytest.raises(ValueError, match="truncated"):
        parse_mrms_listing(xml, date="2024-05-01")


def test_screen_refuses_to_overwrite_existing_output(tmp_path) -> None:
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_screen(output=output)


def test_screen_rejects_impossible_calendar_date(tmp_path) -> None:
    with pytest.raises(ValueError, match="invalid catalog date"):
        run_screen(output=tmp_path / "new", year=2024, month=2, day=31)
