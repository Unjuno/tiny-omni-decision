import json
import zipfile

from scripts.audit_lsc50_timestamp_pairing import audit_lsc50_pairing


def _write_xlsx(path, sheets):
    workbook_sheets = []
    relationships = []
    for index, (name, _records) in enumerate(sheets, 1):
        workbook_sheets.append(
            f'<sheet name="{name}" sheetId="{index}" r:id="rId{index}"/>'
        )
        relationships.append(
            f'<Relationship Id="rId{index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{index}.xml"/>'
        )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "xl/workbook.xml",
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f'<sheets>{"".join(workbook_sheets)}</sheets></workbook>',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'{"".join(relationships)}</Relationships>',
        )
        for index, (_, records) in enumerate(sheets, 1):
            rows = [["N", "Sign", "Time Start (s)", "Time End (s)"]] + records
            serialized = []
            for row_number, row in enumerate(rows, 1):
                cells = []
                for column, value in zip("ABCD", row, strict=True):
                    cell_ref = f"{column}{row_number}"
                    if isinstance(value, str):
                        cells.append(
                            f'<c r="{cell_ref}" t="inlineStr"><is><t>{value}</t></is></c>'
                        )
                    else:
                        cells.append(f'<c r="{cell_ref}"><v>{value}</v></c>')
                serialized.append(f'<row r="{row_number}">{"".join(cells)}</row>')
            archive.writestr(
                f"xl/worksheets/sheet{index}.xml",
                '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                f'<sheetData>{"".join(serialized)}</sheetData></worksheet>',
            )


def _write_inventory(path):
    entries = []
    for view in ("COLOR_BODY", "COLOR_FACE"):
        for name in ("0000_0000_0000.avi", "0000_0000_0001.avi"):
            entries.append({"name": f"VIDEOS/{view}/{name}"})
    entries.append({"name": "VIDEOS/COLOR_BODY/"})
    path.write_text(json.dumps(entries), encoding="utf-8")


def test_audit_reports_timestamp_anomaly_and_unresolved_video_repetition_mapping(tmp_path):
    workbook = tmp_path / "timestamps.xlsx"
    inventory = tmp_path / "video-inventory.json"
    _write_xlsx(
        workbook,
        [
            ("Volunteer 1", [[0, "SIGN_A", 1.0, 2.0], [1, "SIGN_B", 2.0, 2.0]]),
            ("Volunteer 2", [[0, "SIGN_A", 3.0, 4.0], [1, "SIGN_B", 4.0, 5.0]]),
        ],
    )
    _write_inventory(inventory)

    result = audit_lsc50_pairing(workbook, inventory)

    assert result["timestamp_sheet_count"] == 2
    assert result["timestamp_row_count"] == 4
    assert result["timestamp_invalid_interval_count"] == 1
    assert result["timestamp_sheets"][0]["invalid_interval_count"] == 1
    assert result["video_avi_count"] == 4
    assert result["video_unique_sign_volunteer_repetition_triplets"] == 2
    assert result["video_unique_sign_volunteer_pairs"] == 1
    assert result["video_repetition_count_histogram"] == {"2": 1}
    assert result["video_participant_id_values"] == ["0000"]
    assert result["video_repetition_to_timestamp_mapping"] == "unresolved from metadata inspected"
    assert result["media_payload_opened"] is False
    assert result["sensor_signal_opened"] is False


def test_audit_rejects_unrecognized_video_member_name(tmp_path):
    import pytest

    workbook = tmp_path / "timestamps.xlsx"
    inventory = tmp_path / "video-inventory.json"
    _write_xlsx(workbook, [("Volunteer 1", [[0, "SIGN_A", 1.0, 2.0]])])
    inventory.write_text(json.dumps([{"name": "VIDEOS/BAD/not-a-triplet.avi"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="unrecognized video member name"):
        audit_lsc50_pairing(workbook, inventory)
