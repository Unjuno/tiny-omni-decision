"""Read synchronization workbook structure only; do not read sensor/media contents."""

import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from scripts.audit_mmfit import RemoteZip
from scripts.range_generalization import sha, write


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    source = root / "lsc50-source-audit-02"
    output = root / "lsc50-timestamp-structure-01"
    output.mkdir(exist_ok=False)
    metadata = json.loads((source / "metadata.json").read_text())
    file = next(f for f in metadata["files"] if f["name"] == "IMU.zip")
    headers = json.loads((source / "IMU.zip.headers.json").read_text())
    remote = RemoteZip(file["download_url"], file["size"], headers["ETag"])
    with zipfile.ZipFile(remote) as archive:
        payload = archive.read("IMU/INFO/Timestamps.xlsx")
    path = output / "Timestamps.xlsx"
    path.write_bytes(payload)
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    structure = []
    with zipfile.ZipFile(path) as xlsx:
        shared = ET.fromstring(xlsx.read("xl/sharedStrings.xml"))
        strings = ["".join(si.itertext()) for si in shared]
        workbook = ET.fromstring(xlsx.read("xl/workbook.xml"))
        sheets = workbook.find("m:sheets", ns)
        for i, sheet in enumerate(sheets, 1):
            xml = ET.fromstring(xlsx.read(f"xl/worksheets/sheet{i}.xml"))
            row = xml.find("m:sheetData/m:row", ns)
            header = []
            for cell in row:
                value = cell.find("m:v", ns)
                value = value.text if value is not None else None
                header.append(strings[int(value)] if cell.get("t") == "s" else value)
            structure.append({"sheet": sheet.get("name"),
                              "dimension": xml.find("m:dimension", ns).get("ref"),
                              "header": header})
    result = {"member": "IMU/INFO/Timestamps.xlsx", "sha256": sha(path),
              "sheets": structure, "sensor_or_video_read": False,
              "timing_rows_inspected": False}
    write(output / "result.json", result)
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
