"""Validate a complete acquisition against a separately frozen selection."""

import hashlib
import json

SELECTION_SHA = "ca017b24cbe16975c984335355281c6e4f0e2614eca23a27c49d723fddff61d6"


def complete_acquisition(media, expected_sha=SELECTION_SHA, count=128):
    data = (media / "selection.json").read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError("frozen selection hash mismatch")
    selected = json.loads(data)
    report = json.loads((media / "result.json").read_text())
    if report["status"] != "PASS" or report["selection"] != selected:
        raise ValueError("complete matching terminal report required")
    ids = [row["id"] for row in selected]
    if len(ids) != count or len(set(ids)) != count:
        raise ValueError("selection count/uniqueness mismatch")
    if any(not isinstance(x, str) or not x.isdecimal() for x in ids):
        raise ValueError("invalid clip ID")
    if [row["id"] for row in report["files"]] != ids:
        raise ValueError("media order/identity mismatch")
    for row in report["files"]:
        path = media / f"{row['id']}.wav"
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("media hash mismatch")
    return report
