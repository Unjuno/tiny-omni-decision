"""Materialize every train annotation with an explicit eligibility reason."""

import argparse
import collections
import json
import zipfile
from pathlib import Path

from scripts.audit_perception_train import sha
from tiny_omni_decision.perception_visibility import visibility_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive = args.source / "train_annotations.zip"
    cuts_path = args.source / "cut_frame_mapping_train.json"
    pins = {
        archive: "9eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78",
        cuts_path: "cbea6c721fd150f861ec6fe8e7ff005792f754df1359b1364349b5abcaab6cc4",
    }
    if not all(sha(p) == h for p, h in pins.items()):
        raise ValueError("unreviewed source bytes")
    args.output.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        data = json.loads(z.read("all_train.json"))
    cuts = json.loads(cuts_path.read_text(encoding="utf-8"))
    counts, reasons, classes = collections.Counter(), collections.Counter(), collections.Counter()
    output = args.output / "records.jsonl"
    with output.open("w", encoding="utf-8") as stream:
        for video_id, row in sorted(data.items()):
            for record in visibility_records(video_id, row, cuts[video_id]):
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                counts[record["status"]] += 1
                reasons.update(record["hold_reasons"])
                classes[f"{record['status']}:{record['supervision']['target_option_id']}"] += 1
    expected = sum(len(r["sound_localisation"]) for r in data.values())
    if sum(counts.values()) != expected:
        raise ValueError("lost annotations")
    report = dict(
        total=expected,
        status_counts=counts,
        hold_reasons=reasons,
        target_counts=classes,
        records_sha256=sha(output),
        input_hashes={str(p): h for p, h in pins.items()},
        training_admitted=False,
        media_checked=False,
        scope="Oracle event boundaries and sound-class query; candidate task only.",
    )
    (args.output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
