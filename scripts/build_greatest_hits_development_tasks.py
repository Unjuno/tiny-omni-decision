"""Materialize every supported native AV task only after complete fixed-media integrity checks."""

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from scripts.audit_greatest_hits_development_media import ANNOTATIONS_SHA
from scripts.fetch_greatest_hits_development_media import PARTITION_SHA
from scripts.fetch_greatest_hits_train_inventory import write
from scripts.greatest_hits_tasks import TASKS, prior_metrics, tasks_for_parent
from scripts.probe_greatest_hits_train import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inventory", "media", "audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo).strip():
        raise ValueError("clean committed source required")
    if (sha(args.inventory / "annotations.json") != ANNOTATIONS_SHA
            or sha(args.inventory / "partition.json") != PARTITION_SHA):
        raise ValueError("source/partition mismatch")
    plan = json.loads((args.inventory / "partition.json").read_text())
    annotations = json.loads((args.inventory / "annotations.json").read_text())
    members = json.loads((args.media / "members.json").read_text())
    audit = json.loads((args.audit / "result.json").read_text())
    if (audit["status"] != "PASS_COMPLETE_DECODE" or audit["invalid_windows"]
            or audit["media_receipt_sha256"] != sha(args.media / "result.json")):
        raise ValueError("complete window/content integrity required; no automatic filtering")
    args.output.mkdir(exist_ok=False)
    result = dict(status="RUNNING", command=[sys.executable, *sys.argv],
                  source_commit=subprocess.check_output(
                      ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                  media_root=str(args.media / "members"), source_license="CC-BY-4.0",
                  scope="development native material/action/reaction; no blind product claim",
                  physical_av_offset="UNKNOWN", actor_site_identity="UNKNOWN across dates",
                  annotations_sha256=ANNOTATIONS_SHA, partition_sha256=PARTITION_SHA,
                  audit_receipt_sha256=sha(args.audit / "result.json"),
                  splits={}, artifacts={}, optimizer_updates=0, encoder_loaded=False,
                  test_media_or_labels_opened=False, sealed_audit_opened=False)
    try:
        counts = {}
        for split in ("train", "validation"):
            tasks, missing, event_ids = [], Counter(), set()
            for ident in plan[split]:
                item = members[ident]
                decoded = json.loads((args.audit / (ident + ".json")).read_text())
                if (item["split"] != split or decoded["split"] != split
                        or decoded["windows"]["rows"] != len(annotations[ident])):
                    raise ValueError("parent ownership/accounting mismatch")
                rows, absent = tasks_for_parent(ident, split, item["sha256"], annotations[ident],
                                               set(decoded["windows"]["out_of_bounds_windows"]))
                tasks.extend(rows)
                missing.update(absent)
                event_ids.update(r["event_id"] for r in rows)
            if len({r["id"] for r in tasks}) != len(tasks):
                raise ValueError("duplicate question ID")
            counts[split] = {kind: [sum(r["task"] == kind and r["target"] == i for r in tasks)
                                   for i in range(len(options))]
                             for kind, (_, options) in TASKS.items()}
            path = args.output / (split + ".jsonl")
            with path.open("x", encoding="utf-8") as stream:
                for row in tasks:
                    stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            result["splits"][split] = dict(parents=len(plan[split]), questions=len(tasks),
                                            events_with_known_fields=len(event_ids),
                                            missing_fields=dict(missing), labels=counts[split])
            result["artifacts"][path.name] = sha(path)
        result["question_only_train_prior"] = {kind: prior_metrics(
            counts["train"][kind], counts["validation"][kind]) for kind in TASKS}
        result["status"] = "PASS_FIXED_DEVELOPMENT_TASKS"
    except Exception as exc:
        result.update(status="FAILED", error=repr(exc))
        raise
    finally:
        result["source_hashes"] = {p: sha(repo / p) for p in [
            "scripts/greatest_hits_tasks.py", "scripts/build_greatest_hits_development_tasks.py"]}
        write(args.output / "result.json", result)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
