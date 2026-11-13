import collections
import csv
import hashlib
import html
import json
from pathlib import Path

p = Path("C:/CodexArtifacts/pretrained-reuse-r2/clotho-aqa-admission-01")
ledger = json.loads((p / "downloads.json").read_text())
for name, info in ledger.items():
    assert hashlib.sha256((p / name).read_bytes()).hexdigest() == info["sha256"]
meta = list(csv.DictReader((p / "clotho_aqa_metadata.csv").open(encoding="utf-8-sig")))
original_meta = meta
meta = {html.unescape(r["file_name"]): r for r in meta}
assert len(meta) == len(original_meta), "normalized filename collision"
rows = list(csv.DictReader((p / "clotho_aqa_train.csv").open(encoding="utf-8-sig")))
groups = collections.defaultdict(list)
for r in rows:
    groups[(r["file_name"], r["QuestionText"])].append(r["answer"].strip().lower())
fsd = json.loads(
    Path("C:/CodexArtifacts/pretrained-reuse-r2/fsd50k-training-128-02/selection.json").read_text()
)
ids = {r["id"] for r in fsd}
uploaders = {r["uploader"] for r in fsd}
allow = {
    "http://creativecommons.org/licenses/by/3.0/",
    "http://creativecommons.org/publicdomain/zero/1.0/",
}
counts = collections.Counter()
licenses = collections.Counter()
kept = []
for (name, q), answers in groups.items():
    if html.unescape(name) not in meta:
        counts["unresolved_filename_questions"] += 1
        continue
    m = meta[html.unescape(name)]
    licenses[m["license"]] += 1
    binary = set(answers) <= {"yes", "no"}
    counts["questions"] += 1
    counts["binary_questions"] += binary
    if not binary:
        continue
    counts["binary_annotation_count_" + str(len(answers))] += 1
    counts["binary_unanimous" if len(set(answers)) == 1 else "binary_disagreement"] += 1
    if m["license"] not in allow:
        counts["binary_disallowed_license"] += 1
        continue
    fid = m["sound_link"].rstrip("/").split("/")[-1]
    if fid in ids:
        counts["binary_existing_fsd_id"] += 1
        continue
    if m["manufacturer"] in uploaders:
        counts["binary_existing_fsd_uploader"] += 1
        continue
    if len(answers) != 3:
        counts["eligible_invalid_vote_count"] += 1
        continue
    votes = collections.Counter(answers)
    if votes["yes"] == votes["no"]:
        counts["binary_tie"] += 1
        continue
    target = int(votes["yes"] > votes["no"])
    counts["eligible_yes" if target else "eligible_no"] += 1
    kept.append(
        {
            "file_name": name,
            "source_id": fid,
            "uploader": m["manufacturer"],
            "question": q,
            "votes": answers,
            "target": target,
            "license": m["license"],
        }
    )
result = {
    "source": "https://zenodo.org/records/6473207",
    "downloads": ledger,
    "train_rows": len(rows),
    "train_clips": len({r["file_name"] for r in rows}),
    "counts": dict(counts),
    "question_license_counts": dict(licenses),
    "eligible_clips": len({r["file_name"] for r in kept}),
    "eligible_uploaders": len({r["uploader"] for r in kept}),
    "scope": "official train only; no validation/test questions or media loaded",
    "status": "metadata candidate only, no finalized split or media acquisition",
}
(p / "eligible-training-questions.json").write_text(json.dumps(kept, indent=2), encoding="utf-8")
(p / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result))
