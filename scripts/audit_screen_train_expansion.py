"""Next eight preassigned training apps: annotation coverage only."""

import json
from collections import Counter
from pathlib import Path

from scripts.audit_groundcua import REVISION, fetch
from scripts.range_generalization import sha, write

root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
selection = root / "groundcua-pilot-inputs-01/frozen-selection.json"
frozen = json.loads(selection.read_text())
assert frozen["revision"] == REVISION
inventory = root / "groundcua-source-audit-01/inventory.json"
assert sha(inventory) == frozen["inventory_sha256"]
paths = sorted(r["rfilename"] for r in json.loads(inventory.read_text())["siblings"])
apps = frozen["app_splits"]["train"][8:16]
chosen = []
for app in apps:
    candidates = [p for p in paths if p.startswith(f"data/{app}/") and p.endswith(".json")]
    assert len(candidates) >= 4
    chosen.extend({"app": app, "path": p} for p in candidates[:4])
assert len(chosen) == 32
out = root / "groundcua-train-expansion-annotations-01"
out.mkdir(exist_ok=False)
write(
    out / "selection.json",
    {
        "revision": REVISION,
        "source_selection_sha256": sha(selection),
        "rows": chosen,
        "policy": "next8 train apps, first4 lexical annotations; no replacements; no media/audit",
    },
)
records = []
for i, row in enumerate(chosen):
    raw = fetch(row["path"])
    dest = out / f"{i:03d}.json"
    dest.write_bytes(raw)
    annotations = json.loads(raw)
    counts = Counter(r.get("category", "") for r in annotations)
    records.append(
        {**row, "sha256": sha(dest), "regions": len(annotations), "categories": dict(counts)}
    )
    write(out / "progress.json", records)
    print(i + 1, flush=True)
write(
    out / "result.json",
    {
        "records": records,
        "selection_sha256": sha(out / "selection.json"),
        "regions": sum(r["regions"] for r in records),
        "blank": sum(r["categories"].get("", 0) for r in records),
        "media_acquired": False,
    },
)
