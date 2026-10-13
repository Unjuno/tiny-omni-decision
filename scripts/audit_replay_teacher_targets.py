import collections
import hashlib
import json
import math
from pathlib import Path

root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
order = json.loads((root / "context-replay-preflight-01/order.json").read_text())
raw = {
    r["state_id"]: r
    for r in map(json.loads, (root / "text/train.jsonl").read_text(encoding="utf-8").splitlines())
}
stats = collections.defaultdict(list)
rejected = []
for r in order:
    if not r["id"].startswith("n4ze3m/typed-decisions-synth:"):
        continue
    _, state, q = r["id"].split(":", 2)
    source = raw[state]
    spec = json.loads(source["questions"])[q]
    t = json.loads(source["teacher"])[q]
    try:
        if "probabilities" in t:
            d = t["probabilities"]
            if spec["type"] == "score":
                assert r["options"] == spec["criteria"]
                p = [float(d[str(i)]) for i in range(len(r["options"]))]
            else:
                p = [float(d[o.split(":", 1)[0]]) for o in r["options"]]
        else:
            assert "noul" in t
            assert {o.split(":", 1)[0] for o in r["options"]} == {"true", "false"}
            p = [
                float(t["noul"]) if o.split(":", 1)[0] == "true" else 1 - float(t["noul"])
                for o in r["options"]
            ]
        if abs(sum(p) - 1) > 0.001 or not all(0 <= x <= 1 for x in p):
            rejected.append({"id": r["id"], "reason": "invalid_probability_simplex", "values": p})
            continue
    except (KeyError, AssertionError, TypeError, ValueError):
        rejected.append(
            {
                "id": r["id"],
                "reason": "unmapped_teacher_schema",
                "teacher": t,
                "options": r["options"],
            }
        )
        continue
    target = r["options"].index(r["target"])
    stats[spec["type"]].append(
        {
            "sum_error": abs(sum(p) - 1),
            "max": max(p),
            "gold_probability": p[target],
            "entropy": -sum(x * math.log(x) for x in p if x),
            "disagree": p[target] < max(p) - 1e-8,
            "soft": sum(x > 1e-8 for x in p) > 1,
        }
    )
result = {
    "scope": "selected original training rows only; no validation/audit teacher probabilities",
    "source_sha256": hashlib.sha256((root / "text/train.jsonl").read_bytes()).hexdigest(),
    "order_sha256": hashlib.sha256(
        (root / "context-replay-preflight-01/order.json").read_bytes()
    ).hexdigest(),
    "types": {},
    "rejected": rejected,
}
for k, rows in stats.items():
    result["types"][k] = {
        "count": len(rows),
        "soft_count": sum(r["soft"] for r in rows),
        "argmax_disagreement": sum(r["disagree"] for r in rows),
        **{
            f"mean_{name}": sum(r[name] for r in rows) / len(rows)
            for name in ["max", "gold_probability", "entropy", "sum_error"]
        },
    }
Path("docs/pretrained_reuse/R2_REPLAY_TARGET_AUDIT_20261013.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps({"types": result["types"], "rejected_count": len(rejected)}, indent=2))
