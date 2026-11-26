"""Post-hoc replay retention diagnostic using saved development predictions only."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
RUN = ROOT / "context-replay-context-run-01-all-encoder"


def summarize(rows):
    correct, nll, brier, confidence = [], [], [], []
    for r in rows:
        p = np.asarray(r["probabilities"], dtype=float)
        y = r["options"].index(r["target"])
        correct.append(int(p.argmax() == y))
        nll.append(-np.log(max(p[y], 1e-12)))
        brier.append(float((p * p).sum() - 2 * p[y] + 1))
        confidence.append(float(p.max()))
    c, f = np.asarray(correct), np.asarray(confidence)
    ece = sum(
        abs(float((f[m] - c[m]).sum())) / len(c)
        for i in range(15)
        if (m := ((f >= i / 15) & ((f < (i + 1) / 15) if i < 14 else (f <= 1)))).any()
    )
    return dict(
        count=len(c),
        accuracy=float(c.mean()),
        nll=float(np.mean(nll)),
        brier=float(np.mean(brier)),
        ece=ece,
        mean_confidence=float(f.mean()),
        wrong_confidence=float(f[c == 0].mean()) if (c == 0).any() else None,
    )


def main():
    source = ROOT / "text/validation.jsonl"
    meta = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        questions = json.loads(row["questions"])
        for q, spec in questions.items():
            key = f"n4ze3m/typed-decisions-synth:{row['state_id']}:{q}"
            meta[key] = dict(state=row["state_id"], type=spec["type"])
    all_rows = {}
    hashes = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
    for step in (0, 128, 256, 384):
        path = RUN / f"predictions-{step}.json"
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        all_rows[step] = [
            r for r in json.loads(path.read_text(encoding="utf-8")) if r["context"] == "original"
        ]
    ids = [r["id"] for r in all_rows[0]]
    assert len(ids) == len(set(ids)) == 2540
    assert all([r["id"] for r in rows] == ids for rows in all_rows.values())
    assert all(i in meta for i in ids)
    result = {
        "hashes": hashes,
        "scope": "post-hoc observed development; no calibration fitting",
        "steps": {},
    }
    for step, rows in all_rows.items():
        result["steps"][step] = {"all": summarize(rows)}
        for kind in sorted({meta[i]["type"] for i in ids}):
            result["steps"][step][kind] = summarize(
                [r for r in rows if meta[r["id"]]["type"] == kind]
            )
    a, b = all_rows[0], all_rows[384]
    transitions = {}
    for old, new in zip(a, b, strict=True):
        assert old["options"] == new["options"] and old["target"] == new["target"]
        y = old["options"].index(old["target"])
        before = int(np.argmax(old["probabilities"]) == y)
        after = int(np.argmax(new["probabilities"]) == y)
        key = f"{before}->{after}"
        transitions[key] = transitions.get(key, 0) + 1
    result["transitions"] = transitions
    # Resample complete states, preserving question dependence and sample-weighted estimand.
    groups = sorted({meta[i]["state"] for i in ids})
    stats = np.zeros((len(groups), 3))
    lookup = {g: j for j, g in enumerate(groups)}
    for old, new in zip(a, b, strict=True):
        y = old["options"].index(old["target"])
        delta_acc = int(np.argmax(new["probabilities"]) == y) - int(
            np.argmax(old["probabilities"]) == y
        )
        delta_nll = -np.log(max(new["probabilities"][y], 1e-12)) + np.log(
            max(old["probabilities"][y], 1e-12)
        )
        stats[lookup[meta[old["id"]]["state"]]] += [delta_acc, delta_nll, 1]
    rng = np.random.default_rng(17)
    draws = np.asarray(
        [stats[rng.integers(len(groups), size=len(groups))].sum(axis=0) for _ in range(10000)]
    )
    result["paired_state_bootstrap"] = dict(
        states=len(groups),
        repeats=10000,
        seed=17,
        method="percentile 95%, sample-weighted, unadjusted, observed validation",
        accuracy_delta_ci=np.quantile(draws[:, 0] / draws[:, 2], [0.025, 0.975]).tolist(),
        nll_delta_ci=np.quantile(draws[:, 1] / draws[:, 2], [0.025, 0.975]).tolist(),
    )
    output = Path("docs/pretrained_reuse/R2_CONTEXT_REPLAY_RETENTION_DIAGNOSTIC_20261013.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "hashes"}, indent=2))


if __name__ == "__main__":
    main()
