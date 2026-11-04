"""Predeclared question-side wording/alias2x2 on immutable development states."""

import argparse
import copy
import json
from pathlib import Path

PREFIX = "Use cost as price, risk as danger, and time as timestamp. "
CELLS = ("generic", "generic_alias", "domain", "domain_alias")
VARIANTS = tuple(f"{regime}-{cell}" for regime in ("base", "expanded") for cell in CELLS)


def make_cells(generic, domain):
    for key in ("state", "options", "target", "oracle", "task", "group", "base_id"):
        if generic[key] != domain[key]:
            raise ValueError("mismatched paired input")
    regime = generic["variant"].split("-")[0]
    if generic["variant"] != regime + "-0" or domain["variant"] != regime + "-4":
        raise ValueError("incorrect paired families")
    changes = generic["task"] in ("extremum", "temporal_order")
    if changes and not domain["question"].startswith(PREFIX):
        raise ValueError("prefix changed")
    result = []
    for cell in CELLS:
        r = copy.deepcopy(generic)
        if changes:
            r["question"] = {
                "generic": generic["question"],
                "generic_alias": PREFIX + generic["question"],
                "domain": domain["question"][len(PREFIX) :],
                "domain_alias": domain["question"],
            }[cell]
        elif domain["question"] != generic["question"]:
            raise ValueError("retention task changed")
        r.update(variant=regime + "-" + cell, id=regime + "-" + cell + ":" + r["base_id"])
        result.append(r)
    return result


def main():
    from scripts.build_conditional_curriculum import verify
    from scripts.compare_numeric_embedding_predictions import compare
    from scripts.evaluate_vocabulary_transfer import opposing_pairs
    from scripts.range_generalization import evaluate, sha, write

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "evaluate"])
    args = parser.parse_args()
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    data = root / "transfer-factorial-inputs-01"
    out = root / "transfer-factorial-eval-01"
    source = root / "vocabulary-transfer-inputs-01/inputs.jsonl"
    source_sha = "1a825b413e32f319dead47ccbef07ef3b36f2c391ca2c197a694aaeb4ed54dd2"
    if sha(source) != source_sha:
        raise ValueError("source changed")

    def read(p):
        return json.loads(p.read_text(encoding="utf-8"))

    if args.mode == "prepare":
        old = [json.loads(x) for x in source.read_text(encoding="utf-8").splitlines()]
        lookup = {(r["variant"], r["base_id"]): r for r in old}
        rows = []
        for r in old:
            regime, family = r["variant"].split("-")
            if family == "0":
                rows.extend(make_cells(r, lookup[(regime + "-4", r["base_id"])]))
        rows = sorted(rows, key=lambda r: VARIANTS.index(r["variant"]))
        if len(rows) != 8192 or len({r["id"] for r in rows}) != 8192:
            raise ValueError("count")
        for row in rows:
            verify(row)
        data.mkdir(exist_ok=False)
        p = data / "inputs.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        write(
            data / "manifest.json",
            dict(
                input_sha256=sha(p),
                source_sha256=source_sha,
                runner_sha256=sha(Path(__file__)),
                rows=len(rows),
                variants=VARIANTS,
                states=len({r["group"] for r in rows}),
                target_changes=0,
                training=False,
            ),
        )
        print(json.dumps(read(data / "manifest.json")))
        return
    manifest = read(data / "manifest.json")
    expected = "6fcead8721c40534fa994981a63403112df057b8f19daee11016836b7ab6878e"
    if manifest["input_sha256"] != expected:
        raise ValueError("fixed input hash changed")
    if sha(data / "inputs.jsonl") != manifest["input_sha256"]:
        raise ValueError("prepared inputs changed")
    rows = [json.loads(x) for x in (data / "inputs.jsonl").read_text(encoding="utf-8").splitlines()]
    for row in rows:
        verify(row)
    checkpoints = {
        "parent": (
            "numeric-deterministic-control-01",
            "0b785581bc5db5db01a9b39494af41b32d26fc6958d2c6191ccd74edcf016882",
        ),
        "candidate": (
            "text-coverage-training-01",
            "b063f124e59c8a4c1b09c7bf2a8423846f73489d7ce8cd6649735745bc6cebf5",
        ),
    }
    evaluate(
        root,
        Path("C:/CodexArtifacts/pretrained-reuse-r1/crossencoder-minilm-l6-audit-01/model"),
        data,
        manifest["input_sha256"],
        out,
        checkpoints,
        VARIANTS,
        8192,
    )
    predictions = {}
    pairs = {}
    parity = {}
    effects = {}
    for name in checkpoints:
        predictions[name] = {v: read(out / f"{name}-{v}-predictions.json") for v in VARIANTS}
        pairs[name] = {}
        parity[name] = {}
        effects[name] = {}
        for regime in ("base", "expanded"):
            for cell, family in [("generic", "0"), ("domain_alias", "4")]:
                old = read(
                    root
                    / "coverage-transfer-eval-01"
                    / f"{name}-{regime}-{family}-predictions.json"
                )
                fresh = predictions[name][regime + "-" + cell]
                maxdelta = 0.0
                for a, b in zip(old, fresh, strict=True):
                    if a["id"].split(":", 1)[1] != b["id"].split(":", 1)[1] or any(
                        a[k] != b[k] for k in ("options", "target")
                    ):
                        raise ValueError("reference identity mismatch")
                    maxdelta = max(
                        maxdelta,
                        max(
                            abs(x - y)
                            for x, y in zip(a["probabilities"], b["probabilities"], strict=True)
                        ),
                    )
                    if max(
                        range(len(a["probabilities"])), key=a["probabilities"].__getitem__
                    ) != max(range(len(b["probabilities"])), key=b["probabilities"].__getitem__):
                        raise ValueError("reference argmax mismatch")
                parity[name][regime + "-" + cell] = maxdelta
                if maxdelta > 1e-6:
                    raise ValueError("reference probability mismatch")
            for label, a, b in [
                ("alias_generic", "generic", "generic_alias"),
                ("alias_domain", "domain", "domain_alias"),
                ("wording_without_alias", "generic", "domain"),
                ("wording_with_alias", "generic_alias", "domain_alias"),
            ]:

                def aligned(cell, model_name=name, numeric_regime=regime):
                    return [
                        dict(p, id=p["id"].split(":", 1)[1])
                        for p in predictions[model_name][numeric_regime + "-" + cell]
                    ]

                effects[name][regime + "-" + label] = compare(aligned(a), aligned(b))
        for variant in VARIANTS:
            pairs[name][variant] = opposing_pairs(
                [r for r in rows if r["variant"] == variant], predictions[name][variant]
            )
    for directory, h in checkpoints.values():
        if sha(root / directory / "best-delta.pt") != h:
            raise ValueError("source changed")
    write(
        out / "factorial.json",
        dict(
            parity_max_abs=parity,
            paired_effects=effects,
            opposing_pairs=pairs,
            status="PASS",
            input_sha256=sha(data / "inputs.jsonl"),
        ),
    )


if __name__ == "__main__":
    main()
