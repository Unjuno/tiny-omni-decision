"""Fixed reverse-option diagnostic after both learning runs finish."""

import json
import os
import time
from pathlib import Path


def main():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    import yaml
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from scripts.range_generalization import sha, write
    from scripts.train_frozen_text_probe import _load_examples, _metrics
    from scripts.verify_joint_result import verify_joint_pair
    from tiny_omni_decision.joint_candidates import encode_matched, joint_logits
    from tiny_omni_decision.schema import DatasetManifest

    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    runs = [root / n for n in ("independent-matched-training-01", "joint-readout-training-01")]

    def read(p):
        return json.loads(p.read_text(encoding="utf-8"))

    triples = [tuple(read(p / f"{n}.json") for n in ("config", "result", "history")) for p in runs]
    verify_joint_pair(*triples)
    output = root / "joint-readout-order-diagnostic-01"
    if output.exists():
        raise FileExistsError(output)
    base = Path("C:/CodexArtifacts/pretrained-reuse-r1/crossencoder-minilm-l6-audit-01/model")
    if sha(base / "model.safetensors") != (
        "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"
    ):
        raise ValueError("base changed")
    for run, (config, result, _) in zip(runs, triples, strict=True):
        if sha(run / "best-delta.pt") != result["best_sha256"]:
            raise ValueError("selected checkpoint changed")
        for path, expected in config["input_hashes"].items():
            if sha(Path(path)) != expected:
                raise ValueError("source changed")
    manifest = DatasetManifest.model_validate(
        yaml.safe_load(Path("manifests/dataset.example.yaml").read_text(encoding="utf-8"))
    )
    original = [
        dict(id=e.id, question=e.question, state=e.state, options=e.options, target=e.target)
        for e in _load_examples(
            root / "text/validation.jsonl",
            manifest.model_copy(update={"split": "validation"}),
            True,
        )
    ]
    curriculum = [
        json.loads(x)
        for x in (root / "conditional-curriculum-01/validation.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    rows_by_source = dict(original=original, curriculum=curriculum)
    assert [[r["id"] for r in rows] for rows in rows_by_source.values()] == triples[0][0][
        "validation_ids"
    ]
    output.mkdir()
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    summary = {}
    for run, (config, result, _) in zip(runs, triples, strict=True):
        model = AutoModelForSequenceClassification.from_pretrained(base, local_files_only=True)
        model.load_state_dict(
            torch.load(run / "best-delta.pt", map_location="cpu", weights_only=True), strict=False
        )
        model = model.eval().to("cuda")
        arm = config["arm"]
        for source, rows in rows_by_source.items():
            collected = {}
            started = time.perf_counter()
            for reverse in (False, True):
                values = []
                for start in range(0, len(rows), 8):
                    group = rows[start : start + 8]
                    enc = [
                        encode_matched(
                            tokenizer,
                            r["question"],
                            r["state"],
                            list(reversed(r["options"])) if reverse else r["options"],
                        )
                        for r in group
                    ]
                    with torch.inference_mode():
                        if arm == "joint":
                            batch = tokenizer.pad(
                                [r[0][0] for r in enc], return_tensors="pt", padding=True
                            ).to("cuda")
                            scores = joint_logits(model, batch, [r[0][1] for r in enc])
                        else:
                            batch = tokenizer.pad(
                                [p for r in enc for p in r[1]], return_tensors="pt", padding=True
                            ).to("cuda")
                            scores = model(**batch).logits[:, 0].split([len(r[1]) for r in enc])
                    values.extend(s.flip(0).cpu() if reverse else s.cpu() for s in scores)
                collected[reverse] = values
                predictions = [
                    dict(
                        id=r["id"],
                        options=r["options"],
                        target=r["target"],
                        probabilities=v.softmax(-1).tolist(),
                    )
                    for r, v in zip(rows, values, strict=True)
                ]
                write(
                    output / f"{arm}-{source}-{'reverse' if reverse else 'natural'}.json",
                    predictions,
                )
                if not reverse:
                    old = read(run / f"predictions-{result['best_step']}-{source}.json")
                    assert [r["id"] for r in old] == [r["id"] for r in predictions]
                    delta = max(
                        abs(x - y)
                        for a, b in zip(old, predictions, strict=True)
                        for x, y in zip(a["probabilities"], b["probabilities"], strict=True)
                    )
                    if delta > 1e-6:
                        raise ValueError(f"natural replay mismatch {delta}")
            left, right = collected[False], collected[True]
            targets = [r["options"].index(r["target"]) for r in rows]
            summary[arm + "-" + source] = dict(
                natural=_metrics(left, targets),
                reverse=_metrics(right, targets),
                natural_replay_max_abs=delta,
                changed_predictions=sum(
                    int(a.argmax() != b.argmax()) for a, b in zip(left, right, strict=True)
                ),
                maximum_probability_change=max(
                    float((a.softmax(-1) - b.softmax(-1)).abs().max())
                    for a, b in zip(left, right, strict=True)
                ),
                seconds=time.perf_counter() - started,
            )
        assert sha(run / "best-delta.pt") == result["best_sha256"]
        del model
        torch.cuda.empty_cache()
    write(
        output / "result.json",
        dict(
            status="PASS_DIAGNOSTIC_ONLY",
            results=summary,
            runner_sha256=sha(Path(__file__)),
            selection_changed=False,
            training=False,
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
