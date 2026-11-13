"""Reconcile complete consumed TRAIN inference with saved validation, without fitting."""

import json
from pathlib import Path

import numpy as np

from scripts.native_consumed_evaluation import common_task_gap
from scripts.native_fit import summarize
from scripts.native_video_cache import sha

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
OUT = ROOT / "native-consumed-analysis-01"
REPO = Path("C:/CodexWorktrees/pretrained-reuse-omni")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


probe, replay = ROOT / "native-consumed-probe-01", ROOT / "native-consumed-replay-01"
first, second = read(probe / "result.json"), read(replay / "result.json")
assert (
    first["status"] == "PASS_NATIVE_CONSUMED_PROBE"
    and second["status"] == "PASS_NATIVE_CONSUMED_REPLAY"
)
assert first["validated_hashes"] == second["validated_hashes"]
for folder, receipt in ((probe, first), (replay, second)):
    for name, digest in receipt["artifacts"].items():
        assert sha(folder / name) == digest, (folder, name)
config = read(probe / "effective-config.json")
plan = read(probe / "plan.json")
corpus = Path(config["corpus"])
cr = read(corpus / "result.json")
assert sha(corpus / "validation.jsonl") == cr["artifacts"]["validation.jsonl"]
train = {
    r["id"]: r
    for r in (
        json.loads(s) for s in (corpus / "train.jsonl").read_text(encoding="utf-8").splitlines()
    )
}
train = [train[i] for i in plan["ids"]]
validation = [
    json.loads(s) for s in (corpus / "validation.jsonl").read_text(encoding="utf-8").splitlines()
]
models = {}
for name, entry in config["models"].items():
    scores = np.load(probe / (name + ".npy"), allow_pickle=False)
    np.testing.assert_array_equal(scores, np.load(replay / (name + ".npy"), allow_pickle=False))
    assert scores.shape == (6656, 40)
    metrics = read(probe / (name + "-metrics.json"))
    assert summarize(train, scores) == metrics["unique"]
    fit = Path(entry["fit"])
    receipt = read(fit / "result.json")
    step = 0 if name == "initial" else entry["fit_steps"]
    val_file = fit / f"validation-{step:04}.npy"
    assert sha(val_file) == receipt["artifacts"][val_file.name]
    assert read(fit / "validation-ids.json") == [
        dict(id=r["id"], option_ids=r["option_ids"], target_index=r["target_index"])
        for r in validation
    ]
    val = np.load(val_file, allow_pickle=False)
    gap = common_task_gap(train, scores, validation, val)
    models[name] = dict(
        train=metrics,
        validation=summarize(validation, val),
        common_task_gap=gap,
        checkpoint=str(fit / entry["checkpoint"]),
        checkpoint_sha256=sha(fit / entry["checkpoint"]),
    )
report = dict(
    status="PASS_COMPLETE_CONSUMED_TRAIN_ANALYSIS",
    source_commit=first["source_commit"],
    config_sha256=first["config_sha256"],
    analysis_sha256=sha(Path(__file__)),
    plan=plan,
    models=models,
    receipts={
        p: dict(
            path=str(ROOT / f"native-consumed-{p}-01"),
            sha256=sha(ROOT / f"native-consumed-{p}-01/result.json"),
            result=read(ROOT / f"native-consumed-{p}-01/result.json"),
        )
        for p in ("smoke", "probe", "replay")
    },
    cloud_cost=0,
)
(OUT / "full-result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print(
    "common task scope",
    {
        k: len(models["baseline"]["common_task_gap"][k])
        for k in ("common_tasks", "train_only_tasks", "validation_only_tasks")
    },
)
for name, m in models.items():
    g = m["common_task_gap"]
    print(
        name,
        {
            mod: dict(
                train=g["train"]["modality_macro"][mod]["accuracy"],
                validation=g["validation"]["modality_macro"][mod]["accuracy"],
                train_nll=g["train"]["modality_macro"][mod]["nll"],
                val_nll=g["validation"]["modality_macro"][mod]["nll"],
            )
            for mod in g["train"]["modality_macro"]
        },
    )
