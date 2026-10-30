"""Verify same-run budget comparison without refitting or checkpoint selection."""

import json
from pathlib import Path

import numpy as np

from scripts.range_generalization import sha, write
from tiny_omni_decision.cluster_comparison import paired_cluster_interval


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    old = root / "vidimu-gated-fusion-arm-01"
    new = root / "vidimu-gated-fusion-budget-01"
    out = root / "vidimu-gated-budget-verification-01"
    out.mkdir(exist_ok=False)
    old_config, new_config = [json.loads((p / "config.json").read_text()) for p in (old, new)]
    excluded = {"epochs", "script_sha256", "source_commit"}
    if ({k: v for k, v in old_config.items() if k not in excluded}
            != {k: v for k, v in new_config.items() if k not in excluded}):
        raise ValueError("undeclared config difference")
    if sha(old / "records.json") != sha(new / "records.json"):
        raise ValueError("records changed")
    for epoch in range(101):
        filename = f"epoch-{epoch:03}.npz"
        with np.load(old / filename) as a, np.load(new / filename) as b:
            assert set(a.files) == set(b.files)
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
    history = json.loads((new / "history.json").read_text())
    if [r["epoch"] for r in history] != list(range(1001)):
        raise ValueError("incomplete budget")
    result = json.loads((new / "result.json").read_text())
    if sha(new / "best.pt") != result["best_sha256"]:
        raise ValueError("best changed")
    records = json.loads((new / "records.json").read_text())["validation"]
    with np.load(new / "epoch-100.npz") as a, np.load(new / "epoch-1000.npz") as b:
        np.testing.assert_array_equal(a["validation_targets"], b["validation_targets"])
        target = a["validation_targets"]
        index = np.arange(len(target))
        loss_a = -np.log(np.maximum(a["validation"][index, target], 1e-12))
        loss_b = -np.log(np.maximum(b["validation"][index, target], 1e-12))
        fixed_difference = paired_cluster_interval(loss_b, loss_a, [r["subject"] for r in records])
    report = dict(prefix_epochs_exact=101, prefix_arrays_per_epoch=4, config_match=True,
                  records_sha256=sha(new / "records.json"), best_epoch=result["best_epoch"],
                  fixed_1000_minus_100_nll=fixed_difference,
                  checkpoints=[r for r in history if r["epoch"] in
                               {0, 100, 250, 500, 750, 1000, result["best_epoch"]}],
                  full_budget_encounters=130 * 1000,
                  selected_encounters=130 * result["best_epoch"], unique_training_examples=130,
                  warning="development validation,3 subjects; repetitions are not new coverage")
    write(out / "result.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "checkpoints"}))


if __name__ == "__main__":
    main()
