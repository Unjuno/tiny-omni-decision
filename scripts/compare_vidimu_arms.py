"""Fixed selected-checkpoint VIDIMU controls, without optimization."""
import json

import numpy as np
import torch

from scripts.range_generalization import sha, write
from scripts.train_sensor_subjects import metrics
from scripts.train_vidimu_multimodal_arm import ROOT, load
from tiny_omni_decision.cluster_comparison import paired_cluster_interval


def main():
    torch.set_num_threads(4)
    out = ROOT / "vidimu-comparison-controls-01"
    out.mkdir(exist_ok=False)
    x, placement, targets, rows = load("validation", "joint")
    ids = [f"{r['subject']}_{r['activity']}_{r['trial']}" for r in rows]
    control_path = ROOT / "vidimu-validation-control-01/permutation.json"
    control = json.loads(control_path.read_text())
    if control["ids"] != ids:
        raise ValueError("control order differs")
    permutation = np.array([ids.index(x) for x in control["replacement_ids"]])
    if not np.array_equal(placement, placement[permutation]):
        raise ValueError("control changed placement")
    folder = ROOT / "vidimu-joint-arm-01"
    result = json.loads((folder / "result.json").read_text())
    if sha(folder / "best.pt") != result["best_sha256"]:
        raise ValueError("checkpoint changed")
    with np.load(folder / "normalization.npz") as norm:
        standardized = (x - norm["mean"]) / norm["std"]
    model = torch.nn.Linear(6506, 13)
    model.load_state_dict(torch.load(folder / "best.pt", weights_only=True))
    model.eval()
    variants = {"matched": standardized.copy()}
    for modality, section in (("video", slice(0, 6144)), ("sensor", slice(6144, 6504))):
        shuffled = standardized.copy()
        shuffled[:, section] = standardized[permutation, section]
        variants[f"shuffle_{modality}"] = shuffled
        missing = standardized.copy()
        missing[:, section] = 0
        variants[f"missing_{modality}"] = missing
    predictions = {}
    for name, values in variants.items():
        tensor = torch.from_numpy(np.concatenate(
            (values, np.eye(2, dtype=np.float32)[placement]), axis=1))
        with torch.inference_mode():
            predictions[name] = model(tensor).softmax(-1).numpy()
    with np.load(folder / f"epoch-{result['best_epoch']:03}.npz") as saved:
        np.testing.assert_array_equal(predictions["matched"], saved["validation"])
    losses = {}
    for arm in ("sensor", "video", "joint"):
        f = ROOT / f"vidimu-{arm}-arm-01"
        r = json.loads((f / "result.json").read_text())
        arm_rows = json.loads((f / "records.json").read_text())["validation"]
        if [(r['subject'], r['activity'], r['trial']) for r in arm_rows] != [
            (r['subject'], r['activity'], r['trial']) for r in rows
        ]:
            raise ValueError("arm order differs")
        with np.load(f / f"epoch-{r['best_epoch']:03}.npz") as saved:
            np.testing.assert_array_equal(saved['validation_targets'], targets)
            losses[arm] = -np.log(np.maximum(saved['validation'][np.arange(39), targets], 1e-12))
    groups = [r['subject'] for r in rows]
    report = {
        "controls": {n: metrics(p, targets) for n, p in predictions.items()},
        "joint_minus_unimodal_nll": {
            arm: paired_cluster_interval(losses['joint'], losses[arm], groups)
            for arm in ('sensor', 'video')},
        "control_sha256": sha(control_path), "checkpoint_sha256": sha(folder / 'best.pt'),
        "exact_matched_reload": True, "sealed_audit_read": False,
        "limitations": "3 development subjects; checkpoint selection bias; missingness is OOD",
    }
    np.savez_compressed(out / 'predictions.npz', **predictions, targets=targets, ids=ids)
    write(out / 'result.json', report)
    print(json.dumps({"controls": {n: {k: m[k] for k in ('accuracy', 'nll')}
                                  for n, m in report['controls'].items()},
                      "intervals": report['joint_minus_unimodal_nll']}))


if __name__ == '__main__':
    main()
