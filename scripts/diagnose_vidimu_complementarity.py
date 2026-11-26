"""Read only saved development predictions; no inference, fitting or audit access."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.range_generalization import sha, write
from tiny_omni_decision.fusion_diagnostics import complementarity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    probabilities, hashes, reference_ids = {}, {}, None
    for arm in ("sensor", "video", "joint"):
        folder = root / f"vidimu-{arm}-arm-01"
        result_path = folder / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        records_path = folder / "records.json"
        rows = json.loads(records_path.read_text(encoding="utf-8"))["validation"]
        ids = [(r["subject"], r["activity"], r["trial"]) for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate episode")
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise ValueError("episode ordering differs")
        path = folder / f"epoch-{result['best_epoch']:03}.npz"
        with np.load(path, allow_pickle=False) as data:
            probabilities[arm] = data["validation"].copy()
            labels = data["validation_targets"].copy()
        if arm == "sensor":
            targets = labels
        elif not np.array_equal(targets, labels):
            raise ValueError("targets differ")
        for p in (path, result_path, records_path):
            hashes[str(p)] = sha(p)
    arrays = [probabilities[a] for a in ("sensor", "video", "joint")]
    report = dict(
        overall=complementarity(*arrays, targets),
        by_subject={
            subject: complementarity(
                *(a[[r[0] == subject for r in reference_ids]] for a in arrays),
                targets[[r[0] == subject for r in reference_ids]],
            )
            for subject in sorted({r[0] for r in reference_ids})
        },
        input_hashes=hashes,
        independent_subjects=len({r[0] for r in reference_ids}),
        inference=False,
        training=False,
        sealed_audit=False,
        limitation=(
            "Reused validation and selected checkpoints; label-aware oracle is not deployable."
        ),
    )
    if not all(sha(Path(p)) == h for p, h in hashes.items()):
        raise ValueError("source changed")
    write(args.output / "result.json", report)
    print(json.dumps(report["overall"]))


if __name__ == "__main__":
    main()
