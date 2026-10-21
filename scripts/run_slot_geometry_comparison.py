import json
from pathlib import Path

import numpy as np
import torch

from scripts.audit_correspondence_validation import sha
from scripts.evaluate_slot_readout import evaluate
from scripts.train_slot_readout import fit, load_train


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    source = Path("C:/CodexArtifacts/pretrained-reuse-r1/videosaur-train-slot-geometry-01")
    bank, data, hashes = load_train(
        Path("C:/CodexArtifacts/pretrained-reuse-r1/videosaur-train-slots-01")
    )
    ledger = json.loads((source / "result.json").read_text())
    assert (
        sha(source / "result.json")
        == "a40c7f27c46bb689b1dd23188465e679b65b8c452d3271b220c7afb4ddd6b2ab"
    )
    assert ledger["count"] == 99 and ledger["weights_unchanged"]
    geo = {}
    for row in ledger["features"]:
        path = Path(row["path"])
        assert sha(path) == row["sha256"]
        value = np.load(path, allow_pickle=False)
        assert value.shape == (224, 134) and np.isfinite(value).all()
        assert np.array_equal(value[:, :129], bank[row["video_id"]].numpy())
        assert row["video_id"] not in geo
        geo[row["video_id"]] = torch.from_numpy(value)
        hashes[str(path)] = sha(path)
    assert set(geo) == set(bank)
    split_path = Path("docs/pretrained_reuse/R2_SLOT_INTERNAL_SPLIT_20261013.json")
    assert sha(split_path) == "dd8d05c45fab8cbf243753c5e7d405c61b556b0965e382ba5c4cf6676bece54f"
    split = json.loads(split_path.read_text())
    hashes[str(split_path.resolve())] = sha(split_path)
    assert not set(split["fit_video_ids"]) & set(split["development_video_ids"])
    assert set(split["fit_video_ids"] + split["development_video_ids"]) == set(geo)
    for mode in ["zero_geometry", "geometry"]:
        features = {k: v.clone() for k, v in geo.items()}
        if mode == "zero_geometry":
            for v in features.values():
                v[:, 129:] = 0

        def subset(ids, features=features):
            indices = [i for i, p in enumerate(data["pairs"]) if p["video_id"] in set(ids)]
            rows = torch.tensor([j for i in indices for j in [2 * i, 2 * i + 1]])
            return {k: features[k] for k in ids}, dict(
                pairs=[data["pairs"][i] for i in indices],
                text=data["text"][rows],
                targets=data["targets"][rows],
            )

        tb, td = subset(split["fit_video_ids"])
        vb, vd = subset(split["development_video_ids"])
        output = root / f"perception-slot-{mode}-internal-01"
        fit(tb, td, output, hashes, input_dim=134)
        evaluate(
            vb,
            vd,
            root / f"perception-slot-{mode}-internal-eval-01",
            hashes,
            {
                name: (output / (name + ".pt"), sha(output / (name + ".pt")))
                for name in ["final", "untrained"]
            },
            input_dim=134,
        )


if __name__ == "__main__":
    main()

