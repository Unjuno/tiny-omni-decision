import json
from pathlib import Path

import torch

from scripts.audit_correspondence_validation import sha
from scripts.evaluate_slot_readout import evaluate
from scripts.train_slot_readout import fit, load_train


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    split_path = Path("docs/pretrained_reuse/R2_SLOT_INTERNAL_SPLIT_20261013.json")
    split = json.loads(split_path.read_text())
    bank, data, hashes = load_train(
        Path("C:/CodexArtifacts/pretrained-reuse-r1/videosaur-train-slots-01")
    )
    assert not set(split["fit_video_ids"]) & set(split["development_video_ids"])
    assert set(split["fit_video_ids"] + split["development_video_ids"]) == set(bank)

    def subset(ids):
        indices = [i for i, p in enumerate(data["pairs"]) if p["video_id"] in set(ids)]
        rows = torch.tensor([j for i in indices for j in [2 * i, 2 * i + 1]])
        return {k: bank[k] for k in ids}, dict(
            pairs=[data["pairs"][i] for i in indices],
            text=data["text"][rows],
            targets=data["targets"][rows],
        )

    train_bank, train = subset(split["fit_video_ids"])
    dev_bank, dev = subset(split["development_video_ids"])
    assert len(train["pairs"]) == 79 and len(dev["pairs"]) == 20
    hashes[str(split_path.resolve())] = sha(split_path)
    output = root / "perception-occlusion-slot-internal-baseline-01"
    fit(train_bank, train, output, hashes)
    evaluate(
        dev_bank,
        dev,
        root / "perception-occlusion-slot-internal-baseline-eval-01",
        hashes,
        {
            name: (output / (name + ".pt"), sha(output / (name + ".pt")))
            for name in ["final", "untrained"]
        },
    )


if __name__ == "__main__":
    main()
