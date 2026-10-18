"""Frozen candidate semantics for the predeclared complete occlusion pairs."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from scripts.audit_correspondence_validation import sha, write
from scripts.train_frozen_video_probe import _text_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    model_path = root / "path-a-audio-multiquery-runtime-bundle-v2/models/text-minilm-en"
    pins = dict(
        train="8869bb903591af3a0995eeca0eeae1b50b990c8b08752a6e5d4d549f8a356fd7",
        valid="30846aaa87e2292e97ee3187c5f2f986ba3c3d57d46a249b361e5a14c0cc8ec2",
    )
    model_hashes = {
        str(p.relative_to(model_path)): sha(p) for p in model_path.rglob("*") if p.is_file()
    }
    if model_hashes["model.safetensors"] != (
        "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"
    ):
        raise ValueError("text weights changed")
    data = {}
    for split, pin in pins.items():
        path = root / f"perception-occlusion-pairs-01/{split}.json"
        if sha(path) != pin:
            raise ValueError("pairs changed")
        data[split] = json.loads(path.read_text(encoding="utf-8"))["pairs"]
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    report = dict(
        training=False,
        pair_hashes=pins,
        model_hashes=model_hashes,
        revision="4ca70771034acceecb2e72475f72050fcdde4ddc",
        parameters=sum(p.numel() for p in model.parameters()),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        script_sha256=sha(Path(__file__)),
        features={},
    )
    for split, pairs in data.items():
        rows, identities = [], []
        for pair in pairs:
            for question in pair["questions"]:
                rows.append(
                    dict(
                        question=question["question"],
                        options=pair["options"],
                        target=pair["options"][question["target"]],
                    )
                )
                identities.append(
                    dict(
                        video_id=pair["video_id"],
                        condition=question["condition"],
                        target=question["target"],
                    )
                )
        values, targets = _text_features(rows, tokenizer, model, torch.device("cpu"))
        array = torch.stack(values).numpy()
        if array.shape != (len(rows), 3, 1536) or not np.isfinite(array).all():
            raise ValueError("invalid text features")
        if targets != [r["target"] for r in identities]:
            raise ValueError("target remapping mismatch")
        repeated, _ = _text_features(rows, tokenizer, model, torch.device("cpu"))
        np.testing.assert_array_equal(array, torch.stack(repeated).numpy())
        path = args.output / f"{split}.npy"
        np.save(path, array, allow_pickle=False)
        report["features"][split] = dict(
            path=str(path),
            sha256=sha(path),
            rows=identities,
            shape=list(array.shape),
            repeat_max_abs=0,
        )
    report["status"] = "PASS_FROZEN_TEXT_FEATURES"
    write(args.output / "result.json", report)
    print(
        json.dumps(dict(status=report["status"], questions=sum(2 * len(x) for x in data.values())))
    )


if __name__ == "__main__":
    main()
