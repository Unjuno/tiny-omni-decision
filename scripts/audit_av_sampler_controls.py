"""Inference-only TRAIN input controls for completed sampler diagnostics."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from scripts.audit_av_paired_sampler import evaluate
from scripts.audit_av_training_signal import (
    FEATURE_SHA,
    QUERY_SHA,
    ROOT,
    make_model,
    sha,
    write_json,
)
from scripts.run_conditional_clevrer_av_readout import _encode_options
from scripts.train_clevrer_synthetic_av_temporal_decision import _metrics


def apply_control(values, name, donors):
    changed = dict(values)
    if name in ["zero_audio", "zero_video"]:
        kind = name.removeprefix("zero_")
        changed[kind] = torch.zeros_like(values[kind])
    elif name in ["shuffle_video", "shuffle_question"]:
        kind = "video" if name == "shuffle_video" else "text"
        changed[kind] = values[kind][donors]
    else:
        raise ValueError("unknown control")
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("new output directory required")
    metadata = json.loads((args.run / "run-metadata.json").read_text())
    results = json.loads((args.run / "results.json").read_text())
    if metadata["status"] != "complete" or metadata["feature_sha256"] != FEATURE_SHA:
        raise ValueError("completed matching diagnostic required")
    features = Path(metadata["command_arguments"]["features"])
    if sha(features) != FEATURE_SHA:
        raise ValueError("feature hash differs")
    torch.set_num_threads(14)
    torch.set_num_interop_threads(1)
    query, _ = _encode_options(Path(metadata["command_arguments"]["text_model_dir"]))
    query = query.clone()
    if hashlib.sha256(query.numpy().astype("<f4").tobytes()).hexdigest() != QUERY_SHA:
        raise ValueError("query hash differs")
    torch.set_num_threads(2)
    with np.load(features, allow_pickle=False) as bank:
        indexes = np.flatnonzero(bank["splits"] == "train")
        if bank["sample_ids"][indexes].tolist() != metadata["sample_ids"]:
            raise ValueError("training row order differs")
        scenes = bank["scene_ids"][indexes]
        values = {name: torch.from_numpy(bank[key][indexes].copy()) for name, key in
                  {"audio": "audios", "video": "videos", "text": "texts",
                   "targets": "targets"}.items()}
    donors = np.roll(np.arange(len(indexes)), len(indexes) // 2)
    if np.any(scenes == scenes[donors]):
        raise ValueError("fixed half-cohort shift contains same-scene donors")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"scope": "TRAIN perturbation controls; not generalization evaluation",
              "validation_scored": False, "sealed_audit_opened": False,
              "optimizer_updates": 0, "device": "cpu", "threads": 2,
              "donor_rule": "half-cohort cyclic row shift; all donors from different scenes",
              "donor_sha256": hashlib.sha256(donors.astype("<i8").tobytes()).hexdigest(),
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                        cwd=ROOT, text=True).strip(),
              "runner_sha256": sha(__file__), "reference_results_sha256": sha(
                  args.run / "results.json"), "results": {}}
    for arm in ["original", "paired"]:
        checkpoint = args.run / f"diagnostic-final-{arm}.pt"
        if sha(checkpoint) != results[arm]["checkpoint_sha256"]:
            raise ValueError("checkpoint hash differs")
        model = make_model(False)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        reference = evaluate(model, values, query)
        saved = torch.from_numpy(np.load(args.run / f"train-final-logits-{arm}.npy"))
        torch.testing.assert_close(reference, saved, rtol=0, atol=0)
        report["results"][arm] = {"original": _metrics(reference, values["targets"])}
        for name in ["zero_audio", "zero_video", "shuffle_video", "shuffle_question"]:
            logits = evaluate(model, apply_control(values, name, torch.from_numpy(donors)), query)
            report["results"][arm][name] = {
                **_metrics(logits, values["targets"]),
                "prediction_change_fraction": float((reference.argmax(-1) != logits.argmax(-1))
                                                    .float().mean()),
                "max_probability_change": float((reference.softmax(-1) - logits.softmax(-1))
                                                 .abs().max()),
            }
            np.save(args.output / f"{arm}-{name}-logits.npy", logits.numpy(), allow_pickle=False)
        if sha(checkpoint) != results[arm]["checkpoint_sha256"]:
            raise RuntimeError("checkpoint changed")
    report.update(wall_seconds=time.perf_counter() - started, status="complete",
                  exact_original_logit_reload=True)
    write_json(args.output / "report.json", report)
    print(json.dumps(report["results"], indent=2))


if __name__ == "__main__":
    main()
