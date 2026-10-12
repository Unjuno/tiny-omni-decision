"""One fixed endpoint comparison on reused development images; no fitting."""

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from scripts.report_shared_spatial_images import metrics
from scripts.train_dinov2_image_probe import HASHES, sha, write
from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES
from tiny_omni_decision.dinov2_features import require_image_splits
from tiny_omni_decision.paired_image_comparison import paired_image_metrics


def prediction_rows(labels, probabilities):
    p = np.asarray(probabilities)
    if (p.shape != (len(labels), 4, 10) or not np.isfinite(p).all()
            or np.any(p < 0) or not np.allclose(p.sum(-1), 1, atol=1e-10, rtol=0)):
        raise ValueError("finite normalized [image,4,10] probabilities required")
    result = []
    for row, tasks in zip(labels, p, strict=True):
        for (task, vocabulary), values in zip(CLEVR4_TAXONOMIES.items(), tasks, strict=True):
            options = list(map(str, vocabulary))
            target = str(row[task])
            if target not in options:
                raise ValueError("unknown taxonomy target")
            result.append(dict(id=row["id"] + ":" + task, image_id=row["id"],
                               taxonomy=task, target=target, options=options,
                               prediction=options[int(values.argmax())],
                               probabilities=values.tolist()))
    return result


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    fit = root / "shared-affine-training-fit-01"
    cache = root / "videoprism-spatial-image-cache-01"
    corpus = root / "clevr4-sample-seed17-1024-512"
    train = root / "clevr4-full-coverage-seed17/train-image-manifest.jsonl"
    saved = root / "shared-spatial-image-readout-01/global/matched-predictions.json"
    pins = {
        fit / "initial.pt": "3dd6326e4c371b6f58ba47b2a8b3850229c5d117e0e507e280613dacbc8fef25",
        fit / "final.pt": "817d2d054b1b8998da2687fd2d0b64396fa5d7c7d23d0d66d73bde130fc0b015",
        fit / "history.json": "cae553b2b99bd0744474150a642ff1a5da89c3ac46e856ad8b65b7a954acf8b2",
        cache / "val-cells.npy": "1be3736b12cea9265233ebf005234c8e81cfa3c96645704e0cc2d00944b85fec",
        cache / "records.json": "32944fc822c45a7a93ee06904c9a97acaa5bf6364c33c40525f535c59da55b41",
        saved: "710f70258891a7080da42479556f1c117df84d49ebc1b9c1bbd42899ae82d3bd",
        train: HASHES["train-image-manifest.jsonl"],
        corpus / "val-labels.jsonl": HASHES["val-labels.jsonl"],
        corpus / "val-image-manifest.jsonl": HASHES["val-image-manifest.jsonl"],
    }
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError(f"changed input: {path}")

    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    def lines(path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    labels = lines(corpus / "val-labels.jsonl")
    manifest = lines(corpus / "val-image-manifest.jsonl")
    require_image_splits(lines(train), manifest)
    if (len(labels) != 512 or [r["id"] for r in labels] != [r["id"] for r in manifest]
            or manifest != read(cache / "records.json")["val"]
            or not all(r["split"] == "val" for r in labels + manifest)):
        raise ValueError("validation identity/order mismatch")
    cells = np.load(cache / "val-cells.npy", allow_pickle=False)
    if cells.shape != (512, 17, 768) or not np.isfinite(cells).all():
        raise ValueError("invalid frozen features")
    out = root / "shared-affine-endpoint-validation-01"
    out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    started = time.perf_counter()
    config = dict(source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                        text=True).strip(),
                  runner_sha256=sha(Path(__file__)), inputs={str(p): h for p, h in pins.items()},
                  training=False, selection=False, calibration=False, sealed_audit=False,
                  dtype="float64", device="cpu", threads=2, images=512, questions=2048,
                  bootstrap_replicates=10000, bootstrap_seed=17,
                  protocol_sha256=sha(Path(
                      "docs/pretrained_reuse/R2_AFFINE_ENDPOINT_PROTOCOL_20261012.md")))
    write(out / "config.json", config)
    x = torch.from_numpy(np.concatenate((cells[:, 0], cells[:, 1:].mean(1)), axis=1)).double()
    predictions, quality = {}, {}
    for name in ("initial", "final"):
        model = torch.nn.Linear(1536, 40).double().eval()
        model.load_state_dict(torch.load(fit / f"{name}.pt", weights_only=True, map_location="cpu"))
        with torch.inference_mode():
            probabilities = model(x).reshape(-1, 4, 10).softmax(-1).numpy()
        predictions[name] = prediction_rows(labels, probabilities)
        write(out / f"{name}-predictions.json", predictions[name])
        quality[name] = dict(overall=metrics(predictions[name]), tasks={
            task: metrics([r for r in predictions[name] if r["taxonomy"] == task])
            for task in CLEVR4_TAXONOMIES})
    original = read(saved)
    for a, b in zip(original, predictions["initial"], strict=True):
        if any(a[k] != b[k] for k in ("id", "image_id", "target", "options", "prediction")):
            raise ValueError("initial replay identity/answer mismatch")
    delta = float(np.max(np.abs(np.array([r["probabilities"] for r in original]) -
                               np.array([r["probabilities"] for r in predictions["initial"]]))))
    if delta > 1e-5:
        raise ValueError("initial probability replay failed")
    comparisons = {"overall": paired_image_metrics(predictions["initial"], predictions["final"])}
    for task in CLEVR4_TAXONOMIES:
        comparisons[task] = paired_image_metrics(
            [r for r in predictions["initial"] if r["taxonomy"] == task],
            [r for r in predictions["final"] if r["taxonomy"] == task])
    history = read(fit / "history.json")
    if not all(sha(p) == h for p, h in pins.items()):
        raise ValueError("source changed during evaluation")
    primary = comparisons["count"]["metrics"]
    clear = (primary["accuracy"]["percentile_95_interval"][0] > 0
             and primary["nll"]["percentile_95_interval"][1] < 0)
    result = dict(status="PASS_EVALUATION", primary_count_improvement=(
                  "CLEAR_ON_REUSED_VALIDATION" if clear else "UNCERTAIN_OR_NOT_SUPPORTED"),
                  seconds=time.perf_counter() - started, metrics=quality, comparisons=comparisons,
                  initial_probability_max_delta=delta, changed_initial_answers=0,
                  training_metrics={"initial": history[0]["metrics"],
                                    "final": history[-1]["metrics"]},
                  source_unchanged=True,
                  hashes={p.name: sha(p) for p in out.iterdir() if p.is_file()})
    write(out / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
