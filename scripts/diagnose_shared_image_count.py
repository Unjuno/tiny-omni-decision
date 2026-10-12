"""Frozen selected global readout: training-only count errors and affine identity."""

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from scripts.report_shared_spatial_images import metrics
from scripts.train_dinov2_image_probe import sha, write
from tiny_omni_decision.bilinear_readout import BilinearCandidateScorer
from tiny_omni_decision.count_diagnostic import count_errors, fixed_candidate_affine
from tiny_omni_decision.videoprism_spatial_features import readout_features


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    source = root / "shared-spatial-image-readout-01"
    corpus = root / "clevr4-full-coverage-seed17"
    cache = root / "videoprism-spatial-image-cache-01/train-cells.npy"
    pins = {
        cache: "0355ae96cfadf421266452cca5df950ba6d0a09e3bbbfc10d71c356bc556fa5b",
        source
        / "global/best.pt": "dc7053b7c6e949a08e3e3dae4ff5a4e651e042e097d6baba9756442d9fc7a48b",
        source
        / "option-embeddings.npy": "97e212eee5b5f7121e6bdfba631620e5859403665da3ad2ffaf2b044c58a50dc",
        corpus
        / "train-labels.jsonl": "00f03f27982e2785636e85a50f4ecf258c0ac3ee483d7242e297cee14b50ec2e",
        corpus
        / "train-image-manifest.jsonl": "2c56815beb9d2e9951cd38686eafdbe2ecdee05e69e74b5acfad24fbefbb9010",
    }
    for path, digest in pins.items():
        assert sha(path) == digest
    rows = [json.loads(x) for x in (corpus / "train-labels.jsonl").read_text().splitlines()]
    manifest = [
        json.loads(x) for x in (corpus / "train-image-manifest.jsonl").read_text().splitlines()
    ]
    assert [r["id"] for r in rows] == [r["id"] for r in manifest]
    assert len(rows) == len({r["id"] for r in rows}) == 8424
    assert all(r["split"] == "train" for r in rows + manifest)
    out = root / "shared-count-training-diagnostic-01"
    out.mkdir(exist_ok=False)
    config = dict(
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        script_sha256=sha(Path(__file__)),
        inputs={str(p): h for p, h in pins.items()},
        training_only=True,
        logits_atol=1e-4,
        logits_rtol=1e-5,
        probabilities_atol=1e-5,
        allowed_answer_changes=0,
        fitting=False,
        checkpoint_selection=False,
        validation_loaded=False,
        sealed_audit_loaded=False,
        device="cpu",
        threads=2,
    )
    write(out / "config.json", config)
    torch.set_num_threads(2)
    model = BilinearCandidateScorer(13056, 384, 128).eval()
    model.load_state_dict(
        torch.load(source / "global/best.pt", weights_only=True, map_location="cpu")
    )
    cells = np.load(cache, allow_pickle=False)
    assert cells.shape == (8424, 17, 768)
    options = torch.from_numpy(np.load(source / "option-embeddings.npy", allow_pickle=False)[2])
    weight, bias = fixed_candidate_affine(model, options)
    outputs, affine = [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(rows), 128):
            values = torch.from_numpy(readout_features(cells[offset : offset + 128], "global"))
            inputs = torch.cat(
                (values[:, None].expand(-1, 10, -1), options[None].expand(len(values), -1, -1)), -1
            )
            outputs.append(model(inputs))
            affine.append(values @ weight.T + bias)
    outputs, affine = torch.cat(outputs), torch.cat(affine)
    probabilities = outputs.softmax(-1).numpy()
    differences = affine - outputs
    changed = int((affine.argmax(-1) != outputs.argmax(-1)).sum())
    prob_delta = float((affine.softmax(-1) - outputs.softmax(-1)).abs().max())
    passed = bool(
        torch.allclose(outputs, affine, atol=1e-4, rtol=1e-5)
        and prob_delta <= 1e-5
        and changed == 0
    )
    classes = list(range(1, 11))
    predictions = [
        dict(
            id=r["id"] + ":count",
            image_id=r["id"],
            taxonomy="count",
            target=str(r["count"]),
            prediction=str(int(p.argmax()) + 1),
            options=[str(i) for i in classes],
            probabilities=p.tolist(),
        )
        for r, p in zip(rows, probabilities, strict=True)
    ]
    write(out / "training-predictions.json", predictions)
    stats = count_errors(
        [r["count"] for r in rows], [int(r["prediction"]) for r in predictions], classes=classes
    )
    by_count = {str(c): metrics([r for r in predictions if r["target"] == str(c)]) for c in classes}
    groups = {}
    for field in ("color", "shape", "texture"):
        groups[field] = {
            value: metrics([p for r, p in zip(rows, predictions, strict=True) if r[field] == value])
            for value in sorted({r[field] for r in rows})
        }
    write(
        out / "result.json",
        dict(
            status="PASS_DIAGNOSTIC" if passed else "FAIL_AFFINE_GATE",
            metrics=metrics(predictions),
            errors=stats,
            by_true_count=by_count,
            training_subgroups=groups,
            max_logit_delta=float(differences.abs().max()),
            max_probability_delta=prob_delta,
            changed_answers=changed,
            seconds=time.perf_counter() - started,
            source_unchanged=sha(source / "global/best.pt") == pins[source / "global/best.pt"],
            predictions_sha256=sha(out / "training-predictions.json"),
        ),
    )
    print(json.dumps(dict(passed=passed, metrics=metrics(predictions), errors=stats)))


if __name__ == "__main__":
    main()
