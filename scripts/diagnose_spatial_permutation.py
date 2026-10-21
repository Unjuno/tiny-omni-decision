"""Fixed inference-only spatial-cell intervention; no model selection."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.report_spatial_curve import paired, read
from scripts.train_dinov2_spatial_control import (
    CLEVR4_TAXONOMIES,
    HASHES,
    PARENT_HASHES,
    TEXT,
    BilinearCandidateScorer,
    _option_embeddings,
    _prediction_rows,
    _quality_metrics,
    _taxonomy_metrics,
    readout_features,
    sha,
    write,
)


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    source = root / "dinov2-spatial-bilinear-01"
    cache = root / "dinov2-spatial-control-02/val-cells.npy"
    labels = root / "clevr4-sample-seed17-1024-512/val-labels.jsonl"
    text = root / "path-a-audio-multiquery-runtime-bundle-v2/models/text-minilm-en"
    checkpoint = source / "spatial/best.pt"
    out = root / "dinov2-spatial-permutation-03"
    if out.exists():
        raise FileExistsError(out)
    assert sha(cache) == PARENT_HASHES["val-cells.npy"]
    assert sha(labels) == HASHES[labels.name]
    assert sha(text / "model.safetensors") == TEXT
    assert sha(checkpoint) == read(source / "result.json")["conditions"]["spatial"]["best_sha256"]
    device = torch.device("cuda")
    torch.set_num_threads(2)
    start = time.perf_counter()
    cells = np.load(cache, allow_pickle=False)
    assert cells.shape == (512, 17, 384) and np.isfinite(cells).all()
    rows = [json.loads(x) for x in labels.read_text().splitlines()]
    targets = [opts.index(str(r[t])) for r in rows for t, opts in CLEVR4_TAXONOMIES.items()]
    types = list(CLEVR4_TAXONOMIES) * len(rows)
    embeddings, encoder = _option_embeddings(device, text)
    options = torch.stack(
        [
            torch.stack([embeddings[f"{t}|{o}"] for o in opts])
            for t, opts in CLEVR4_TAXONOMIES.items()
        ]
    )
    options = options.to(device)
    del encoder, embeddings
    model = BilinearCandidateScorer(6528, 384, 128).to(device).eval()
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    model.requires_grad_(False)
    permutations = dict(
        identity=np.arange(16),
        horizontal=np.arange(16).reshape(4, 4)[:, ::-1].ravel(),
        seed17=np.random.default_rng(17).permutation(16),
    )
    out.mkdir()
    results, predictions = {}, {}
    saved = read(source / "spatial/matched-predictions.json")
    with torch.inference_mode():
        for name, permutation in permutations.items():
            assert sorted(permutation.tolist()) == list(range(16))
            changed = np.concatenate([cells[:, :1], cells[:, 1:][:, permutation]], axis=1)
            features = torch.from_numpy(readout_features(changed, "spatial")).to(device)
            logits = []
            for offset in range(0, len(targets), 128):
                ids = torch.arange(offset, min(offset + 128, len(targets)), device=device)
                visual = features[ids // 4, None].expand(-1, 10, -1)
                logits.extend(model(torch.cat([visual, options[ids % 4]], -1)).cpu().unbind())
            pred = _prediction_rows(rows, logits, targets)
            if name == "identity":
                delta = np.max(
                    np.abs(
                        np.array([p["probabilities"] for p in pred])
                        - np.array([p["probabilities"] for p in saved])
                    )
                )
                assert delta <= 1e-5, delta
            predictions[name] = pred
            write(out / (name + "-predictions.json"), pred)
            results[name] = dict(
                metrics=_quality_metrics(logits, targets),
                tasks=_taxonomy_metrics(logits, targets, types),
                permutation=permutation.tolist(),
            )
    result = dict(
        results=results,
        identity_probability_max_abs=float(delta),
        comparisons={
            k: paired(predictions["identity"], p) for k, p in predictions.items() if k != "identity"
        },
        seconds=time.perf_counter() - start,
        device="cuda",
        training=False,
        hashes={str(p): sha(p) for p in [cache, labels, checkpoint, Path(__file__)]},
    )
    write(out / "result.json", result)
    write(Path("docs/pretrained_reuse/R2_SPATIAL_PERMUTATION_RESULTS_20261012.json"), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
