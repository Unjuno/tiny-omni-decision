"""Fresh algebraic export; fixed validation replay, no fitting or selection."""

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from scripts.report_shared_spatial_images import metrics
from scripts.train_dinov2_image_probe import sha, write
from tiny_omni_decision.bilinear_readout import BilinearCandidateScorer
from tiny_omni_decision.global_readout import fold_global_readout
from tiny_omni_decision.videoprism_spatial_features import readout_features


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    source = root / "shared-spatial-image-readout-01"
    cache = root / "videoprism-spatial-image-cache-01"
    out = root / "shared-global-folded-readout-01"
    out.mkdir(exist_ok=False)
    config = dict(source_commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(), script_sha256=sha(Path(__file__)),
        source_sha256="dc7053b7c6e949a08e3e3dae4ff5a4e651e042e097d6baba9756442d9fc7a48b",
        logits_atol=1e-4, logits_rtol=1e-5, probabilities_atol=1e-5,
        metric_atol=1e-5, answer_changes_allowed=0, threads=2, dtype="float32",
        transform="W_mean=sum(W_cell[0:16])/4; separate unscaled summary and cell mean",
        fitting=False, sealed_audit=False)
    write(out / "config.json", config)
    assert sha(source / "global/best.pt") == config["source_sha256"]
    parent = json.loads((source / "config.json").read_text())
    for name, digest in parent["cache_hashes"].items():
        assert sha(cache / name) == digest
    assert sha(source / "option-embeddings.npy") == parent["option_embeddings_sha256"]
    original = BilinearCandidateScorer(13056, 384, 128).eval()
    original.load_state_dict(torch.load(source / "global/best.pt", weights_only=True,
                                        map_location="cpu"))
    torch.set_num_threads(2)
    model = fold_global_readout(original, condition="global")
    torch.save(model.state_dict(), out / "readout.pt")
    reload = BilinearCandidateScorer(1536, 384, 128).eval()
    reload.load_state_dict(torch.load(out / "readout.pt", weights_only=True))
    cells = np.load(cache / "val-cells.npy", allow_pickle=False)
    wide = torch.from_numpy(readout_features(cells, "global"))
    narrow = torch.from_numpy(np.concatenate((cells[:, 0], cells[:, 1:].mean(1)), 1))
    options = torch.from_numpy(np.load(source / "option-embeddings.npy", allow_pickle=False))
    saved = json.loads((source / "global/matched-predictions.json").read_text())
    records = json.loads((cache / "records.json").read_text())["val"]
    assert [r["image_id"] for r in saved] == [r["id"] for r in records for _ in range(4)]
    logits, old_logits, timings = [], [], {"original": [], "folded": []}
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, 2048, 128):
            ids = torch.arange(offset, min(offset + 128, 2048))
            text = options[ids % 4]
            a = torch.cat((wide[ids // 4, None].expand(-1, 10, -1), text), -1)
            b = torch.cat((narrow[ids // 4, None].expand(-1, 10, -1), text), -1)
            old, new = original(a), model(b)
            torch.testing.assert_close(old, new, atol=1e-4, rtol=1e-5)
            assert torch.equal(new, reload(b))
            old_logits.append(old)
            logits.append(new)
        # Fixed batch128 scorer-only benchmark; inputs/checkpoints already resident.
        for _ in range(3):
            original(a)
            model(b)
        for repeat in range(20):
            arms = [("original", original, a), ("folded", model, b)]
            for name, net, inputs in arms[::1 if repeat % 2 == 0 else -1]:
                start = time.perf_counter()
                net(inputs)
                timings[name].append(time.perf_counter() - start)
    probabilities = torch.cat(logits).softmax(-1).numpy()
    old_probs = torch.cat(old_logits).softmax(-1).numpy()
    np.testing.assert_allclose(probabilities, old_probs, atol=1e-5, rtol=0)
    np.testing.assert_allclose(
        probabilities, [r["probabilities"] for r in saved], atol=1e-5, rtol=0)
    rows = [dict(r, probabilities=p.tolist(), prediction=r["options"][int(p.argmax())])
            for r, p in zip(saved, probabilities, strict=True)]
    assert all(a["prediction"] == b["prediction"] for a, b in zip(rows, saved, strict=True))
    actual, reference = metrics(rows), metrics(saved)
    for name in actual:
        assert abs(actual[name] - reference[name]) <= 1e-5
    write(out / "predictions.json", rows)
    write(out / "result.json", dict(status="PASS_REPLAY", config=config,
        metrics=actual, reference_metrics=reference, count=2048, unique_images=512,
        max_logit_delta=float((torch.cat(logits)-torch.cat(old_logits)).abs().max()),
        max_probability_delta=float(np.abs(probabilities-old_probs).max()),
        parameters=sum(p.numel() for p in model.parameters()),
        original_parameters=sum(p.numel() for p in original.parameters()),
        bytes=(out / "readout.pt").stat().st_size,
        original_bytes=(source / "global/best.pt").stat().st_size,
        weights_sha256=sha(out / "readout.pt"), predictions_sha256=sha(out / "predictions.json"),
        source_unchanged=sha(source / "global/best.pt") == config["source_sha256"],
        reload_exact=True, changed_answers=0, timings_seconds=timings,
        seconds=time.perf_counter()-started, torch=torch.__version__, cloud_cost=0))
    print(json.dumps(dict(status="PASS_REPLAY", metrics=actual,
                         parameters=sum(p.numel() for p in model.parameters()))))


if __name__ == "__main__":
    main()
