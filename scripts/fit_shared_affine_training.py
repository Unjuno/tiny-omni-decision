"""Bounded training-only affine fit; never loads validation or audit inputs."""

import argparse
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from scripts.train_dinov2_image_probe import sha, write
from tiny_omni_decision.bilinear_readout import BilinearCandidateScorer
from tiny_omni_decision.count_diagnostic import fixed_candidate_affine, fixed_taxonomy_loss
from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES
from tiny_omni_decision.global_readout import fold_global_readout
from tiny_omni_decision.videoprism_spatial_features import readout_features


def quality(logits, targets):
    probabilities = logits.detach().softmax(-1).numpy()
    truth = targets.numpy()

    def measure(p, y):
        confidence = p.max(-1)
        correct = p.argmax(-1) == y
        onehot = np.eye(10)[y]
        ece = 0.0
        bins = np.minimum((confidence * 15).astype(int), 14)
        for index in range(15):
            mask = bins == index
            if mask.any():
                ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
        return dict(count=len(y), accuracy=float(correct.mean()),
                    nll=float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-300)).mean()),
                    brier=float(((p - onehot) ** 2).sum(-1).mean()),
                    ece_15_bins=float(ece), mean_confidence=float(confidence.mean()))

    return dict(overall=measure(probabilities.reshape(-1, 10), truth.reshape(-1)),
                tasks={task: measure(probabilities[:, i], truth[:, i])
                       for i, task in enumerate(CLEVR4_TAXONOMIES)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    source = root / "shared-spatial-image-readout-01"
    corpus = root / "clevr4-full-coverage-seed17"
    cache = root / "videoprism-spatial-image-cache-01/train-cells.npy"
    weights = source / "global/best.pt"
    options_path = source / "option-embeddings.npy"
    labels = corpus / "train-labels.jsonl"
    manifest = corpus / "train-image-manifest.jsonl"
    pins = {
        cache: "0355ae96cfadf421266452cca5df950ba6d0a09e3bbbfc10d71c356bc556fa5b",
        weights: "dc7053b7c6e949a08e3e3dae4ff5a4e651e042e097d6baba9756442d9fc7a48b",
        options_path: "97e212eee5b5f7121e6bdfba631620e5859403665da3ad2ffaf2b044c58a50dc",
        labels: "00f03f27982e2785636e85a50f4ecf258c0ac3ee483d7242e297cee14b50ec2e",
        manifest: "2c56815beb9d2e9951cd38686eafdbe2ecdee05e69e74b5acfad24fbefbb9010",
    }
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError(f"input changed: {path}")
    rows = [json.loads(line) for line in labels.read_text(encoding="utf-8").splitlines()]
    assets = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    if ([r["id"] for r in rows] != [r["id"] for r in assets]
            or len(rows) != 8424 or len({r["id"] for r in rows}) != 8424
            or not all(r["split"] == "train" for r in rows + assets)):
        raise ValueError("training identity/order mismatch")
    cells = np.load(cache, allow_pickle=False)
    if cells.shape != (8424, 17, 768):
        raise ValueError("feature shape mismatch")
    if args.smoke:
        rows, cells = rows[:8], cells[:8]
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(17)
    started = time.perf_counter()
    config = dict(smoke=args.smoke, started_at=datetime.now(UTC).isoformat(),
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       text=True).strip(),
                  runner_sha256=sha(Path(__file__)), inputs={str(p): h for p, h in pins.items()},
                  python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                  device="cpu", threads=2, dtype="float64", seed=17,
                  optimizer="LBFGS", lr=1, history_size=50, line_search_fn="strong_wolfe",
                  max_iter=2 if args.smoke else 10, max_eval=20,
                  calls=1 if args.smoke else 20, tolerance_grad=1e-7, tolerance_change=1e-10,
                  regularization="none", training_only=True, validation_loaded=False,
                  checkpoint_rule="earliest minimum training CE; diagnostic only",
                  unique_images=len(rows), unique_questions=len(rows) * 4)
    write(args.output / "config.json", config)
    original = BilinearCandidateScorer(13056, 384, 128).eval()
    original.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
    original = original.double()
    folded = fold_global_readout(original, condition="global")
    options = torch.from_numpy(np.load(options_path, allow_pickle=False)).double()
    weight, bias = fixed_candidate_affine(folded, options.reshape(40, 384))
    model = torch.nn.Linear(1536, 40).double()
    with torch.no_grad():
        model.weight.copy_(weight)
        model.bias.copy_(bias)
    x = torch.from_numpy(np.concatenate([cells[:, 0], cells[:, 1:].mean(1)], axis=1)).double()
    targets = torch.tensor([[list(map(str, vocab)).index(str(row[task]))
                             for task, vocab in CLEVR4_TAXONOMIES.items()] for row in rows])
    delta = 0.0
    with torch.no_grad():
        for offset in range(0, len(rows), 32):
            wide = torch.from_numpy(readout_features(cells[offset:offset + 32], "global")).double()
            actual = model(x[offset:offset + 32]).reshape(-1, 4, 10)
            for i in range(4):
                reference = original(torch.cat((wide[:, None].expand(-1, 10, -1),
                                                options[i][None].expand(len(wide), -1, -1)), -1))
                delta = max(delta, float((reference - actual[:, i]).abs().max()))
                if not torch.allclose(reference, actual[:, i], atol=1e-10, rtol=1e-10):
                    raise ValueError("initial affine identity failed")
        projected = original.candidate(options.reshape(40, 384))
        singular = torch.linalg.svdvals(projected)
        rank = int(torch.linalg.matrix_rank(projected))
    write(args.output / "identity.json", dict(max_logit_delta=delta, projected_rank=rank,
                                             singular_values=singular.tolist(),
                                             trainable_parameters=sum(
                                                 p.numel() for p in model.parameters())))
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=config["max_iter"],
                                 max_eval=20, history_size=50, line_search_fn="strong_wolfe",
                                 tolerance_grad=1e-7, tolerance_change=1e-10)
    history = []
    closure_count = 0

    def snapshot(call):
        optimizer.zero_grad()
        logits = model(x).reshape(-1, 4, 10)
        loss = fixed_taxonomy_loss(logits, targets)
        loss.backward()
        grad = max(float(p.grad.abs().max()) for p in model.parameters())
        if not np.isfinite(float(loss.detach())) or not np.isfinite(grad):
            raise FloatingPointError("nonfinite objective/gradient")
        state = optimizer.state.get(model.weight, {})
        row = dict(call=call, seconds=time.perf_counter() - started,
                   loss=float(loss.detach()), gradient_inf=grad,
                   optimizer_iterations=state.get("n_iter", 0),
                   closure_evaluations=closure_count, metrics=quality(logits, targets))
        history.append(row)
        write(args.output / "history.json", history)
        print(json.dumps({k: v for k, v in row.items() if k != "metrics"}), flush=True)
        return row, logits.detach()

    def save(name, logits):
        torch.save(model.state_dict(), args.output / f"{name}.pt")
        np.save(args.output / f"{name}-logits.npy", logits.numpy(), allow_pickle=False)

    initial, logits = snapshot(0)
    save("initial", logits)
    save("best", logits)
    best_loss, best_call = initial["loss"], 0
    reason = "budget_completed"
    with (args.output / "closures.jsonl").open("w", encoding="utf-8") as log:
        for call in range(1, config["calls"] + 1):
            def closure(call=call):
                nonlocal closure_count
                optimizer.zero_grad()
                loss = fixed_taxonomy_loss(model(x).reshape(-1, 4, 10), targets)
                loss.backward()
                grad = max(float(p.grad.abs().max()) for p in model.parameters())
                value = float(loss.detach())
                if not np.isfinite(value) or not np.isfinite(grad):
                    raise FloatingPointError("nonfinite closure")
                closure_count += 1
                log.write(json.dumps(dict(call=call, evaluation=closure_count,
                                          loss=value, gradient_inf=grad)) + "\n")
                log.flush()
                return loss

            optimizer.step(closure)
            row, logits = snapshot(call)
            if row["loss"] < best_loss:
                best_loss, best_call = row["loss"], call
                save("best", logits)
            if row["gradient_inf"] <= 1e-7:
                reason = "gradient_tolerance"
                break
    save("final", logits)
    model.load_state_dict(torch.load(args.output / "best.pt", weights_only=True))
    with torch.no_grad():
        reloaded = model(x).reshape(-1, 4, 10).numpy()
    exact = np.array_equal(reloaded, np.load(args.output / "best-logits.npy"))
    unchanged = all(sha(path) == digest for path, digest in pins.items())
    if not exact or not unchanged:
        raise ValueError("reload or source integrity failed")
    write(args.output / "result.json", dict(status="PASS_TRAINING_DIAGNOSTIC", reason=reason,
          finished_at=datetime.now(UTC).isoformat(), seconds=time.perf_counter() - started,
          best_call=best_call, best_loss=best_loss, reload_exact=exact, source_unchanged=unchanged,
          validation_loaded=False, source_projected_rank=rank, history=history,
          hashes={p.name: sha(p) for p in args.output.iterdir() if p.is_file()}))


if __name__ == "__main__":
    main()
