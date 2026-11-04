"""Two predeclared anchored affine fits; fixed endpoint development comparison."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate_shared_affine_endpoint import prediction_rows
from scripts.fit_shared_affine_training import quality
from scripts.report_shared_spatial_images import metrics
from scripts.train_dinov2_image_probe import sha, write
from tiny_omni_decision.count_diagnostic import anchored_weight_penalty, fixed_taxonomy_loss
from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES
from tiny_omni_decision.dinov2_features import require_image_splits
from tiny_omni_decision.paired_image_comparison import paired_image_metrics


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def fit(x, targets, initial, coefficient, out, smoke):
    out.mkdir()
    model = torch.nn.Linear(1536, 40).double()
    model.load_state_dict(initial)
    anchor = initial["weight"].clone()
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=2 if smoke else 10,
                                 max_eval=20, history_size=50, line_search_fn="strong_wolfe",
                                 tolerance_grad=1e-7, tolerance_change=1e-10)
    history = []
    evaluations = 0
    started = time.perf_counter()

    def objective():
        logits = model(x).reshape(-1, 4, 10)
        ce = fixed_taxonomy_loss(logits, targets)
        penalty = anchored_weight_penalty(model.weight, anchor, coefficient)
        return logits, ce, penalty, ce + penalty

    def snapshot(call):
        optimizer.zero_grad()
        logits, ce, penalty, loss = objective()
        loss.backward()
        grad = max(float(p.grad.abs().max()) for p in model.parameters())
        if not np.isfinite(grad) or not torch.isfinite(loss):
            raise FloatingPointError("nonfinite fit")
        row = dict(call=call, ce=float(ce.detach()), penalty=float(penalty.detach()),
                   objective=float(loss.detach()), gradient_inf=grad,
                   iterations=optimizer.state.get(model.weight, {}).get("n_iter", 0),
                   evaluations=evaluations, seconds=time.perf_counter() - started,
                   training=quality(logits, targets))
        history.append(row)
        write(out / "history.json", history)
        return row

    torch.save(model.state_dict(), out / "initial.pt")
    best = snapshot(0)["objective"]
    best_call = 0
    torch.save(model.state_dict(), out / "best-objective.pt")
    with (out / "closures.jsonl").open("w", encoding="utf-8") as log:
        for call in range(1, 2 if smoke else 21):
            def closure(call=call):
                nonlocal evaluations
                optimizer.zero_grad()
                _, ce, penalty, loss = objective()
                loss.backward()
                grad = max(float(p.grad.abs().max()) for p in model.parameters())
                if not np.isfinite(grad) or not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite closure")
                evaluations += 1
                log.write(json.dumps(dict(call=call, evaluation=evaluations,
                                         ce=float(ce.detach()), penalty=float(penalty.detach()),
                                         objective=float(loss.detach()), gradient_inf=grad)) + "\n")
                log.flush()
                return loss

            optimizer.step(closure)
            row = snapshot(call)
            if row["objective"] < best:
                best, best_call = row["objective"], call
                torch.save(model.state_dict(), out / "best-objective.pt")
            print(json.dumps(dict(coefficient=coefficient, **{k: v for k, v in row.items()
                                                             if k != "training"})), flush=True)
            if row["gradient_inf"] <= 1e-7:
                break
    torch.save(model.state_dict(), out / "final.pt")
    with torch.no_grad():
        logits = model(x).reshape(-1, 4, 10)
        np.save(out / "final-training-logits.npy", logits.numpy(), allow_pickle=False)
        model.load_state_dict(torch.load(out / "final.pt", weights_only=True))
        if not torch.equal(logits, model(x).reshape(-1, 4, 10)):
            raise ValueError("reload mismatch")
    result = dict(coefficient=coefficient, history=history, best_objective_call=best_call,
                  reload_exact=True, final_sha256=sha(out / "final.pt"),
                  reason=("gradient_tolerance" if row["gradient_inf"] <= 1e-7
                          else "budget_completed"))
    write(out / "fit-result.json", result)
    return model.eval(), result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    previous = root / "shared-affine-training-fit-01"
    endpoint = root / "shared-affine-endpoint-validation-01"
    train_config = read(previous / "config.json")
    if sha(previous / "config.json") != (
            "de03947e832377eee6726787361bd699a1b90db889332ae84910a23ed0b8eff8"):
        raise ValueError("training source configuration changed")
    pins = {Path(p): h for p, h in train_config["inputs"].items()}
    pins[previous / "initial.pt"] = (
        "3dd6326e4c371b6f58ba47b2a8b3850229c5d117e0e507e280613dacbc8fef25")
    for p, h in pins.items():
        if sha(p) != h:
            raise ValueError(f"input changed: {p}")
    args.output.mkdir(parents=True, exist_ok=False)
    config = dict(source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                        text=True).strip(),
                  runner_sha256=sha(Path(__file__)), smoke=args.smoke,
                  coefficients=[0.01, 0.1], inputs={str(p): h for p, h in pins.items()},
                  device="cpu", threads=2, dtype="float64", seed=17,
                  settings_inherited=train_config, validation_selector="fixed final endpoints",
                  objective="CE + coefficient*sum((W-W_initial)^2)/(2*40); no bias penalty")
    write(args.output / "config.json", config)
    torch.set_num_threads(2)
    torch.manual_seed(17)
    rows = lines(root / "clevr4-full-coverage-seed17/train-labels.jsonl")
    cells = np.load(root / "videoprism-spatial-image-cache-01/train-cells.npy")
    if args.smoke:
        rows, cells = rows[:8], cells[:8]
    x = torch.from_numpy(np.concatenate((cells[:, 0], cells[:, 1:].mean(1)), axis=1)).double()
    targets = torch.tensor([[list(map(str, vocab)).index(str(row[task]))
                             for task, vocab in CLEVR4_TAXONOMIES.items()] for row in rows])
    initial = torch.load(previous / "initial.pt", weights_only=True, map_location="cpu")
    models, results = {}, {}
    for coefficient in config["coefficients"]:
        name = str(coefficient)
        models[name], results[name] = fit(x, targets, initial, coefficient,
                                         args.output / name, args.smoke)
    if not args.smoke:
        validation_config = read(endpoint / "config.json")
        for p, h in validation_config["inputs"].items():
            if sha(Path(p)) != h:
                raise ValueError("validation input changed")
        validation_result = read(endpoint / "result.json")
        for p, h in validation_result["hashes"].items():
            if sha(endpoint / p) != h:
                raise ValueError("comparison artifact changed")
        corpus = root / "clevr4-sample-seed17-1024-512"
        require_image_splits(lines(root / "clevr4-full-coverage-seed17/train-image-manifest.jsonl"),
                             lines(corpus / "val-image-manifest.jsonl"))
        labels = lines(corpus / "val-labels.jsonl")
        cells = np.load(root / "videoprism-spatial-image-cache-01/val-cells.npy")
        values = torch.from_numpy(np.concatenate((cells[:, 0], cells[:, 1:].mean(1)), 1)).double()
        baseline = read(endpoint / "final-predictions.json")
        for name, model in models.items():
            with torch.no_grad():
                probabilities = model(values).reshape(-1, 4, 10).softmax(-1).numpy()
            predictions = prediction_rows(labels, probabilities)
            write(args.output / name / "validation-predictions.json", predictions)
            results[name]["validation"] = dict(overall=metrics(predictions), tasks={
                task: metrics([r for r in predictions if r["taxonomy"] == task])
                for task in CLEVR4_TAXONOMIES})
            results[name]["paired_vs_zero"] = {
                "overall": paired_image_metrics(baseline, predictions)}
            for task in CLEVR4_TAXONOMIES:
                results[name]["paired_vs_zero"][task] = paired_image_metrics(
                    [r for r in baseline if r["taxonomy"] == task],
                    [r for r in predictions if r["taxonomy"] == task])
        write(args.output / "validation-inputs.json", validation_config)
    if not all(sha(p) == h for p, h in pins.items()):
        raise ValueError("sources changed")
    write(args.output / "result.json", dict(status="PASS_EXECUTION", smoke=args.smoke,
          results=results, source_unchanged=True,
          hashes={str(p.relative_to(args.output)): sha(p)
                  for p in args.output.rglob("*") if p.is_file()}))


if __name__ == "__main__":
    main()
