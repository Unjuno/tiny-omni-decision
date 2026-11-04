"""Verify and report the completed training-only diagnostic, without validation."""

import json
from pathlib import Path

import numpy as np
import torch

from scripts.fit_shared_affine_training import quality
from scripts.train_dinov2_image_probe import sha, write
from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2/shared-affine-training-fit-01")
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    identity = json.loads((root / "identity.json").read_text(encoding="utf-8"))
    for name, digest in result["hashes"].items():
        assert sha(root / name) == digest
    for name, digest in config["inputs"].items():
        assert sha(Path(name)) == digest
    assert sha(Path("scripts/fit_shared_affine_training.py")) == config["runner_sha256"]
    assert not config["smoke"] and not result["validation_loaded"]
    assert result["reload_exact"] and result["source_unchanged"]
    history = result["history"]
    assert [h["optimizer_iterations"] for h in history] == list(range(0, 201, 10))
    assert result["best_call"] == min(history, key=lambda h: h["loss"])["call"] == 20
    assert result["reason"] == "budget_completed" and history[-1]["gradient_inf"] > 1e-7
    labels_path = next(Path(p) for p in config["inputs"] if p.endswith("train-labels.jsonl"))
    rows = [json.loads(line) for line in labels_path.read_text(encoding="utf-8").splitlines()]
    targets = torch.tensor([[list(map(str, vocab)).index(str(row[task]))
                             for task, vocab in CLEVR4_TAXONOMIES.items()] for row in rows])
    for name, expected in (("initial", history[0]), ("best", history[20]), ("final", history[20])):
        logits = torch.from_numpy(np.load(root / f"{name}-logits.npy", allow_pickle=False))
        actual = quality(logits, targets)
        for key in actual["overall"]:
            np.testing.assert_allclose(actual["overall"][key], expected["metrics"]["overall"][key],
                                       atol=1e-12, rtol=1e-12)
        for task in CLEVR4_TAXONOMIES:
            for key in actual["tasks"][task]:
                np.testing.assert_allclose(actual["tasks"][task][key],
                                           expected["metrics"]["tasks"][task][key],
                                           atol=1e-12, rtol=1e-12)
    # Preserve the exact dirty-tree implementation used; source commit alone is insufficient.
    archive = root / "executed-source"
    archive.mkdir(exist_ok=True)
    sources = ["scripts/fit_shared_affine_training.py",
               "src/tiny_omni_decision/count_diagnostic.py",
               "src/tiny_omni_decision/global_readout.py",
               "src/tiny_omni_decision/bilinear_readout.py",
               "docs/pretrained_reuse/R2_AFFINE_FIT_PROTOCOL_20261012.md"]
    for name in sources:
        target = archive / Path(name).name
        if target.exists():
            assert target.read_bytes() == Path(name).read_bytes()
        else:
            target.write_bytes(Path(name).read_bytes())
    report = dict(config=config, result=result, identity=identity,
                  executed_source={name: sha(Path(name)) for name in sources},
                  artifact_bytes={p.name: p.stat().st_size for p in root.glob("*.pt")})
    base = Path("docs/pretrained_reuse/R2_AFFINE_FIT_RESULTS_20261012")
    write(base.with_suffix(".json"), report)
    lines = ["# Training-only affine fit (2026-10-12)", "",
             "## Scope and execution", "",
             "One bounded CPU diagnostic completed: 8,424 original training images, all 33,696",
             "color/texture/count/shape questions, equal task weight, unchanged cropped frozen",
             "features and fixed 40 candidates. No validation or sealed-audit data were loaded.",
             "The original selected bilinear checkpoint initializes the affine weights; this",
             "is a training diagnostic warm start, not a new independent quality experiment.", "",
             "Run: `python -m scripts.fit_shared_affine_training --output " + str(root) + "`.",
             "A separate `--smoke` process used eight images and two inner iterations first.",
             "The full run used float64, two CPU threads, unregularized full-batch L-BFGS,",
             "200 inner iterations / 229 closure evaluations, no extra seeds or extension.",
             f"Elapsed {result['seconds']:.3f}s including identity check; optimization/history",
             f"after initialization took {history[-1]['seconds'] - history[0]['seconds']:.3f}s.",
             "No GPU compute or cloud cost. Peak process RAM was not measured.", "",
             "## Training results only", "",
             "| Task | Initial Accuracy | Final Accuracy | Initial NLL | Final NLL |"
             " Final Brier | Final ECE |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for task in ["overall", *CLEVR4_TAXONOMIES]:
        a = (history[0]["metrics"]["overall"] if task == "overall"
             else history[0]["metrics"]["tasks"][task])
        b = (history[-1]["metrics"]["overall"] if task == "overall"
             else history[-1]["metrics"]["tasks"][task])
        lines.append(f"| {task} | {a['accuracy']:.4%} | {b['accuracy']:.4%} | {a['nll']:.6f} | "
                     f"{b['nll']:.6f} | {b['brier']:.6f} | {b['ece_15_bins']:.6f} |")
    lines += ["", "| Inner iterations | Training CE | Count Accuracy | Gradient infinity norm |",
              "|---:|---:|---:|---:|"]
    for h in history[::5]:
        lines.append(f"| {h['optimizer_iterations']} | {h['loss']:.6f} | "
                     f"{h['metrics']['tasks']['count']['accuracy']:.4%} | "
                     f"{h['gradient_inf']:.8f} |")
    lines += ["", "## Interpretation and limits", "",
              "The same frozen representation supports a lower training error. Existing weak",
              "counting therefore cannot be attributed solely to missing encoder information.",
              "Optimizer, parameterization, precision and regularization all changed; this is",
              "not a pure learning-budget comparison. Projected candidates have full row rank",
              "40, so unrestricted affine scores are representable in the original fixed-candidate",
              "bilinear family. This statement does not extend to unseen semantic candidates.", "",
              "Loss was still decreasing and the final gradient infinity norm is 0.004755,",
              "above 1e-7. Budget completion is not convergence or a representation ceiling.",
              "Overall training ECE worsened 0.005212 -> 0.020295; count ECE worsened",
              "0.017059 -> 0.066527. Increased training accuracy is not calibrated quality.",
              "No generalization gain, absence of overfitting, or Omni 95–98% success is claimed.",
              "Keep the original runtime reference. Do not promote this closed-taxonomy artifact.",
              "Any follow-up must predefine a separate regularized validation protocol; the",
              "completed diagnostic does not authorize an unbounded training-budget extension.", "",
              "## Integrity and artifacts", "",
              f"Initial all-question logit difference: {identity['max_logit_delta']:.3g}, "
              "within 1e-10.",
              "61,480 trainable affine parameters; exact best reload; best and final predictions",
              "are identical. Original input hashes remain unchanged. Initialized/best/final",
              "weights and logits, every closure loss/gradient, and full curves are preserved.",
              f"Artifacts: `{root}`. Config, source/input/artifact hashes and all metrics are",
              "in the adjacent JSON report. Run source was e077068 plus the archived uncommitted",
              "implementation; the runner SHA and executed-source archive identify it exactly.",
              "The files contain no optimizer state and are not an exact-resume package.", ""]
    base.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
