"""CPU truth-table integration smoke, never a generalization benchmark."""

import argparse
import hashlib
import itertools
import json
import subprocess
import time
from pathlib import Path

import torch

from tiny_omni_decision.classification_json import classification_json
from tiny_omni_decision.fusion_decision import FusionDecisionScorer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    # Exhaustive 2-input parity with an optional criterion bit that reverses it.
    # All eight rows are training fixtures; no held-out accuracy is claimed.
    rows = list(itertools.product((0, 1), repeat=3))
    values = torch.tensor(rows, dtype=torch.float32)
    features = {"signal_a": values[:, :1], "signal_b": values[:, 1:2]}
    available = {name: torch.ones(8, dtype=torch.bool) for name in features}
    options = torch.eye(2).expand(8, -1, -1).clone()
    option_valid = torch.ones(8, 2, dtype=torch.bool)
    criterion = values[:, 2:]
    criterion_present = torch.ones(8, dtype=torch.bool)
    targets = values.long().sum(-1) % 2
    model = FusionDecisionScorer({"signal_a": 1, "signal_b": 1}, 2, 1, width=16)
    config = dict(seed=17, steps=512, optimizer="AdamW", lr=0.01, weight_decay=0.01,
                  device="cpu", width=16, rows=rows, evaluation="training fixtures only",
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                        text=True).strip())
    source_paths = [Path(__file__), Path("src/tiny_omni_decision/fusion_decision.py"),
                    Path("src/tiny_omni_decision/masked_fusion.py"),
                    Path("src/tiny_omni_decision/decision.py")]
    config["source_hashes"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in source_paths}
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    history = []
    start = time.perf_counter()
    for step in range(512):
        optimizer.zero_grad(set_to_none=True)
        logits = model(features, available, options, option_valid, criterion, criterion_present)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
        optimizer.step()
        history.append(dict(step=step + 1, pre_update_loss=loss.item(), grad_norm=norm.item()))
    model.eval()
    with torch.no_grad():
        def predict(f=features, c=criterion, o=options):
            return model(f, available, o, option_valid, c, criterion_present).softmax(-1)
        probabilities = predict()
        reversed_p = predict(o=options.flip(1)).flip(1)
        torch.testing.assert_close(probabilities, reversed_p, rtol=0, atol=0)
        flipped_criterion = predict(c=1 - criterion)
        flipped_a = predict(f={**features, "signal_a": 1 - features["signal_a"]})
        flipped_b = predict(f={**features, "signal_b": 1 - features["signal_b"]})
    path = args.output / "final.pt"
    torch.save(model.state_dict(), path)
    reloaded = FusionDecisionScorer({"signal_a": 1, "signal_b": 1}, 2, 1, width=16)
    reloaded.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        replay = reloaded(features, available, options, option_valid,
                          criterion, criterion_present).softmax(-1)
    torch.testing.assert_close(probabilities, replay, rtol=0, atol=0)
    result = dict(training_fixture_correct=int((probabilities.argmax(-1) == targets).sum()),
                  total=8, parameters=sum(p.numel() for p in model.parameters()),
                  seconds=time.perf_counter() - start, torch_version=torch.__version__,
                  exact_option_permutation=True, exact_reload=True,
                  final_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  intervention_correct={
                      name: int((p.argmax(-1) == 1 - targets).sum())
                      for name, p in {"flip_a": flipped_a, "flip_b": flipped_b,
                                      "flip_criterion": flipped_criterion}.items()},
                  probabilities=probabilities.tolist(),
                  json=[classification_json(p, ["false", "true"],
                                            [{"option_id": "reject", "class_id": "false"},
                                             {"option_id": "accept", "class_id": "true"}])
                        for p in probabilities.tolist()])
    (args.output / "history.json").write_text(json.dumps(history), encoding="utf-8")
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"json", "probabilities"}}))


if __name__ == "__main__":
    main()
