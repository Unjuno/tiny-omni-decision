"""Fresh-process CPU reload audit for the fixed Greatest Hits AV readouts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from scripts.train_greatest_hits_av_match_readouts import (
    _load_feature_map,
    _read_corpus,
    probability_metrics,
    sha256_file,
)


def compare_probabilities(
    expected: list[list[float]], actual: list[list[float]], tolerance: float = 1e-7
) -> dict[str, Any]:
    if len(expected) != len(actual) or not expected:
        raise ValueError("nonempty prediction lists with equal lengths required")
    max_delta = max(
        abs(float(left) - float(right))
        for expected_row, actual_row in zip(expected, actual, strict=True)
        for left, right in zip(expected_row, actual_row, strict=True)
    )
    if any(len(row) != 2 for row in expected + actual):
        raise ValueError("binary probability rows required")
    if max_delta > tolerance:
        raise ValueError(f"reloaded probabilities exceed tolerance: {max_delta}")
    return {"max_abs_delta": max_delta, "tolerance": tolerance, "within_tolerance": True}


def verify_run(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite audit directory: {output_dir}")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    config_path = run_dir / "effective-config.json"
    result_path = run_dir / "result.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "PASS_COMPLETED_ALL_FIXED_READOUTS":
        raise ValueError("completed fixed readout result is required")
    source_config_path = Path(result["config_path"])
    if not source_config_path.is_absolute():
        source_config_path = Path(__file__).resolve().parents[1] / source_config_path
    if sha256_file(source_config_path) != result.get("config_sha256"):
        raise ValueError("input config hash differs from run receipt")
    if json.loads(source_config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("effective config values differ from the hashed input config")
    predictions_path = run_dir / "validation-predictions.jsonl"
    if sha256_file(predictions_path) != result.get("predictions_sha256"):
        raise ValueError("validation prediction hash differs from run receipt")

    paths = config["dataset"]
    corpora = _read_corpus(Path(paths["preflight"]))
    audio_features = _load_feature_map(Path(paths["audio_features"]), 220)
    video_features = _load_feature_map(Path(paths["video_features"]), 220)
    saved = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
    expected_by_condition = {}
    for condition in ("audio_only", "video_only", "joint_av"):
        rows = [row for row in saved if row.get("condition") == condition]
        if len(rows) != len(corpora["validation"]):
            raise ValueError(f"wrong saved prediction count for {condition}")
        if [row["id"] for row in rows] != [row["id"] for row in corpora["validation"]]:
            raise ValueError(f"saved prediction order differs for {condition}")
        expected_by_condition[condition] = rows
    if len(saved) != sum(len(rows) for rows in expected_by_condition.values()):
        raise ValueError("unexpected or duplicate prediction conditions")

    class LinearHead(nn.Module):
        def __init__(self, input_size: int):
            super().__init__()
            self.linear = nn.Linear(input_size, 2)

        def forward(self, audio, video):
            value = audio if audio is not None else video
            return self.linear(value)

    class PairedCosine(nn.Module):
        def __init__(self):
            super().__init__()
            self.audio_projection = nn.Linear(384, 64, bias=False)
            self.video_projection = nn.Linear(768, 64, bias=False)

        def forward(self, audio, video):
            left = F.normalize(self.audio_projection(audio), dim=-1)
            right = F.normalize(self.video_projection(video), dim=-1)
            score = (left * right).sum(dim=-1) / 0.1
            return torch.stack((-score, score), dim=-1)

    rows = corpora["validation"]
    audio = torch.tensor(
        np.stack([audio_features[row["audio_source_group"]] for row in rows]),
        dtype=torch.float32,
    )
    video = torch.tensor(
        np.stack([video_features[row["video_source_group"]] for row in rows]),
        dtype=torch.float32,
    )
    targets = [int(row["target"]) for row in rows]
    audits = {}
    for condition in ("audio_only", "video_only", "joint_av"):
        path = run_dir / f"{condition}-best.pt"
        expected_hash = result["conditions"][condition]["checkpoint_sha256"]["best"]
        if sha256_file(path) != expected_hash:
            raise ValueError(f"best checkpoint hash differs for {condition}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        selected_epoch = int(result["conditions"][condition]["best_epoch"])
        if checkpoint.get("epoch") != selected_epoch:
            raise ValueError(f"checkpoint epoch differs from selected epoch for {condition}")
        if condition == "audio_only":
            model = LinearHead(384)
            logits = model(audio, None)
        elif condition == "video_only":
            model = LinearHead(768)
            logits = model(None, video)
        else:
            model = PairedCosine()
            logits = model(audio, video)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        with torch.inference_mode():
            if condition == "audio_only":
                logits = model(audio, None)
            elif condition == "video_only":
                logits = model(None, video)
            else:
                logits = model(audio, video)
            temperature = float(result["conditions"][condition]["temperature"])
            probabilities = torch.softmax(logits / temperature, dim=1).tolist()
        stored_rows = expected_by_condition[condition]
        stored_probabilities = [row["probabilities"] for row in stored_rows]
        parity = compare_probabilities(stored_probabilities, probabilities)
        stored_predictions = [int(row["prediction"]) for row in stored_rows]
        recomputed_predictions = [int(max(range(2), key=row.__getitem__)) for row in probabilities]
        if stored_predictions != recomputed_predictions:
            raise ValueError(f"reloaded argmax differs for {condition}")
        metrics = probability_metrics(targets, probabilities)
        audits[condition] = {
            "selected_epoch": selected_epoch,
            "checkpoint_sha256": sha256_file(path),
            "prediction_count": len(probabilities),
            "probability_parity": parity,
            "argmax_exact": True,
            "metrics": metrics,
        }

    audit = {
        "status": "PASS_FRESH_PROCESS_CPU_RELOAD_AUDIT",
        "verifier_source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "verifier_sha256": sha256_file(Path(__file__)),
        "training": False,
        "optimizer_updates": 0,
        "gpu_used": False,
        "sealed_audit_opened": False,
        "run_result_sha256": sha256_file(result_path),
        "input_config_sha256": sha256_file(source_config_path),
        "effective_config_sha256": sha256_file(config_path),
        "validation_predictions_sha256": sha256_file(predictions_path),
        "conditions": audits,
    }
    output_dir.mkdir(parents=True)
    audit_path = output_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (output_dir / "audit.sha256").write_text(sha256_file(audit_path) + "\n", encoding="ascii")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_run(args.run_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
