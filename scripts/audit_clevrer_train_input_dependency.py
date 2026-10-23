"""Read-only TRAIN input-dependency audit for frozen CLEVRER AV LSTM checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from scripts.train_clevrer_synthetic_av_temporal_decision import _load_inputs, _metrics
from tiny_omni_decision.temporal_decision import CausalGatedTemporalDecision


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def pair_donor_rows(metadata) -> np.ndarray:
    """Map each row to the same target slot in another scene's pair."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, pair in enumerate(metadata["pair_ids"]):
        groups[str(pair)].append(index)
    pairs = []
    pair_scenes = []
    for pair in sorted(groups):
        rows = groups[pair]
        if (len(rows) != 2 or {int(metadata["targets"][i]) for i in rows} != {0, 1}):
            raise ValueError("every event pair must contain both target indices")
        scenes = {str(metadata["scene_ids"][i]) for i in rows}
        if len(scenes) != 1:
            raise ValueError("both rows in each pair must use the same scene")
        pairs.append(sorted(rows, key=lambda i: int(metadata["targets"][i])))
        pair_scenes.append(scenes.pop())
    if sorted(np.asarray(pairs).ravel().tolist()) != list(range(len(metadata["targets"]))):
        raise ValueError("event pairs must partition train rows exactly once")
    if len(set(pair_scenes)) < 2:
        raise ValueError("different-scene donors require at least two scenes")
    count = len(pairs)
    offset = max(1, count // 2)
    while offset < count and any(pair_scenes[i] == pair_scenes[(i + offset) % count]
                                 for i in range(count)):
        offset += 1
    if offset == count:
        raise ValueError("no fixed cyclic shift maps every pair to a different scene")
    donor = np.empty(len(metadata["targets"]), dtype=np.int64)
    for i, pair in enumerate(pairs):
        other = pairs[(i + offset) % count]
        for target in (0, 1):
            donor[pair[target]] = other[target]
    if sorted(donor.tolist()) != list(range(len(donor))):
        raise ValueError("cross-scene donor rows must form a bijection")
    if any(metadata["scene_ids"][i] == metadata["scene_ids"][donor[i]]
           for i in range(len(donor))):
        raise ValueError("donor scene overlaps its recipient")
    return donor


def apply_control(values, name: str, donors: torch.Tensor):
    """Return a shallow input view with exactly one named stream modified."""
    changed = dict(values)
    if name in {"zero_audio", "zero_video"}:
        key = name.removeprefix("zero_")
        changed[key] = torch.zeros_like(values[key])
    elif name in {"shuffle_audio", "shuffle_video"}:
        key = name.removeprefix("shuffle_")
        if donors.ndim != 1 or len(donors) != len(values[key]):
            raise ValueError("one donor row per example required")
        changed[key] = values[key].index_select(0, donors.to(values[key].device))
    elif name == "reverse_video":
        changed["video"] = values["video"].flip(1)
    elif name == "reverse_audio":
        changed["audio"] = values["audio"].flip(1)
    else:
        raise ValueError("unknown input control")
    return changed


def _forward(model, values, *, batch_size=64):
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(values["targets"]), batch_size):
            stop = min(start + batch_size, len(values["targets"]))
            sl = slice(start, stop)
            logits = model(values["video"][sl], values["audio"][sl], values["text"][sl]).logits
            output.append(logits[:, -1].float().cpu())
    return torch.cat(output)


def _model(weights_path: Path, name: str) -> CausalGatedTemporalDecision:
    mode = name.removesuffix("_lstm")
    model = CausalGatedTemporalDecision(video_size=768, audio_size=1,
                                        text_size=384, hidden_size=32,
                                        modality_mode=mode)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if state["candidate"] != name:
        raise ValueError("checkpoint candidate differs")
    model.load_state_dict(state["model_state"], strict=True)
    return model.eval()


def _prediction_rows(rows, logits, targets):
    probabilities = logits.softmax(-1)
    indices = logits.argmax(-1)
    return [{"sample_id": row["id"], "pair_id": row["pair_id"],
             "scene_group_id": row["scene_group_id"], "target": int(target),
             "prediction": int(prediction),
             "probabilities": [float(p) for p in probability]}
            for row, target, prediction, probability in zip(
                rows, targets.tolist(), indices.tolist(), probabilities.tolist(), strict=True)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--feature-report", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("new output directory required")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    train_rows, _, all_tensors, input_report = _load_inputs(
        config, args.feature_dir, args.feature_report, args.sample_dir)
    train_count = len(train_rows)
    values = {k: v[:train_count].clone() for k, v in all_tensors.items()}
    del all_tensors
    target = values["targets"]
    metadata = {"pair_ids": np.asarray([row["pair_id"] for row in train_rows]),
                "scene_ids": np.asarray([row["scene_group_id"] for row in train_rows]),
                "targets": target.numpy()}
    donors_array = pair_donor_rows(metadata)
    donors = torch.from_numpy(donors_array)
    protected = [args.config, args.feature_report,
                 args.feature_dir / "decision-features.npz",
                 args.sample_dir / "train.jsonl", args.sample_dir / "validation.jsonl",
                 args.run_dir / "effective-config.json",
                 args.run_dir / "results.json"] + [
                     args.run_dir / f"best-{name}.pt" for name in
                     ["joint_lstm", "audio_only_lstm", "video_only_lstm"]]
    protected_hashes = {str(p.resolve()): sha(p) for p in protected}
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    donor_hash = hashlib.sha256(donors_array.astype("<i8").tobytes()).hexdigest()
    metadata_out = {"status": "running", "scope": "official train split only; frozen checkpoints",
                    "utc_start": datetime.now(UTC).isoformat(), "device": "cpu",
                    "torch": torch.__version__, "python": platform.python_version(),
                    "cpu_threads": 2, "optimizer_updates": 0,
                    "sealed_audit_opened": False, "validation_forwarded_or_scored": False,
                    "feature_container_includes_historical_validation": True,
                    "selected_indices": "TRAIN rows only; existing validation was not evaluated",
                    "train_rows": train_count,
                    "train_scenes": len(set(metadata["scene_ids"])),
                    "train_pairs": len(donors_array) // 2,
                    "model_order": ["joint_lstm", "audio_only_lstm", "video_only_lstm"],
                    "controls": ["zero_audio", "zero_video", "shuffle_audio",
                                 "shuffle_video", "reverse_audio", "reverse_video"],
                    "shuffle_rule": (
                        "same-target-slot pairs; fixed cyclic offset chosen from metadata only"
                    ),
                    "scene_donor_index_sha256": donor_hash,
                    "source_commit": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                        text=True).strip(),
                    "runner_sha256": sha(Path(__file__).resolve()),
                    "frozen_input_hashes": protected_hashes,
                    "training_input_integrity": input_report["split"],
                    "command_arguments": {k: str(v.resolve()) for k, v in vars(args).items()}}
    write_json(output / "run-metadata.json", metadata_out)
    started = time.perf_counter()
    result = {"condition": "original", "models": {}}
    for name in ["joint_lstm", "audio_only_lstm", "video_only_lstm"]:
        checkpoint = args.run_dir / f"best-{name}.pt"
        model = _model(checkpoint, name)
        logits = _forward(model, values)
        result["models"][name] = {"metrics": _metrics(logits, target)}
        rows = _prediction_rows(train_rows, logits, target)
        (output / f"predictions-{name}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        if name == "joint_lstm":
            for control in metadata_out["controls"]:
                changed = apply_control(values, control, donors)
                altered = _forward(model, changed)
                result["models"][name][control] = {
                    "metrics": _metrics(altered, target),
                    "prediction_change_fraction": float(
                        (altered.argmax(-1) != logits.argmax(-1)).float().mean()),
                    "max_probability_change": float(
                        (altered.softmax(-1) - logits.softmax(-1)).abs().max()),
                }
                rows = _prediction_rows(train_rows, altered, target)
                (output / f"predictions-joint_lstm-{control}.jsonl").write_text(
                    "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                    encoding="utf-8")
    if protected_hashes != {str(p.resolve()): sha(p) for p in protected}:
        raise RuntimeError("protected source or checkpoint changed")
    write_json(output / "results.json", result)
    metadata_out.update(status="complete", utc_end=datetime.now(UTC).isoformat(),
                        wall_seconds=time.perf_counter() - started,
                        protected_hashes_unchanged=True,
                        prediction_output_hashes={p.name: sha(p) for p in output.glob("*.jsonl")})
    write_json(output / "run-metadata.json", metadata_out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
