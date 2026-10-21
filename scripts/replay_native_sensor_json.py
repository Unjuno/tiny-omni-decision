"""Replay eight existing development windows through the temporal JSON contract."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from tiny_omni_decision.native_sensor import ACTIVITY_CODES, CHANNELS
from tiny_omni_decision.native_sensor_runtime import StaticSensor, decide_sensor_window
from tiny_omni_decision.observations import ObservationBatch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    manifest_path = root / "harth-subject-windows-01/result.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(r for r in manifest["subjects"] if r["split"] == "validation")
    source = Path(record["path"])
    if sha(source) != record["sha256"]:
        raise ValueError("window hash mismatch")
    trained = root / "sensor-subject-training-01"
    weights = trained / "static/best.pt"
    weight_hash = sha(weights)
    if weight_hash != "5780b04c5022de2f6a085c27032de05c5f86e84a6c638b02a93ed91a9b7db40d":
        raise ValueError("selected weight hash mismatch")
    result = json.loads((trained / "result.json").read_text(encoding="utf-8"))
    reference = trained / f"static/epoch-{result['static']['best_epoch']}.npz"
    model = StaticSensor().eval()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    with np.load(source, allow_pickle=False) as data:
        values = data["values"][:8].copy()
        timestamps = data["timestamps_ms"][:8].copy()
    payload = args.output / "input-only.npz"
    np.savez(payload, values=values, timestamps_ms=timestamps)
    decisions, max_delta, changed = [], 0.0, 0
    with np.load(reference, allow_pickle=False) as saved:
        for i in range(8):
            sample_id = f"{record['subject']}:{i}"
            indices = np.flatnonzero(saved["ids"] == sample_id)
            if len(indices) != 1:
                raise ValueError("reference ID missing or duplicated")
            expected = saved["probabilities"][indices[0]]
            batch = ObservationBatch.model_validate(
                dict(
                    schema_version="1.0",
                    observations=[
                        dict(
                            modality="sensor",
                            source_id=sample_id,
                            payload_ref=str(payload),
                            payload_index=i,
                            payload_sha256=sha(payload),
                            clock_id=record["subject"],
                            timestamps_ms=timestamps[i].tolist(),
                            valid_mask=[True] * 128,
                            channels=[
                                dict(name=c, unit="g", scale=1.0, offset=0.0) for c in CHANNELS
                            ],
                        )
                    ],
                    options=[
                        dict(option_id=f"custom-{c}", attributes={"activity_code": c})
                        for c in ACTIVITY_CODES
                    ],
                )
            )
            decision = decide_sensor_window(
                model, batch, query_time_ms=float(timestamps[i, -1]), clock_id=record["subject"]
            )
            probabilities = np.array([o["probability"] for o in decision["options"]])
            max_delta = max(max_delta, float(np.max(np.abs(probabilities - expected))))
            changed += int(probabilities.argmax() != expected.argmax())
            decisions.append(dict(sample_id=sample_id, decision=decision))
    report = dict(
        status="PASS" if max_delta <= 1e-5 and changed == 0 else "FAIL_PARITY",
        windows=8,
        probability_tolerance=1e-5,
        max_probability_delta=max_delta,
        changed_answers=changed,
        trainable_updates=0,
        device="cpu",
        parameters=sum(p.numel() for p in model.parameters()),
        weights_sha256=weight_hash,
        reference_sha256=sha(reference),
        source_sha256=record["sha256"],
        manifest_sha256=sha(manifest_path),
        sources_unchanged=sha(weights) == weight_hash and sha(source) == record["sha256"],
        decisions=decisions,
        limitation="Central-label replay only; no new quality or instantaneous recognition claim.",
    )
    (args.output / "result.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "decisions"}))
    if report["status"] != "PASS" or not report["sources_unchanged"]:
        raise RuntimeError("replay gate failed")


if __name__ == "__main__":
    main()
