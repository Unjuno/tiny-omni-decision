"""Reload and replay the fixed-budget native-feature iNaturalist diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts.fit_native_feature_inat_diagnostic import (
    BASELINE_DIR,
    EnumOptionField,
    NativeFeatureDecisionScorer,
    TypedOptionSchema,
    load_data,
    sha256,
)

ROOT = Path(
    r"C:\CodexArtifacts\pretrained-reuse-r2\native-feature-decision-inat-20261031-attempt-01"
)


def main() -> None:
    features, rows, classes, hashes = load_data()
    valid_indices = [i for i, row in enumerate(rows) if row["split"] == "validation"]
    checkpoint_path = ROOT / "final.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = NativeFeatureDecisionScorer(768, len(classes), hidden_width=64)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    x = torch.from_numpy(features[valid_indices].copy())
    x = (x - checkpoint["mean"]) / checkpoint["scale"]
    schema = TypedOptionSchema(enums=(EnumOptionField("taxon_id", tuple(classes)),))
    option_rows = [
        type("TypedOption", (), {"attributes": {"taxon_id": value}})() for value in classes
    ]
    options = schema.encode(option_rows).unsqueeze(0).expand(len(x), -1, -1).contiguous()
    mask = torch.ones(options.shape[:2], dtype=torch.bool)
    with torch.inference_mode():
        logits = torch.cat(
            [
                model(
                    x[start : start + 128], options[start : start + 128], mask[start : start + 128]
                )
                for start in range(0, len(x), 128)
            ]
        )
        probabilities = torch.softmax(logits.float(), dim=-1).numpy()
    saved_rows = [
        json.loads(line)
        for line in (ROOT / "validation-predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if len(saved_rows) != len(valid_indices):
        raise ValueError("saved prediction count mismatch")
    expected = np.asarray(
        [[row["probabilities_by_taxon_id"][class_id] for class_id in classes] for row in saved_rows]
    )
    max_abs = float(np.max(np.abs(probabilities - expected)))
    predictions = probabilities.argmax(axis=1)
    stored_predictions = np.asarray(
        [
            int(np.argmax([row["probabilities_by_taxon_id"][c] for c in classes]))
            for row in saved_rows
        ]
    )
    if max_abs > 1e-6 or not np.array_equal(predictions, stored_predictions):
        raise ValueError(f"checkpoint replay mismatch: max_abs={max_abs}")
    receipt = {
        "status": "PASS_FINAL_CHECKPOINT_REPLAY",
        "count": len(saved_rows),
        "max_abs_probability_delta": max_abs,
        "prediction_identity_match": True,
        "checkpoint_sha256": sha256(checkpoint_path),
        "saved_predictions_sha256": sha256(ROOT / "validation-predictions.jsonl"),
        "feature_hashes": hashes,
        "baseline_prediction_sha256": sha256(BASELINE_DIR / "validation-predictions.jsonl"),
    }
    output = ROOT / "replay.json"
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
