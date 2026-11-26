"""Exercise the real aligned-cache JSON adapter on the fixed image diagnostic."""

from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.fit_native_feature_inat_diagnostic import (
    FEATURE_DIR,
    load_data,
    sha256,
)
from tiny_omni_decision.native_feature_decision import (
    EnumOptionField,
    NativeFeatureDecisionScorer,
    TypedOptionSchema,
    decide_cached_native_feature,
)
from tiny_omni_decision.observations import NativeOption, ObservationBatch

ROOT = Path(
    r"C:\CodexArtifacts\pretrained-reuse-r2\native-feature-decision-inat-20261031-attempt-01"
)
RUNTIME_ROOT = ROOT / "runtime-json-replay"


def npz_bytes(values: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        values=values.astype(np.float32, copy=False)[None, :],
        timestamps_ms=np.asarray([0.0], dtype=np.float64),
        valid_mask=np.asarray([True], dtype=np.bool_),
    )
    return buffer.getvalue()


def main() -> None:
    if RUNTIME_ROOT.exists():
        raise FileExistsError(f"runtime replay output must be fresh: {RUNTIME_ROOT}")
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=False)
    cache_root = RUNTIME_ROOT / "feature-cache"
    cache_root.mkdir()
    features, rows, classes, input_hashes = load_data()
    valid_indices = [i for i, row in enumerate(rows) if row["split"] == "validation"]
    checkpoint = torch.load(ROOT / "final.pt", map_location="cpu", weights_only=True)
    model = NativeFeatureDecisionScorer(768, len(classes), hidden_width=64)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    schema = TypedOptionSchema(enums=(EnumOptionField("taxon_id", tuple(classes)),))
    class_features = schema.encode(
        [
            NativeOption(option_id=f"taxon-{value}", attributes={"taxon_id": value})
            for value in classes
        ]
    )
    del class_features  # The production adapter must construct candidates from request JSON.
    option_records = [
        NativeOption(option_id=f"taxon-{value}", attributes={"taxon_id": value})
        for value in classes
    ]
    direct = [
        json.loads(line)
        for line in (ROOT / "validation-predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    runtime_rows = []
    cache_rows = []
    started = time.perf_counter()
    for local_index, row_index in enumerate(valid_indices):
        source_row = rows[row_index]
        cache_path = cache_root / f"feature-{local_index:04d}.npz"
        payload = npz_bytes(features[row_index])
        cache_path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        source_id = f"image:{source_row['sample_id']}"
        batch = ObservationBatch.model_validate(
            {
                "schema_version": "1.0",
                "observations": [
                    {
                        "modality": "image",
                        "source_id": source_id,
                        "payload_ref": str(cache_path),
                        "payload_sha256": digest,
                        "clock_id": source_id,
                        "timestamps_ms": [0.0],
                        "valid_mask": [True],
                    }
                ],
                "options": [option.model_dump() for option in option_records],
            }
        )
        decision = decide_cached_native_feature(
            model,
            batch,
            modality="image",
            query_time_ms=0.0,
            max_age_ms=0.0,
            option_schema=schema,
            class_ids=classes,
            feature_mean=checkpoint["mean"],
            feature_scale=checkpoint["scale"],
        )
        if decision["status"] != "selected":
            raise ValueError("fresh image feature unexpectedly abstained")
        probabilities = {
            class_id: option["probability"]
            for class_id, option in zip(classes, decision["options"], strict=True)
        }
        runtime_rows.append(
            {
                "sample_id": source_row["sample_id"],
                "observation_id": source_row["observation_id"],
                "target_taxon_id": str(source_row["taxon_id"]),
                "decision_json": decision,
                "probabilities_by_taxon_id": probabilities,
            }
        )
        cache_rows.append(
            {
                "sample_id": source_row["sample_id"],
                "cache_file": cache_path.name,
                "cache_sha256": digest,
            }
        )

    elapsed = time.perf_counter() - started
    new_probs = np.asarray(
        [[r["probabilities_by_taxon_id"][c] for c in classes] for r in runtime_rows]
    )
    old_probs = np.asarray(
        [[direct[i]["probabilities_by_taxon_id"][c] for c in classes] for i in range(len(direct))]
    )
    max_abs = float(np.max(np.abs(new_probs - old_probs)))
    runtime_pred = new_probs.argmax(axis=1)
    direct_pred = old_probs.argmax(axis=1)
    if max_abs > 2e-6 or not np.array_equal(runtime_pred, direct_pred):
        raise ValueError(f"runtime replay differs from fixed readout outputs: {max_abs}")
    predictions_path = RUNTIME_ROOT / "predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in runtime_rows),
        encoding="utf-8",
    )
    caches_path = RUNTIME_ROOT / "cache-manifest.json"
    caches_path.write_text(json.dumps(cache_rows, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "status": "PASS_ALIGNED_CACHE_DECISION_JSON_REPLAY",
        "count": len(runtime_rows),
        "modality": "image",
        "criterion_or_text_encoder": False,
        "mean_cache_write_read_decide_seconds_per_example": elapsed / len(runtime_rows),
        "max_abs_probability_delta_vs_training_prediction": max_abs,
        "argmax_identity_match": True,
        "feature_mean_sha256": hashlib.sha256(checkpoint["mean"].numpy().tobytes()).hexdigest(),
        "feature_scale_sha256": hashlib.sha256(checkpoint["scale"].numpy().tobytes()).hexdigest(),
        "checkpoint_sha256": sha256(ROOT / "final.pt"),
        "prediction_sha256": sha256(predictions_path),
        "cache_manifest_sha256": sha256(caches_path),
        "feature_inputs": input_hashes,
        "cache_feature_rows_sha256": sha256(FEATURE_DIR / "feature-rows.jsonl"),
        "interpretation_limit": (
            "closed eight-taxon post-selection development image task; runtime parity only, "
            "not broad or blind quality evidence"
        ),
    }
    result_path = RUNTIME_ROOT / "result.json"
    print(json.dumps(receipt, indent=2))

    result_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
