"""Check end-to-end CPU runtime parity against saved internal Clotho predictions."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np

from scripts.predict_clotho_aqa import ClothoAqaPredictor, sha256

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
OUTPUT = ROOT / "clotho-aqa-runtime-parity-20261126-01"
EXPECTED_VALIDATION_IDS_SHA256 = (
    "273197786d3eba4b88b8394d5ac807e08243ad74f1912858123ee7a6db2868a6"
)
MAX_ABS_ERROR = 1e-4
MAX_ORDER_ERROR = 1e-6


def run(output: Path = OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    split = ROOT / "clotho-aqa-development-split-01" / "validation.json"
    media = ROOT / "clotho-aqa-development-media-02"
    model_dir = ROOT / "clotho-aqa-joint-control-01"
    ids_path = model_dir / "validation_ids.json"
    probabilities_path = model_dir / "validation_probabilities.npy"
    if sha256(ids_path) != EXPECTED_VALIDATION_IDS_SHA256:
        raise ValueError("internal validation ID order differs from frozen reference")
    rows = json.loads(split.read_text(encoding="utf-8"))
    fixed_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    if [[row["source_id"], row["question"]] for row in rows] != fixed_ids:
        raise ValueError("validation rows differ from the saved prediction order")
    saved_probabilities = np.load(probabilities_path, allow_pickle=False)
    if saved_probabilities.shape != (len(rows),) or not np.isfinite(saved_probabilities).all():
        raise ValueError("saved internal probabilities are invalid")

    selected, seen_sources = [], set()
    for index, row in enumerate(rows):
        if row["source_id"] not in seen_sources:
            seen_sources.add(row["source_id"])
            selected.append(index)
        if len(selected) == 6:
            break
    if len(selected) != 6:
        raise ValueError("fewer than six distinct validation audio sources")

    started = time.perf_counter()
    predictor = ClothoAqaPredictor()
    model_load_seconds = time.perf_counter() - started
    records, prediction_seconds, order_seconds = [], 0.0, 0.0
    for index in selected:
        row = rows[index]
        audio_path = media / f"{row['source_id']}.wav"
        normal = time.perf_counter()
        normal_json = json.loads(
            predictor.predict(
                audio_path,
                row["question"],
                ["no", "yes"],
                ["No", "Yes"],
                request_id=f"parity-{index}",
            )
        )
        prediction_seconds += time.perf_counter() - normal
        reversed_order = time.perf_counter()
        reversed_json = json.loads(
            predictor.predict(
                audio_path,
                row["question"],
                ["yes", "no"],
                ["Yes", "No"],
                request_id=f"reversed-{index}",
            )
        )
        order_seconds += time.perf_counter() - reversed_order
        p_yes = normal_json["options"][1]["probability"]
        delta = abs(p_yes - float(saved_probabilities[index]))
        order_delta = max(
            abs(reversed_json["options"][0]["probability"] - p_yes),
            abs(
                reversed_json["options"][1]["probability"]
                - normal_json["options"][0]["probability"]
            ),
        )
        if normal_json["decision_type"] != "yes_no" or delta > MAX_ABS_ERROR:
            raise ValueError(f"saved prediction parity failed at row {index}: {delta}")
        if order_delta > MAX_ORDER_ERROR:
            raise ValueError(f"option-order parity failed at row {index}: {order_delta}")
        records.append(
            {
                "row_index": index,
                "source_id": row["source_id"],
                "audio_sha256": sha256(audio_path),
                "saved_p_yes": float(saved_probabilities[index]),
                "runtime_p_yes": float(p_yes),
                "max_abs_probability_delta": delta,
                "max_option_order_delta": order_delta,
            }
        )

    script_hash = sha256(Path(__file__))
    files = {
        "MiniLM_model.safetensors": ROOT / "models/text-minilm-en/model.safetensors",
        "EfficientAT_dymn04_as.pt": Path(
            "C:/CodexArtifacts/pretrained-reuse-r1/efficientat-audit-v1/dymn04_as.pt"
        ),
        "Decision_best.pt": model_dir / "best.pt",
        "normalization.npz": model_dir / "normalization.npz",
    }
    file_sizes = {name: path.stat().st_size for name, path in files.items()}
    expected_sizes = {
        "MiniLM_model.safetensors": 69_569_488,
        "EfficientAT_dymn04_as.pt": 8_086_973,
        "Decision_best.pt": 199_544,
        "normalization.npz": 7_182,
    }
    if file_sizes != expected_sizes:
        raise ValueError("frozen inference file sizes differ from recorded inventory")
    report = {
        "status": "PASS",
        "scope": (
            "runtime reload and feature-to-JSON parity only; "
            "no quality metrics or checkpoint selection"
        ),
        "conditions": {
            "checkpoint": "fixed Clotho audio+question best.pt; hash verified by predictor",
            "validation_ids_sha256": EXPECTED_VALIDATION_IDS_SHA256,
            "rows": "first six rows with distinct source IDs in frozen internal validation order",
            "option_orders": [["No", "Yes"], ["Yes", "No"]],
            "device": "cpu",
        },
        "max_abs_probability_error": max(row["max_abs_probability_delta"] for row in records),
        "max_option_order_error": max(row["max_option_order_delta"] for row in records),
        "model_load_seconds": model_load_seconds,
        "normal_order_mean_seconds": prediction_seconds / len(records),
        "reversed_order_mean_seconds": order_seconds / len(records),
        "total_seconds": time.perf_counter() - started,
        "runtime": {
            "python": platform.python_version(),
            "torch": predictor.torch.__version__,
            "torchaudio": predictor.torchaudio.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "gpu_used": False,
        },
        "encoder_parameters": {
            "MiniLM": 17_389_824,
            "EfficientAT": 1_961_239,
            "Decision_readout": 49_409,
            "total_resident_parameters": 19_400_472,
        },
        "unquantized_checkpoint_bytes": {
            **file_sizes,
            "weight_and_normalization_subtotal": sum(file_sizes.values()),
            "excludes": "tokenizer files, source code, Python runtime and dependencies",
        },
        "predictor_script_sha256": sha256(Path(__file__).with_name("predict_clotho_aqa.py")),
        "parity_harness_sha256": script_hash,
        "rows": records,
        "limitation": (
            "This fixed binary audio-question path is not a four-modality Omni model "
            "or a task-quality pass."
        ),
    }
    output.mkdir(parents=False, exist_ok=False)
    report_path = output / "result.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    integrity = {
        "result_sha256": sha256(report_path),
        "parity_harness_sha256": script_hash,
        "predictor_sha256": sha256(Path(__file__).with_name("predict_clotho_aqa.py")),
    }
    (output / "integrity.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8"
    )
    report["integrity"] = integrity
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
