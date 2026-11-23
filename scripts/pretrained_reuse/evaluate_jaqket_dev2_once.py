"""One-shot inference for frozen JAQKET Dev2 readouts; never trains or selects."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from scripts.pretrained_reuse.build_jaqket_dev1_split import _normalized
from scripts.pretrained_reuse.train_jaqket_frozen_readout import (
    WEIGHT_HASHES,
    SharedOptionScorer,
    _targets,
    encode_split,
    metrics,
    read_jsonl,
    sha256_file,
)

FROZEN_READOUTS = {
    "ruri-pair-readout": {
        "candidate": "ruri-v3-30m",
        "hash": "b9468418c36e7597c8cfede62fe5989fa7f87707a905adfb2e60106068b3287b",
        "option_only": False,
    },
    "ruri-option-only": {
        "candidate": "ruri-v3-30m",
        "hash": "f1f4baa5d65277aaa90573791f92e1f8ef9e57141c99ac6f98f9f4b88636a4ca",
        "option_only": True,
    },
    "distil4-pair-readout": {
        "candidate": "line-distilbert-ja-first4",
        "hash": "ad8154dff55d69fc31e24457cdb880dc248a0fb68938aaf335b3fd4690b7564b",
        "option_only": False,
    },
    "distil4-option-only": {
        "candidate": "line-distilbert-ja-first4",
        "hash": "06027c4ed828f2f4904b1f6613a2be4a371d98e4ee21d09e6d718cf98be9e384",
        "option_only": True,
    },
    "bert-base-pair-readout": {
        "candidate": "tohoku-bert-base-ja-v3",
        "hash": "431b895b36089dc625b39fda2feee27adec18df6db8964c04bbf6a44f4a12505",
        "option_only": False,
    },
    "bert-base-option-only": {
        "candidate": "tohoku-bert-base-ja-v3",
        "hash": "c25d69b969030e60a2c114675113f35819e308b30716ff52c73f7861bd4159b5",
        "option_only": True,
    },
}
MODEL_DIRS = {
    "ruri-v3-30m": Path(
        r"C:\CodexArtifacts\HuggingFace-cache-MIDI\hub\models--cl-nagoya--ruri-v3-30m\snapshots\24899e5de370b56d179604a007c0d727bf144504"
    ),
    "line-distilbert-ja-first4": Path(
        r"C:\CodexArtifacts\pretrained-reuse-r2-local\line-distilbert-ja-20261123-attempt-01"
    ),
    "tohoku-bert-base-ja-v3": Path(
        r"C:\CodexArtifacts\pretrained-reuse-r2-local\tohoku-bertbase-v3-runtime-20261123-attempt-01"
    ),
}


def _load_encoder(
    candidate: str, model_dir: Path, device: torch.device
) -> tuple[Any, torch.nn.Module, dict[str, Any]]:
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    if candidate == "line-distilbert-ja-first4":
        if config.model_type != "distilbert" or int(config.n_layers) < 5:
            raise ValueError(
                "expected full LINE DistilBERT checkpoint for first-four-layer inference"
            )
        config.n_layers = 4
    weight_file = "model.safetensors" if candidate == "ruri-v3-30m" else "pytorch_model.bin"
    weight_hash = sha256_file(model_dir / weight_file)
    if weight_hash != WEIGHT_HASHES[candidate]:
        raise ValueError(f"pinned encoder weight hash mismatch for {candidate}")
    trust_remote_code = candidate == "line-distilbert-ja-first4"
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
        use_fast=candidate == "ruri-v3-30m",
    )
    encoder, loading = AutoModel.from_pretrained(
        model_dir,
        config=config,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
        use_safetensors=candidate == "ruri-v3-30m",
        output_loading_info=True,
    )
    missing = sorted(str(key) for key in loading["missing_keys"])
    unexpected = sorted(str(key) for key in loading["unexpected_keys"])
    if missing:
        raise ValueError(f"missing checkpoint keys for {candidate}: {missing[:8]}")
    if candidate == "line-distilbert-ja-first4":
        dropped = sorted(
            {
                int(key.split("transformer.layer.", 1)[1].split(".", 1)[0])
                for key in unexpected
                if "transformer.layer." in key
                and key.split("transformer.layer.", 1)[1].split(".", 1)[0].isdigit()
            }
        )
        if dropped != [4, 5]:
            raise ValueError(
                f"expected only pretrained DistilBERT layers 4 and 5 dropped: {dropped}"
            )
    elif candidate == "tohoku-bert-base-ja-v3" and any(
        not key.startswith("cls.") for key in unexpected
    ):
        raise ValueError(f"unexpected BERT-base checkpoint keys: {unexpected[:8]}")
    elif candidate == "ruri-v3-30m" and unexpected:
        raise ValueError(f"unexpected Ruri checkpoint keys: {unexpected[:8]}")
    encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return (
        tokenizer,
        encoder,
        {
            "weight_sha256": weight_hash,
            "parameters": sum(parameter.numel() for parameter in encoder.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad
            ),
            "missing_keys": missing,
            "unexpected_keys": unexpected,
        },
    )


def _split_overlap(dev1: list[dict[str, Any]], dev2: list[dict[str, Any]]) -> dict[str, int]:
    def qids(rows: list[dict[str, Any]]) -> set[str]:
        return {str(row["qid"]) for row in rows}

    def questions(rows: list[dict[str, Any]]) -> set[str]:
        return {_normalized(row["question"]) for row in rows}

    def targets(rows: list[dict[str, Any]]) -> set[str]:
        return {_normalized(row["answer_entity"]) for row in rows}

    def candidate_sets(rows: list[dict[str, Any]]) -> set[tuple[str, ...]]:
        return {
            tuple(sorted(_normalized(option) for option in row["answer_candidates"]))
            for row in rows
        }

    return {
        "qid": len(qids(dev1) & qids(dev2)),
        "normalized_question": len(questions(dev1) & questions(dev2)),
        "answer_entity": len(targets(dev1) & targets(dev2)),
        "unordered_candidate_set": len(candidate_sets(dev1) & candidate_sets(dev2)),
    }


def _write_predictions(
    path: Path,
    rows: list[dict[str, Any]],
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, Any]:
    probabilities = logits.softmax(dim=-1).cpu().numpy()
    predictions = logits.argmax(dim=-1).cpu().numpy()
    with path.open("w", encoding="utf-8") as file:
        for row, target, prediction, probability in zip(
            rows, targets.cpu().tolist(), predictions.tolist(), probabilities, strict=True
        ):
            file.write(
                json.dumps(
                    {
                        "qid": row["qid"],
                        "target_index": target,
                        "prediction_index": int(prediction),
                        "target_entity": row["answer_entity"],
                        "predicted_entity": row["answer_candidates"][int(prediction)],
                        "probabilities": probability.tolist(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return metrics(logits, targets)


def evaluate_once(
    dev1_path: Path,
    dev2_path: Path,
    training_artifact_dir: Path,
    output_dir: Path,
    *,
    selection_commit: str,
    device_name: str = "cuda",
    expected_rows: int = 997,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"one-shot output path already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / ".ONE_SHOT_STARTED").write_text(
        json.dumps({"started_utc": datetime.now(UTC).isoformat()}, indent=2) + "\n",
        encoding="utf-8",
    )
    dev1 = read_jsonl(dev1_path)
    dev2_bytes = dev2_path.read_bytes()
    dev2 = read_jsonl(dev2_path)
    if len(dev2) != expected_rows:
        raise ValueError(f"expected {expected_rows} Dev2 rows, found {len(dev2)}")
    overlap = _split_overlap(dev1, dev2)
    if overlap["qid"] or overlap["normalized_question"] or overlap["unordered_candidate_set"]:
        raise ValueError(f"Dev1/Dev2 record-level or content overlap detected: {overlap}")
    if len({str(row["qid"]) for row in dev2}) != len(dev2):
        raise ValueError("Dev2 qids are not unique")
    (output_dir / "preflight.json").write_text(
        json.dumps(
            {
                "dev2_sha256": hashlib.sha256(dev2_bytes).hexdigest(),
                "dev2_bytes": len(dev2_bytes),
                "dev2_rows": len(dev2),
                "dev1_dev2_overlap": overlap,
                "started_utc": datetime.now(UTC).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    device = torch.device(device_name)
    targets = _targets(dev2).to(device)
    results: dict[str, Any] = {}
    run_started = datetime.now(UTC).isoformat()
    for candidate in WEIGHT_HASHES:
        model_dir = MODEL_DIRS[candidate]
        tokenizer, encoder, model_metadata = _load_encoder(candidate, model_dir, device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        frozen_hash = hashlib.sha256()
        for name, tensor in sorted(encoder.state_dict().items()):
            frozen_hash.update(name.encode("utf-8"))
            frozen_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        frozen_hash_before = frozen_hash.hexdigest()
        for condition, metadata in FROZEN_READOUTS.items():
            if metadata["candidate"] != candidate:
                continue
            readout_path = training_artifact_dir / condition / "best-readout.pt"
            actual_readout_hash = sha256_file(readout_path)
            if actual_readout_hash != metadata["hash"]:
                raise ValueError(f"frozen readout hash mismatch for {condition}")
            condition_dir = output_dir / condition
            condition_dir.mkdir()
            features_path = condition_dir / "dev2-features.npy"
            feature_meta = encode_split(
                dev2,
                tokenizer,
                encoder,
                device,
                batch_size=64,
                max_length=256,
                option_only=bool(metadata["option_only"]),
                output_path=features_path,
            )
            features = torch.tensor(
                np.asarray(np.load(features_path, mmap_mode="r"), dtype=np.float32), device=device
            )
            readout = SharedOptionScorer(features.shape[-1]).to(device)
            state = torch.load(readout_path, map_location=device, weights_only=True)
            readout.load_state_dict(state, strict=True)
            readout.eval()
            with torch.inference_mode():
                logits = readout(features)
            metric_values = _write_predictions(
                condition_dir / "dev2-predictions.jsonl", dev2, logits, targets
            )
            results[condition] = {
                "readout_sha256": actual_readout_hash,
                "encoder": model_metadata,
                "encoding": feature_meta,
                "metrics": metric_values,
                "prediction_sha256": sha256_file(condition_dir / "dev2-predictions.jsonl"),
                "feature_cache_sha256": sha256_file(features_path),
                "peak_gpu_memory_bytes": (
                    torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
                ),
            }
            del readout, features, logits
        after_hash = hashlib.sha256()
        for name, tensor in sorted(encoder.state_dict().items()):
            after_hash.update(name.encode("utf-8"))
            after_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        if after_hash.hexdigest() != frozen_hash_before:
            raise ValueError(f"frozen encoder changed during Dev2 evaluation: {candidate}")
        del encoder, tokenizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = {
        "experiment_id": "jaqket-dev2-one-shot-20261124",
        "started_utc": run_started,
        "selection_commit": selection_commit,
        "dev2_sha256": hashlib.sha256(dev2_bytes).hexdigest(),
        "dev2_rows": len(dev2),
        "dev1_dev2_overlap": overlap,
        "device": device_name,
        "evaluation_only": True,
        "selection_or_tuning_on_dev2": False,
        "sealed_audit": False,
        "conditions": results,
    }
    (output_dir / "one-shot-result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev1", type=Path, required=True)
    parser.add_argument("--dev2", type=Path, required=True)
    parser.add_argument("--training-artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-commit", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_once(
                args.dev1,
                args.dev2,
                args.training_artifact_dir,
                args.output_dir,
                selection_commit=args.selection_commit,
                device_name=args.device,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
