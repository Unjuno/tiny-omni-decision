"""Embed the frozen, supplied HABIT action options with pinned MiniLM."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from tiny_omni_decision.habit_transition import resolve_episode_options

DATASET_REVISION = "bae97d251907743f8fb6db7e70c61d361df4dc1e"
PLAN_SHA256 = "fe9d8012131db25578d25a6501a9c56c50dd2d4ddef33cfd2145d3aa81cb8f1b"
MINILM_ID = "sentence-transformers/paraphrase-MiniLM-L3-v2"
MINILM_REVISION = "4ca70771034acceecb2e72475f72050fcdde4ddc"
MINILM_WEIGHT_SHA256 = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"
MINILM_PARAMETERS = 17_389_824


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--encoder-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan_path = args.attempt_root / "metadata-plan.json"
    if sha256_file(plan_path) != PLAN_SHA256:
        raise ValueError("metadata-plan hash differs from frozen preflight")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("dataset") != "configinc/HABIT" or plan.get("revision") != DATASET_REVISION:
        raise ValueError("unexpected HABIT dataset revision")
    if any(item.get("split") == "reserved" for item in plan["file_tree_files"]):
        raise ValueError("reserved split appears in frozen media plan")

    episodes = {
        int(row["episode_index"]): row
        for row in (
            json.loads(line)
            for line in (args.metadata_root / "full__meta__episodes.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    subtasks = {
        int(row["task_index"]): row["task"]
        for row in (
            json.loads(line)
            for line in (args.metadata_root / "full__meta__subtasks.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    options_by_id: dict[int, str] = {}
    option_ids_by_split = {"train": set(), "validation": set()}
    for split in ("train", "validation"):
        for item in plan["sampled_episode_ids"][split]:
            row = episodes[int(item["episode_index"])]
            if row.get("sid") != item["sid"] or item["sid"] not in item["parent_group"].split("_"):
                raise ValueError(f"episode/group mismatch: {item['episode_index']}")
            options = resolve_episode_options(row["low_level_tasks"], subtasks)
            if len(options) < 2:
                continue
            for option_id, text in options:
                previous = options_by_id.setdefault(option_id, text)
                if previous != text:
                    raise ValueError(f"global option ID maps to multiple texts: {option_id}")
                option_ids_by_split[split].add(option_id)

    if option_ids_by_split["train"] & option_ids_by_split["validation"]:
        raise ValueError("train/validation candidate ID overlap")
    option_ids = sorted(options_by_id)
    texts = [options_by_id[option_id] for option_id in option_ids]
    if not texts:
        raise ValueError("no multi-option candidate text in frozen corpus")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite feature output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    import torch
    from transformers import AutoModel, AutoTokenizer

    snapshot = args.encoder_snapshot.resolve(strict=True)
    weight_path = snapshot / "model.safetensors"
    if sha256_file(weight_path) != MINILM_WEIGHT_SHA256:
        raise ValueError("MiniLM weight SHA-256 mismatch")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model = AutoModel.from_pretrained(str(snapshot), local_files_only=True).eval().to("cpu")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != MINILM_PARAMETERS:
        raise ValueError(f"unexpected MiniLM parameter count: {parameters}")

    started = time.perf_counter()
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), 32):
            batch_texts = texts[start : start + 32]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            hidden = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            if not torch.isfinite(pooled).all():
                raise FloatingPointError("MiniLM returned non-finite option features")
            embeddings.append(pooled.cpu().numpy().astype(np.float32, copy=False))
    vectors = np.concatenate(embeddings, axis=0)
    if vectors.shape != (len(option_ids), 384) or not np.isfinite(vectors).all():
        raise ValueError(f"invalid option feature matrix: {vectors.shape}")

    feature_path = args.output_dir / "option-vectors.npz"
    np.savez_compressed(
        feature_path,
        option_ids=np.asarray(option_ids, dtype=np.int64),
        option_texts=np.asarray(texts),
        vectors=vectors,
    )
    report = {
        "status": "pass_pinned_minilm_option_feature_cache",
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "dataset_revision": f"configinc/HABIT@{DATASET_REVISION}",
        "metadata_plan_sha256": PLAN_SHA256,
        "encoder_id": MINILM_ID,
        "encoder_revision": MINILM_REVISION,
        "encoder_weight_sha256": MINILM_WEIGHT_SHA256,
        "encoder_parameters": parameters,
        "encoder_dtype": str(next(model.parameters()).dtype),
        "pooling": "attention-mask mean over last hidden state",
        "options": len(option_ids),
        "train_option_ids": len(option_ids_by_split["train"]),
        "validation_option_ids": len(option_ids_by_split["validation"]),
        "train_validation_option_id_overlap": 0,
        "vectors_shape": list(vectors.shape),
        "vectors_sha256": hashlib.sha256(vectors.tobytes()).hexdigest(),
        "artifact_sha256": sha256_file(feature_path),
        "artifact_bytes": feature_path.stat().st_size,
        "wall_seconds": time.perf_counter() - started,
        "reserved_data_loaded": False,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
    }
    (args.output_dir / "receipt.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
