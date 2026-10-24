"""TRAIN-only shared-readout gradient diagnostic for the frozen native fit."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import types
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.dont_write_bytecode = True

RUN = Path(r"C:\CodexArtifacts\pretrained-reuse-r2\native-coverage-repeat-fit-01")
REPO = Path(__file__).resolve().parents[1]
SOURCE_SNAPSHOT = REPO / "docs/pretrained_reuse/source_snapshots/native_coverage_repeat_fit_01"
CONFIG = RUN / "effective-config.json"
CHECKPOINT = RUN / "best.pt"
OUTPUT_DEFAULT = Path(r"C:\CodexArtifacts\pretrained-reuse-r2\native-shared-gradient-audit-01")
FROZEN_MANIFEST = REPO / "configs/pretrained_reuse/native_shared_gradient_audit_v1.json"
MODALITIES = ("text", "image", "audio", "video")
MAX_STEPS = 2048
SAMPLED_STEPS = tuple((i * (MAX_STEPS - 1)) // 31 for i in range(32))


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def option_order(row: dict, occurrence: int, seed: int = 17) -> list[int]:
    digest = hashlib.sha256(f"{seed}:{occurrence}:{row['id']}".encode()).digest()
    order = list(range(len(row["options"])))
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(order)
    return order


def run_source_hashes() -> dict[str, str]:
    return {
        "scripts/native_fit.py": sha_file(SOURCE_SNAPSHOT / "scripts/native_fit.py"),
        "scripts/native_fit_runtime.py": sha_file(
            SOURCE_SNAPSHOT / "scripts/native_fit_runtime.py"
        ),
        "scripts/native_candidate_inputs.py": sha_file(
            SOURCE_SNAPSHOT / "scripts/native_candidate_inputs.py"
        ),
        "scripts/native_video_cache.py": sha_file(
            SOURCE_SNAPSHOT / "scripts/native_video_cache.py"
        ),
        "src/tiny_omni_decision/native_candidate.py": sha_file(
            SOURCE_SNAPSHOT / "tiny_omni_decision/native_candidate.py"
        ),
    }


def load_snapshot_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load pinned run source: " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    parent_name, attribute = name.rsplit(".", 1)
    parent = sys.modules.get(parent_name)
    if parent is None:
        parent = types.ModuleType(parent_name)
        parent.__path__ = []
        sys.modules[parent_name] = parent
    setattr(parent, attribute, module)
    return module


def load_run_source(run_result: dict):
    modules = {
        "tiny_omni_decision.native_candidate": (
            SOURCE_SNAPSHOT / "tiny_omni_decision/native_candidate.py"
        ),
        "scripts.native_candidate_inputs": SOURCE_SNAPSHOT / "scripts/native_candidate_inputs.py",
        "scripts.native_fit_runtime": SOURCE_SNAPSHOT / "scripts/native_fit_runtime.py",
        "scripts.native_video_cache": SOURCE_SNAPSHOT / "scripts/native_video_cache.py",
    }
    for module_name, path in modules.items():
        suffix = "\\" + module_name.rsplit(".", 1)[-1].casefold() + ".py"
        if module_name.startswith("tiny_omni_decision."):
            suffix = "\\src\\tiny_omni_decision\\native_candidate.py"
        else:
            suffix = "\\scripts\\" + module_name.rsplit(".", 1)[-1].casefold() + ".py"
        matches = [
            digest
            for source, digest in run_result["validated_hashes"].items()
            if source.casefold().endswith(suffix)
        ]
        if len(matches) != 1 or sha_file(path) != matches[0]:
            raise ValueError("bundled run-source hash differs: " + module_name)
        load_snapshot_module(module_name, path)
    native_fit_path = SOURCE_SNAPSHOT / "scripts/native_fit.py"
    fit_hashes = [
        digest
        for source, digest in run_result["validated_hashes"].items()
        if source.casefold().endswith("\\scripts\\native_fit.py")
    ]
    if len(fit_hashes) != 1 or sha_file(native_fit_path) != fit_hashes[0]:
        raise ValueError("bundled native_fit source hash differs")
    return (
        sys.modules["tiny_omni_decision.native_candidate"].NativeCandidateReadout,
        sys.modules["scripts.native_fit_runtime"].FeatureStore,
        run_source_hashes(),
    )


def selected_updates(order: list[list[int]], occurrences: list[list[int]]) -> list[dict]:
    if len(order) != MAX_STEPS or len(occurrences) != MAX_STEPS:
        raise ValueError("the completed 2048-update repeat sampler is required")
    selected = []
    for step_index in SAMPLED_STEPS:
        row_indices, occurrence_ids = order[step_index], occurrences[step_index]
        if len(row_indices) != 8 or len(occurrence_ids) != 8:
            raise ValueError("each recorded update must contain exactly eight examples")
        if len(set(occurrence_ids)) != 8:
            raise ValueError("sampler occurrence IDs must be unique within an update")
        selected.append(
            {
                "optimizer_step": step_index + 1,
                "row_indices": row_indices,
                "occurrence_ids": occurrence_ids,
            }
        )
    return selected


def cosine_and_norms(gradients: dict[str, torch.Tensor]) -> dict:
    names = MODALITIES
    norms = {name: float(torch.linalg.vector_norm(gradients[name]).item()) for name in names}
    cosines = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            denominator = norms[left] * norms[right]
            value = (
                float(torch.dot(gradients[left], gradients[right]).item() / denominator)
                if denominator > 0
                else None
            )
            cosines[f"{left}|{right}"] = value
    summed = sum((gradients[name] for name in names), torch.zeros_like(gradients[names[0]]))
    summed_norm = float(torch.linalg.vector_norm(summed).item())
    norm_sum = math.fsum(norms.values())
    batch_gradient_norm = summed_norm / len(names)
    clip_factor = min(1.0, 1.0 / batch_gradient_norm) if batch_gradient_norm > 0 else 1.0
    projections = {}
    denominator = float(torch.dot(summed, summed).item())
    for name in names:
        projections[name] = (
            float(torch.dot(gradients[name], summed).item() / denominator)
            if denominator > 0
            else None
        )
    return {
        "modality_gradient_norms": norms,
        "pairwise_cosines": cosines,
        "cancellation_ratio": summed_norm / norm_sum if norm_sum > 0 else None,
        "unclipped_mean_batch_gradient_norm": batch_gradient_norm,
        "clip_factor_at_training_max_norm_1": clip_factor,
        "would_clip": batch_gradient_norm > 1.0,
        "modality_projection_shares": projections,
    }


def validate_hotgpu_gate() -> dict:
    required = (
        "TINY_OMNI_GPU_GATE_RESERVED_MIB",
        "TINY_OMNI_GPU_GATE_PARENT_PID",
        "TINY_OMNI_GPU_GATE_BLOCK_IF_FREE_MIB",
        "TINY_OMNI_GPU_GATE_MIN_CHILD_FREE_MIB",
    )
    if any(key not in os.environ for key in required):
        raise RuntimeError("HotUGPU reservation required; direct GPU launch is forbidden")
    utilization, used, total = (
        int(item.strip())
        for item in subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        .splitlines()[0]
        .split(",")
    )
    active_pids = {
        int(item.strip())
        for item in subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True,
        ).splitlines()
        if item.strip() and item.strip().casefold() != "no running processes found"
    }
    parent_pid = int(os.environ["TINY_OMNI_GPU_GATE_PARENT_PID"])
    free = total - used
    if (
        parent_pid not in active_pids
        or active_pids - {parent_pid, os.getpid()}
        or utilization >= 5
        or int(os.environ["TINY_OMNI_GPU_GATE_RESERVED_MIB"]) < 1
        or not int(os.environ["TINY_OMNI_GPU_GATE_MIN_CHILD_FREE_MIB"])
        <= free
        < int(os.environ["TINY_OMNI_GPU_GATE_BLOCK_IF_FREE_MIB"])
    ):
        raise RuntimeError("HotUGPU handoff is absent, raced, or outside its free-memory gate")
    return {
        "mode": "hotgpu_reserved",
        "parent_pid": parent_pid,
        "active_compute_pids": sorted(active_pids),
        "utilization_pct": utilization,
        "memory_used_mib": used,
        "memory_free_mib": free,
        "memory_total_mib": total,
        "reserved_mib": int(os.environ["TINY_OMNI_GPU_GATE_RESERVED_MIB"]),
    }


def write_json(path: Path, value: dict | list) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def audit(output: Path) -> dict:
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source_tree_clean = not subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    ).strip()
    if not source_tree_clean:
        raise RuntimeError("committed clean source required before the GPU diagnostic")
    frozen = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result = {
        "status": "RUNNING",
        "started_utc": datetime.now(UTC).isoformat(),
        "source_commit": source_commit,
        "source_tree_clean": source_tree_clean,
        "run_id": RUN.name,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha_file(CHECKPOINT),
        "checkpoint_step": 2048,
        "checkpoint_selector": (
            "repeat-run best by frozen validation macro NLL; validation not loaded here"
        ),
        "validation_loaded": False,
        "sealed_audit_loaded": False,
        "optimizer_updates": 0,
        "cloud_cost": 0,
    }
    write_json(output / "result.json", result)
    try:
        gate = validate_hotgpu_gate()
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        run_result = json.loads((RUN / "result.json").read_text(encoding="utf-8"))
        best_meta = json.loads((RUN / "best.json").read_text(encoding="utf-8"))
        if (
            run_result.get("status") != "PASS_NATIVE_COVERAGE_TRAIN"
            or run_result.get("step") != MAX_STEPS
            or run_result.get("best_step") != MAX_STEPS
            or best_meta.get("step") != MAX_STEPS
            or best_meta.get("checkpoint") != "checkpoint-2048.pt"
            or sha_file(CHECKPOINT) != sha_file(RUN / "checkpoint-2048.pt")
            or run_result.get("source_commit") != "b3b7bb3abc4d27b6da51aa05032d2ff14704362f"
        ):
            raise ValueError("pinned completed repeat run and selected checkpoint required")
        readout_class, feature_store_class, snapshot_hashes = load_run_source(run_result)
        result.update(
            run_reported_source_commit=run_result["source_commit"],
            run_source_commit_caveat=(
                "5 validated dependency files differ from the tree at the reported source commit; "
                "the exact validated bytes are bundled and hashed"
            ),
            run_source_snapshot_hashes=snapshot_hashes,
        )
        pinned_run_files = {
            str(CONFIG): run_result["config_sha256"],
            str(CHECKPOINT): run_result["artifacts"]["best.pt"],
            str(RUN / "result.json"): sha_file(RUN / "result.json"),
            str(RUN / "best.json"): run_result["artifacts"]["best.json"],
            str(RUN / "sampling.json"): run_result["artifacts"]["sampling.json"],
        }
        for path_string, expected in pinned_run_files.items():
            if sha_file(Path(path_string)) != expected:
                raise ValueError("frozen fit run file hash differs: " + path_string)
        corpus = Path(config["corpus"])
        queries_dir = Path(config["queries"])
        corpus_receipt_path = corpus / "result.json"
        query_receipt_path = queries_dir / "result.json"
        for path in (corpus_receipt_path, query_receipt_path):
            expected = run_result["validated_hashes"].get(str(path))
            if not expected or sha_file(path) != expected:
                raise ValueError("frozen corpus/cache receipt hash differs: " + str(path))
        corpus_receipt = json.loads(corpus_receipt_path.read_text(encoding="utf-8"))
        query_receipt = json.loads(query_receipt_path.read_text(encoding="utf-8"))
        if (
            corpus_receipt.get("status") != "PASS_COMPLETE_NATIVE_COMMON_CORPUS"
            or query_receipt.get("status") != "PASS_NATIVE_COMMON_QUERIES_CACHE"
        ):
            raise ValueError("completed pinned corpus and query cache are required")
        train = [
            json.loads(line)
            for line in (corpus / "train.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        if len(train) != 100580 or any(row.get("split") != "train" for row in train):
            raise ValueError("pinned TRAIN corpus required; validation is not opened")
        sampling = json.loads((RUN / "sampling.json").read_text(encoding="utf-8"))
        selected = selected_updates(sampling["order"], sampling["occurrence_plan"]["occurrences"])
        sample_rows = []
        for update in selected:
            batch = []
            for row_index, occurrence_id in zip(
                update["row_indices"], update["occurrence_ids"], strict=True
            ):
                row = train[row_index]
                batch.append((row, occurrence_id))
            counts = {m: sum(row["modality"] == m for row, _ in batch) for m in MODALITIES}
            if counts != {m: 2 for m in MODALITIES}:
                raise ValueError("sampled training update must contain two rows per modality")
            update["counts_by_modality"] = counts
            update["sample_ids"] = [row["id"] for row, _ in batch]
            sample_rows.extend(row for row, _ in batch)
        if any(len(row["options"]) != len(row["queries"]) for row in sample_rows):
            raise ValueError("fixed TRAIN rows must have aligned options and queries")
        ids = [row["id"] for row in sample_rows]
        write_json(output / "selected-train-occurrences.json", selected)
        config_hashes = {
            str(path): sha_file(path)
            for path in (CONFIG, RUN / "result.json", RUN / "best.json", RUN / "sampling.json")
        }
        data_hashes = {
            str(path): sha_file(path)
            for path in (
                corpus / "result.json",
                corpus / "train.jsonl",
                corpus / "observations.json",
                corpus / "banks.json",
                corpus / "query-map.json",
                queries_dir / "result.json",
                queries_dir / "queries.npy",
            )
        }
        for name in ("train.jsonl", "observations.json", "banks.json", "query-map.json"):
            if data_hashes[str(corpus / name)] != corpus_receipt["artifacts"][name]:
                raise ValueError("frozen corpus artifact hash differs: " + name)
        if (
            data_hashes[str(queries_dir / "queries.npy")]
            != query_receipt["artifacts"]["queries.npy"]
        ):
            raise ValueError("frozen query feature hash differs")
        used_observations = {row["observation"] for row in sample_rows}
        observations = json.loads((corpus / "observations.json").read_text(encoding="utf-8"))
        banks = json.loads((corpus / "banks.json").read_text(encoding="utf-8"))
        query_map = json.loads((corpus / "query-map.json").read_text(encoding="utf-8"))
        banks["new/queries.npy"] = {
            "path": str(queries_dir / "queries.npy"),
            "sha256": query_receipt["artifacts"]["queries.npy"],
            "shape": [2622, 384],
        }
        bank_paths = {
            banks[feature["bank"]]["path"]
            for ident in used_observations
            for feature in observations[ident]["features"]
        }
        bank_paths.add(str(queries_dir / "queries.npy"))
        files_to_verify = {}
        for path_string in bank_paths:
            path = Path(path_string)
            expected = (
                query_receipt["artifacts"]["queries.npy"]
                if path == queries_dir / "queries.npy"
                else next(
                    (
                        digest
                        for source, digest in run_result.get("validated_hashes", {}).items()
                        if Path(source) == path
                    ),
                    None,
                )
            )
            if expected is None:
                raise ValueError("sampled feature bank lacks a frozen source hash")
            files_to_verify[str(path)] = expected
        for path_string, expected in files_to_verify.items():
            if sha_file(Path(path_string)) != expected:
                raise ValueError("sampled feature bank hash differs: " + path_string)
        expected_corpus_hashes = {
            name: digest
            for name, digest in corpus_receipt["artifacts"].items()
            if name in ("train.jsonl", "observations.json", "banks.json", "query-map.json")
        }
        if (
            frozen.get("run_receipt_sha256") != sha_file(RUN / "result.json")
            or frozen.get("config_sha256") != sha_file(CONFIG)
            or frozen.get("best_metadata_sha256") != sha_file(RUN / "best.json")
            or frozen.get("corpus_receipt_sha256") != sha_file(corpus_receipt_path)
            or frozen.get("query_cache_receipt_sha256") != sha_file(query_receipt_path)
            or frozen.get("corpus_artifact_hashes") != expected_corpus_hashes
            or frozen.get("query_feature_sha256") != query_receipt["artifacts"]["queries.npy"]
            or frozen.get("sampled_feature_bank_hashes") != files_to_verify
        ):
            raise ValueError("pre-registered input hashes differ from the frozen experiment")
        if len(ids) != 256:
            raise ValueError("the predeclared 256 TRAIN occurrences are required")
        selected_projection = [
            {
                "optimizer_step": update["optimizer_step"],
                "row_indices": update["row_indices"],
                "occurrence_ids": update["occurrence_ids"],
                "sample_ids": update["sample_ids"],
            }
            for update in selected
        ]
        ids_payload = json.dumps(
            selected_projection, ensure_ascii=False, separators=(",", ":")
        ).encode()
        if (
            frozen.get("source_commit") != run_result["source_commit"]
            or frozen.get("checkpoint_sha256") != sha_file(CHECKPOINT)
            or frozen.get("selected_updates") != selected_projection
            or frozen.get("sampler_sha256") != sha_file(RUN / "sampling.json")
            or frozen.get("row_id_order_sha256") != digest_bytes("\n".join(ids).encode())
            or frozen.get("occurrences_sha256") != digest_bytes(ids_payload)
        ):
            raise ValueError("pre-registered TRAIN sample manifest differs from the frozen run")
        media_asset_ids = {
            row["observation"].rsplit(":", 1)[0]
            if row["source"] == "starss"
            else row["observation"]
            for row in sample_rows
            if row["modality"] != "text"
        }
        write_json(
            output / "preflight.json",
            {
                "status": "PASS",
                "checkpoint_sha256": sha_file(CHECKPOINT),
                "frozen_manifest_sha256": sha_file(FROZEN_MANIFEST),
                "checkpoint_step": 2048,
                "sampler_sha256": sha_file(RUN / "sampling.json"),
                "selected_occurrences_sha256": digest_bytes(ids_payload),
                "selected_row_id_order_sha256": digest_bytes("\n".join(ids).encode()),
                "selected_update_indices_zero_based": list(SAMPLED_STEPS),
                "selected_optimizer_steps_one_based": [s + 1 for s in SAMPLED_STEPS],
                "sample_count": len(ids),
                "unique_sample_count": len(set(ids)),
                "duplicate_sample_count": len(ids) - len(set(ids)),
                "modality_counts": {
                    m: sum(row["modality"] == m for row in sample_rows) for m in MODALITIES
                },
                "source_counts": {
                    k: sum(row["source"] == k for row in sample_rows)
                    for k in sorted({row["source"] for row in sample_rows})
                },
                "question_type_counts": {
                    k: sum(row["task"] == k for row in sample_rows)
                    for k in sorted({row["task"] for row in sample_rows})
                },
                "observation_count": len(used_observations),
                "unique_media_asset_count": len(media_asset_ids),
                "unique_video_scene_count": len(
                    {row["group"] for row in sample_rows if row["modality"] == "video"}
                ),
                "unique_group_count": len({(row["modality"], row["group"]) for row in sample_rows}),
                "config_and_run_hashes": config_hashes,
                "data_hashes": data_hashes,
                "sampled_feature_bank_hashes": files_to_verify,
                "hotgpu_gate": gate,
                "run_reported_source_commit": run_result["source_commit"],
                "run_source_snapshot_hashes": snapshot_hashes,
            },
        )
        torch.set_num_threads(4)
        torch.manual_seed(17)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        model = readout_class().cuda()
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or checkpoint.get("step") != MAX_STEPS:
            raise ValueError("expected checkpoint wrapper with global step 2048")
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        parameter_list = tuple(model.parameters())
        parameter_count = sum(p.numel() for p in parameter_list)
        if parameter_count != 889345:
            raise ValueError("shared-readout parameter count changed")
        store = feature_store_class(observations, banks, query_map)
        accumulated = defaultdict(list)
        measurement_rows = []
        for update in selected:
            rows = [
                (train[index], occurrence)
                for index, occurrence in zip(
                    update["row_indices"], update["occurrence_ids"], strict=True
                )
            ]
            gradients = {}
            for modality in MODALITIES:
                modality_losses = []
                for row, occurrence in rows:
                    if row["modality"] != modality:
                        continue
                    blocks, queries = store.get(row)
                    permutation = option_order(row, occurrence)
                    scores = model(blocks, queries[permutation])
                    target = torch.tensor([permutation.index(row["target_index"])], device="cuda")
                    modality_losses.append(F.cross_entropy(scores.unsqueeze(0), target))
                if len(modality_losses) != 2:
                    raise ValueError("expected two sampled examples per modality/update")
                gradients_tuple = torch.autograd.grad(
                    torch.stack(modality_losses).mean(), parameter_list
                )
                gradients[modality] = torch.cat(
                    [grad.reshape(-1) for grad in gradients_tuple]
                ).detach()
            stats = cosine_and_norms(gradients)
            stats["optimizer_step"] = update["optimizer_step"]
            stats["sample_ids"] = update["sample_ids"]
            measurement_rows.append(stats)
            for name, value in stats["pairwise_cosines"].items():
                accumulated["cosine:" + name].append(value)
            accumulated["clip_factor"].append(stats["clip_factor_at_training_max_norm_1"])
            accumulated["global_norm"].append(stats["unclipped_mean_batch_gradient_norm"])
            accumulated["cancellation_ratio"].append(stats["cancellation_ratio"])
            accumulated["would_clip"].append(float(stats["would_clip"]))
            if (len(measurement_rows) % 4) == 0:
                print("measured updates", len(measurement_rows), "/", len(selected), flush=True)
        summary = {}
        for key, values in accumulated.items():
            clean = [v for v in values if v is not None and math.isfinite(v)]
            summary[key] = {
                "mean": float(np.mean(clean)),
                "median": float(np.median(clean)),
                "std": float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0,
                "min": float(np.min(clean)),
                "max": float(np.max(clean)),
                "negative_fraction": float(np.mean(np.asarray(clean) < 0))
                if key.startswith("cosine:")
                else None,
                "n": len(clean),
            }
        torch.cuda.synchronize()
        result.update(
            status="PASS_TRAIN_ONLY_SHARED_GRADIENT_AUDIT",
            finished_utc=datetime.now(UTC).isoformat(),
            optimizer_updates=0,
            sampled_optimizer_steps=[update["optimizer_step"] for update in selected],
            model_weight_updates=0,
            examples_evaluated=len(ids),
            unique_examples=len(set(ids)),
            trainable_parameters=parameter_count,
            total_parameters=195365529 + parameter_count,
            device=torch.cuda.get_device_name(),
            dtype="float32",
            python=platform.python_version(),
            torch=torch.__version__,
            numpy=np.__version__,
            cuda=torch.version.cuda,
            attention_backend="torch eager MultiheadAttention",
            validation_loaded=False,
            sealed_audit_loaded=False,
            hotgpu_gate=gate,
            elapsed_seconds=time.perf_counter() - started,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            summary=summary,
            artifacts={
                "preflight.json": sha_file(output / "preflight.json"),
                "selected-train-occurrences.json": sha_file(
                    output / "selected-train-occurrences.json"
                ),
            },
        )
        with (output / "measurements.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in measurement_rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        result["artifacts"]["measurements.jsonl"] = sha_file(output / "measurements.jsonl")
        write_json(output / "result.json", result)
        return result
    except Exception as error:
        result.update(
            status="FAIL_TRAIN_ONLY_SHARED_GRADIENT_AUDIT",
            finished_utc=datetime.now(UTC).isoformat(),
            error=repr(error),
            elapsed_seconds=time.perf_counter() - started,
        )
        write_json(output / "result.json", result)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    audit(args.output)


if __name__ == "__main__":
    main()
