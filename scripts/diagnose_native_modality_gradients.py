"""Measure fixed-window, per-modality gradients without training or validation."""

import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs/pretrained_reuse/native_modality_gradient_diagnostic_v1.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
SOURCE_ROOT = Path(CONFIG["historical_source_root"])
sys.path.insert(0, str(SOURCE_ROOT))

from scripts.native_fit import option_order  # noqa: E402
from scripts.native_fit_runtime import FeatureStore  # noqa: E402
from scripts.native_gradient_diagnostic import (  # noqa: E402
    MODALITIES,
    fixed_update_window,
    summarize_updates,
    update_gradient_metrics,
)
from scripts.native_video_cache import sha  # noqa: E402
from tiny_omni_decision.native_candidate import NativeCandidateReadout  # noqa: E402


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def verify_file(path, expected):
    path = Path(path)
    if not path.is_file() or sha(path) != expected:
        raise ValueError("pinned file hash mismatch: " + str(path))


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def flatten_gradients(named_parameters, gradients, predicate):
    pieces = [
        (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
        for (name, parameter), gradient in zip(named_parameters, gradients, strict=True)
        if predicate(name)
    ]
    if not pieces:
        raise ValueError("gradient parameter selection is empty")
    return torch.cat(pieces)


def load_training_inputs(config):
    fit = Path(config["baseline_fit"])
    fit_result_path = fit / "result.json"
    verify_file(fit_result_path, config["baseline_result_sha256"])
    fit_result = read_json(fit_result_path)
    if (
        fit_result.get("status") != "PASS_NATIVE_LONG_BUDGET_TRAIN"
        or fit_result.get("step") != 32768
        or fit_result.get("best_step") != config["selected_step"]
        or fit_result.get("source_commit") != config["training_source_commit"]
        or fit_result.get("encoder_updates") != 0
        or fit_result.get("cloud_cost") != 0
    ):
        raise ValueError("completed frozen-encoder baseline receipt required")

    verify_file(fit / config["selected_checkpoint"], config["selected_checkpoint_sha256"])
    verify_file(fit / "sampling.json", config["sampling_sha256"])
    verify_file(fit / "train-probe-ids.json", config["train_probe_ids_sha256"])
    source_config = REPO / config["source_config"]
    verify_file(source_config, config["source_config_sha256"])
    source_config_data = read_json(source_config)
    corpus = Path(source_config_data["corpus"])
    queries = Path(source_config_data["queries"])
    for relative, expected in CONFIG["source_file_sha256"].items():
        verify_file(SOURCE_ROOT / relative, expected)

    corpus_result_path = corpus / "result.json"
    query_result_path = queries / "result.json"
    verify_file(corpus_result_path, config["corpus_result_sha256"])
    verify_file(query_result_path, config["query_cache_result_sha256"])
    corpus_result, query_result = read_json(corpus_result_path), read_json(query_result_path)
    if (
        corpus_result.get("status") != "PASS_COMPLETE_NATIVE_COMMON_CORPUS"
        or query_result.get("status") != "PASS_NATIVE_COMMON_QUERIES_CACHE"
    ):
        raise ValueError("pinned corpus and query-cache receipts required")
    for name, expected in config["corpus_artifact_sha256"].items():
        verify_file(corpus / name, expected)
    for name, expected in config["query_cache_artifact_sha256"].items():
        verify_file(queries / name, expected)

    train = [
        json.loads(line)
        for line in (corpus / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if len(train) != 100580 or any(row.get("split") != "train" for row in train):
        raise ValueError("exact train-only corpus required")
    if len({row["id"] for row in train}) != len(train):
        raise ValueError("duplicate TRAIN sample IDs")
    rows_by_id = {row["id"]: row for row in train}
    train_ids = read_json(fit / "train-probe-ids.json")
    if len(train_ids) != config["expected_train_probe_count"] or len(set(train_ids)) != len(
        train_ids
    ):
        raise ValueError("fixed long-budget TRAIN probe ID list differs")

    sampling = read_json(fit / "sampling.json")
    if len(sampling.get("order", [])) != 32768:
        raise ValueError("complete baseline update-order receipt required")
    start, window = fixed_update_window(
        sampling["order"], config["selected_step"], config["window_updates"]
    )
    selected_rows = []
    update_rows = []
    for batch in window:
        rows = [train[index] for index in batch]
        counts = {
            modality: sum(row["modality"] == modality for row in rows) for modality in MODALITIES
        }
        if counts != dict.fromkeys(MODALITIES, 2):
            raise ValueError("baseline window must retain two examples per modality per update")
        update_rows.append(rows)
        selected_rows.extend(rows)
    selected_ids = [row["id"] for row in selected_rows]
    if not set(selected_ids) <= set(train_ids):
        raise ValueError("selected window is outside the fixed consumed-TRAIN sample pool")

    observations_all = read_json(corpus / "observations.json")
    banks_all = read_json(corpus / "banks.json")
    query_map_all = read_json(corpus / "query-map.json")
    observation_ids = {row["observation"] for row in selected_rows}
    observations = {ident: observations_all[ident] for ident in observation_ids}
    bank_names = {
        feature["bank"] for ident in observation_ids for feature in observations[ident]["features"]
    }
    banks = {name: banks_all[name] for name in bank_names}
    query_names = {name for row in selected_rows for name in row["queries"]}
    if not query_names <= set(query_map_all):
        raise ValueError("TRAIN query map incomplete")
    query_map = {name: query_map_all[name] for name in query_names}
    query_bank = {
        "path": str(queries / "queries.npy"),
        "sha256": query_result["artifacts"]["queries.npy"],
        "shape": [2622, 384],
    }
    banks["new/queries.npy"] = query_bank
    for entry in banks.values():
        verify_file(entry["path"], entry["sha256"])

    metadata = {
        "selected_step": config["selected_step"],
        "window_start_update_1based": start + 1,
        "window_end_update_inclusive": config["selected_step"],
        "window_updates": len(window),
        "occurrences": len(selected_rows),
        "unique_questions": len(set(selected_ids)),
        "repeat_occurrences": len(selected_ids) - len(set(selected_ids)),
        "sample_ids_sha256": hashlib.sha256(
            json.dumps(selected_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "modalities": {
            modality: sum(row["modality"] == modality for row in selected_rows)
            for modality in MODALITIES
        },
        "sources": {},
        "tasks": {},
        "unique_observations": len(observation_ids),
        "unique_assets": len(
            {
                row["observation"].rsplit(":", 1)[0]
                if row["source"] == "starss"
                else row["observation"]
                for row in selected_rows
                if row["modality"] != "text"
            }
        ),
        "unique_groups": len({(row["modality"], row["group"]) for row in selected_rows}),
        "validation_rows_loaded": 0,
        "validation_labels_loaded": False,
        "validation_features_selected": False,
        "sealed_audit_loaded": False,
    }
    for row in selected_rows:
        metadata["sources"][row["source"]] = metadata["sources"].get(row["source"], 0) + 1
        key = row["source"] + ":" + row["task"]
        metadata["tasks"][key] = metadata["tasks"].get(key, 0) + 1
    for row in update_rows:
        for item in row:
            if item["id"] not in rows_by_id:
                raise ValueError("non-TRAIN row selected")
    return train, update_rows, observations, banks, query_map, metadata, fit_result


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "RUNNING",
        "experiment_id": CONFIG["id"],
        "command": sys.argv,
        "source_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "started_utc": datetime.now(UTC).isoformat(),
        "training_performed": False,
        "optimizer_updates": 0,
        "validation_loaded": False,
        "calibration_loaded": False,
        "sealed_audit_loaded": False,
        "cloud_cost": 0,
        "errors": [],
    }
    write(output / "result.json", result)
    start_time = time.perf_counter()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("pinned CUDA runtime required for the diagnostic")
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        gpu_samples = []
        for index in range(3):
            gpu_usage = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
            used_mib, utilization = [int(value.strip()) for value in gpu_usage.split(",")]
            gpu_samples.append(
                {
                    "sample": index + 1,
                    "memory_used_mib": used_mib,
                    "utilization_percent": utilization,
                }
            )
            if index < 2:
                time.sleep(2)
        free_memory, total_memory = torch.cuda.mem_get_info()
        result["gpu_preflight"] = {
            "samples": gpu_samples,
            "mean_memory_used_mib": sum(item["memory_used_mib"] for item in gpu_samples) / 3,
            "mean_utilization_percent": sum(item["utilization_percent"] for item in gpu_samples)
            / 3,
            "torch_free_memory_bytes": free_memory,
            "torch_total_memory_bytes": total_memory,
        }
        write(output / "result.json", result)
        if (
            max(item["memory_used_mib"] for item in gpu_samples) > 512
            or max(item["utilization_percent"] for item in gpu_samples) > 5
            or free_memory < 8 * 2**30
        ):
            raise RuntimeError("GPU resource preflight failed")
        if shutil.disk_usage(output).free < 2 * 2**30:
            raise RuntimeError("free disk space is below the preserved 2 GiB reserve")
        train, update_rows, observations, banks, query_map, accounting, fit_result = (
            load_training_inputs(CONFIG)
        )
        base = NativeCandidateReadout()
        checkpoint_path = Path(CONFIG["baseline_fit"]) / CONFIG["selected_checkpoint"]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("step") != CONFIG["selected_step"]:
            raise ValueError("selected checkpoint state step differs")
        base.load_state_dict(checkpoint["model"], strict=True)
        model = base.to(device)
        model.eval()
        torch.cuda.reset_peak_memory_stats()
        named_parameters = [(name, parameter) for name, parameter in model.named_parameters()]
        if sum(parameter.numel() for _, parameter in named_parameters) != 889345:
            raise ValueError("unexpected trainable/readout parameter count")

        def shared_mask(name):
            return not name.startswith("projections.")

        def private_mask(name):
            return name.startswith("projections.")

        def all_parameters(_name):
            return True

        before_state = state_digest(model)
        store = FeatureStore(observations, banks, query_map, device=str(device))
        clip_norm = CONFIG["optimizer_clip_norm"]
        update_records = []
        write(
            output / "window-accounting.json",
            accounting,
        )
        write(output / "sample-ids.json", [[row["id"] for row in batch] for batch in update_rows])
        progress = dict(status="RUNNING", completed_updates=0, total_updates=len(update_rows))
        write(output / "progress.json", progress)
        for position, rows in enumerate(update_rows):
            gradients, shared_gradients, private_gradients = {}, {}, {}
            losses, correct = {}, {}
            for modality in MODALITIES:
                modality_rows = [row for row in rows if row["modality"] == modality]
                losses_for_modality = []
                correct_for_modality = 0
                for row in modality_rows:
                    blocks, queries = store.get(row)
                    permutation = option_order(
                        row, position + (CONFIG["selected_step"] - len(update_rows)) + 1
                    )
                    target = permutation.index(row["target_index"])
                    logits = model(blocks, queries[permutation])
                    loss = F.cross_entropy(
                        logits.unsqueeze(0), torch.tensor([target], device=device)
                    )
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite fixed-TRAIN loss")
                    losses_for_modality.append(loss)
                    correct_for_modality += int(logits.detach().argmax().item() == target)
                objective = torch.stack(losses_for_modality).mean()
                grads = torch.autograd.grad(
                    objective,
                    [parameter for _, parameter in named_parameters],
                    allow_unused=True,
                )
                gradients[modality] = flatten_gradients(named_parameters, grads, all_parameters)
                shared_gradients[modality] = flatten_gradients(named_parameters, grads, shared_mask)
                private_gradients[modality] = flatten_gradients(
                    named_parameters, grads, private_mask
                )
                losses[modality] = float(objective.detach().item())
                correct[modality] = correct_for_modality / len(modality_rows)
            metrics = update_gradient_metrics(
                gradients, shared_gradients, private_gradients, clip_norm
            )
            metrics.update(
                update=CONFIG["selected_step"] - len(update_rows) + position + 1,
                losses=losses,
                train_accuracy=correct,
                modality_sample_ids={
                    modality: [row["id"] for row in rows if row["modality"] == modality]
                    for modality in MODALITIES
                },
            )
            update_records.append(metrics)
            if (position + 1) % 16 == 0:
                write(
                    output / "progress.json",
                    dict(
                        status="RUNNING",
                        completed_updates=position + 1,
                        total_updates=len(update_rows),
                    ),
                )
                if torch.cuda.max_memory_allocated() > 6 * 2**30:
                    raise RuntimeError("diagnostic exceeded the 6 GiB CUDA allocation ceiling")
                if psutil.Process().memory_info().rss > 10 * 2**30:
                    raise RuntimeError("diagnostic exceeded the 10 GiB process RSS ceiling")
                if shutil.disk_usage(output).free < 2 * 2**30:
                    raise RuntimeError("free disk space fell below the preserved 2 GiB reserve")
            if (position + 1) % 32 == 0:
                print(f"measured {position + 1}/{len(update_rows)} fixed TRAIN updates", flush=True)
        after_state = state_digest(model)
        if before_state != after_state or any(
            parameter.grad is not None for _, parameter in named_parameters
        ):
            raise ValueError("diagnostic modified model weights or persistent parameter gradients")
        summary = summarize_updates(update_records)
        result.update(
            status="PASS_NATIVE_MODALITY_GRADIENT_DIAGNOSTIC",
            completed_utc=datetime.now(UTC).isoformat(),
            elapsed_seconds=time.perf_counter() - start_time,
            baseline={
                "source_commit": fit_result["source_commit"],
                "training_updates": fit_result["step"],
                "selected_step": fit_result["best_step"],
                "checkpoint_sha256": CONFIG["selected_checkpoint_sha256"],
                "source_module_hashes": CONFIG["source_file_sha256"],
            },
            runtime={
                "python": sys.version,
                "torch": torch.__version__,
                "numpy": np.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(),
                "dtype": "float32",
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "gpu_preflight": result["gpu_preflight"],
                "model_state_sha256_before": before_state,
                "model_state_sha256_after": after_state,
                "trainable_parameters": sum(parameter.numel() for _, parameter in named_parameters),
            },
            training_window=accounting,
            gradient_summary=summary,
            validation_rows_loaded=0,
            validation_labels_loaded=False,
            validation_features_selected=False,
            sealed_audit_loaded=False,
            optimizer_updates=0,
            checkpoint_selection_performed=False,
            calibration_performed=False,
            cloud_cost=0,
        )
        write(output / "updates.json", update_records)
        write(output / "result.json", result)
        write(
            output / "progress.json",
            dict(
                status="COMPLETE",
                completed_updates=len(update_rows),
                total_updates=len(update_rows),
            ),
        )
    except Exception as error:
        result["status"] = "FAILED"
        result["elapsed_seconds"] = time.perf_counter() - start_time
        result["errors"] = [str(error)]
        write(output / "result.json", result)
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.output)
