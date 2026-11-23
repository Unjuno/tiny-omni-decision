"""Capture a pinned BERT overlay for the existing native Path A feature corpus."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import psutil
import torch
from transformers import AutoModel, AutoTokenizer

from scripts.native_bert_feature_cache import build_plan
from scripts.native_language_cache import inventory
from scripts.native_training_readiness import TEXT
from scripts.native_video_cache import sha
from tiny_omni_decision.native_language import candidate_vector, observation_tokens

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/pretrained_reuse/path_a_line_distilbert_features_v1.json"
EXPECTED_MLM_HEAD = {
    "vocab_transform.weight",
    "vocab_transform.bias",
    "vocab_layer_norm.weight",
    "vocab_layer_norm.bias",
    "vocab_projector.weight",
    "vocab_projector.bias",
}


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def digest_model(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_inputs(config):
    root = Path(config["corpus_root"])
    receipt_path = root / "result.json"
    if sha(receipt_path) != config["corpus_result_sha256"]:
        raise ValueError("common-corpus receipt changed")
    receipt = read_json(receipt_path)
    if receipt.get("status") != "PASS_COMPLETE_NATIVE_COMMON_CORPUS":
        raise ValueError("completed native common corpus required")
    for name, expected in config["corpus_artifact_hashes"].items():
        path = root / name
        if sha(path) != expected or receipt["artifacts"].get(name) != expected:
            raise ValueError("pinned common-corpus artifact changed: " + name)

    files = {
        name: [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
        for name, path in config["source_files"].items()
    }
    index = inventory(files)
    state_splits = {
        split: {
            row["state"]
            for name in names
            for row in index["rows"][name]
        }
        for split, names in TEXT.items()
    }
    observations = read_json(root / "observations.json")
    text_observation_ids = {
        key for key, entry in observations.items() if entry.get("source") == "text"
    }
    pairs = read_json(root / "pairs.json")
    query_map = read_json(root / "query-map.json")
    plan = build_plan(
        state_texts=index["states"],
        state_splits=state_splits,
        pairs=pairs,
        query_map=query_map,
        text_observation_ids=text_observation_ids,
        tokenizer=AutoTokenizer.from_pretrained(
            config["model_path"], local_files_only=True, trust_remote_code=True
        ),
        width=config["hidden_width"],
        max_tokens=config["max_tokens"],
    )
    if (
        plan["state_count"] != config["expected_unique_text_states"]
        or plan["query_count"] != config["expected_all_query_pairs"]
    ):
        raise ValueError("pinned state or query count differs")
    return index, observations, pairs, query_map, plan


def check_gpu():
    usage = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    used, utilization = [int(value.strip()) for value in usage.split(",")]
    if used > 512 or utilization > 5:
        raise RuntimeError("local GPU occupied; refusing to compete")
    if not torch.cuda.is_available():
        raise RuntimeError("local CUDA device unavailable")
    return usage


def load_encoder(config):
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_path"], local_files_only=True, trust_remote_code=True
    )
    model, loading = AutoModel.from_pretrained(
        config["model_path"],
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.float32,
        attn_implementation="eager",
        output_loading_info=True,
    )
    if (
        loading.get("missing_keys")
        or loading.get("mismatched_keys")
        or set(loading.get("unexpected_keys", ())) != EXPECTED_MLM_HEAD
    ):
        raise ValueError("checkpoint encoder tensors did not load exactly")
    if len(model.transformer.layer) != 6:
        raise ValueError("expected the pinned six-layer source checkpoint")
    model.transformer.layer = torch.nn.ModuleList(list(model.transformer.layer[:4]))
    model.config.n_layers = config["retained_layers"]
    model.eval().requires_grad_(False).to("cuda")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != config["expected_parameters"] or any(
        parameter.requires_grad for parameter in model.parameters()
    ):
        raise ValueError("four-layer frozen parameter contract changed")
    return model, tokenizer, loading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["smoke", "cache"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-sha256")
    args = parser.parse_args()
    if (args.phase == "smoke") != (args.reference is None) or bool(args.reference) != bool(
        args.reference_sha256
    ):
        raise ValueError("cache requires the verified smoke receipt")
    output = args.output.resolve()
    if output == REPO or REPO in output.parents:
        raise ValueError("feature artifacts must be stored outside the Git worktree")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO).strip():
        raise ValueError("clean committed source required")
    gpu_preflight = check_gpu()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result = {
        "status": "RUNNING",
        "phase": args.phase,
        "command": sys.argv,
        "pid": os.getpid(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "config_sha256": sha(CONFIG),
        "started_utc": datetime.now(UTC).isoformat(),
        "gpu_preflight": gpu_preflight,
        "optimizer_updates": 0,
        "quality_evaluated": False,
        "sealed_audit_loaded": False,
        "cloud_cost": 0,
        "completed_states": 0,
        "completed_queries": 0,
    }
    write(args.output / "result.json", result)
    banks = {}
    try:
        config = read_json(CONFIG)
        pins = dict(config["input_pins"])
        pins.update(
            {
                str(Path(config["model_path"]) / name): value
                for name, value in config["model_files"].items()
            }
        )
        corpus_root = Path(config["corpus_root"])
        pins[str(corpus_root / "result.json")] = config["corpus_result_sha256"]
        pins.update(
            {
                str(corpus_root / name): digest
                for name, digest in config["corpus_artifact_hashes"].items()
            }
        )
        names = subprocess.check_output(
            ["git", "ls-files", "-z", "--", "*.py"],
            cwd=REPO,
            text=True,
        ).split("\0")
        pins.update({str(REPO / name): sha(REPO / name) for name in names if name})
        pins[str(CONFIG)] = sha(CONFIG)
        for path, expected in pins.items():
            if sha(Path(path)) != expected:
                raise ValueError("pinned input or source changed: " + path)
        write(args.output / "effective-config.json", config)

        reference = None
        if args.reference:
            if sha(args.reference / "result.json") != args.reference_sha256:
                raise ValueError("smoke receipt hash changed")
            reference = read_json(args.reference / "result.json")
            if (
                reference.get("status") != "PASS_NATIVE_BERT_FEATURES_SMOKE"
                or reference.get("config_sha256") != result["config_sha256"]
                or reference.get("validated_hashes") != pins
            ):
                raise ValueError("matching successful smoke receipt required")
            for name, expected in reference["artifact_hashes"].items():
                if sha(args.reference / name) != expected:
                    raise ValueError("smoke artifact changed: " + name)
            result["smoke_reference"] = {
                "path": str(args.reference),
                "result_sha256": args.reference_sha256,
            }

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.manual_seed(config["seed"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()
        index, observations, pairs, query_map, plan = build_inputs(config)
        write(args.output / "index.json", plan)
        runtime = config["runtime"]
        needed = plan["tensor_bytes"] if args.phase == "cache" else 0
        if shutil.disk_usage(output).free < needed + runtime["disk_reserve_bytes"]:
            raise RuntimeError("insufficient free disk for planned arrays and reserve")

        model, tokenizer, loading = load_encoder(config)
        before = digest_model(model)
        result.update(
            validated_hashes=pins,
            model_id=config["model_id"],
            model_revision=config["model_revision"],
            model_parameters=config["expected_parameters"],
            trainable_parameters=0,
            initial_model_state_sha256=before,
            device=torch.cuda.get_device_name(),
            python=sys.version,
            cuda=torch.version.cuda,
            versions={
                package: importlib.metadata.version(package)
                for package in (
                    "torch",
                    "transformers",
                    "numpy",
                    "psutil",
                    "safetensors",
                    "tokenizers",
                )
            },
            dtype=runtime["dtype"],
            attention=runtime["attention"],
            batch_size=runtime["batch_size"],
            tf32=runtime["tf32"],
            discarded_pretraining_head_keys=sorted(loading["unexpected_keys"]),
            unique_text_states=plan["state_count"],
            unique_query_pairs=plan["query_count"],
            feature_bytes=plan["tensor_bytes"],
            state_order_sha256=plan["state_order_sha256"],
            query_order_sha256=plan["query_order_sha256"],
            longest_state_tokens=plan["maximum_state_tokens"],
            longest_query_tokens=plan["maximum_query_tokens"],
        )
        result["runtime_implementations"] = {
            inspect.getfile(type(obj)): sha(Path(inspect.getfile(type(obj))))
            for obj in (model, tokenizer)
        }
        if reference and any(
            reference[key] != result[key]
            for key in ("versions", "runtime_implementations", "initial_model_state_sha256")
        ):
            raise ValueError("model/runtime differs from smoke")

        state_keys = list(plan["states"])
        query_keys = list(plan["queries"])
        if args.phase == "smoke":
            train_keys = [key for key in state_keys if plan["states"][key]["split"] == "train"]
            longest_state = max(train_keys, key=lambda key: plan["states"][key]["tokens"])
            longest_query = max(query_keys, key=lambda key: plan["queries"][key]["tokens"])
            state_keys = list(dict.fromkeys([*train_keys[:3], longest_state]))
            query_keys = list(dict.fromkeys([*query_keys[:7], longest_query]))
            shapes = {
                "smoke-train-states.npy": [
                    sum(
                        plan["states"][key]["tokens"]
                        for key in state_keys
                        if plan["states"][key]["split"] == "train"
                    ),
                    config["hidden_width"],
                ],
                "smoke-queries.npy": [len(query_keys), config["hidden_width"]],
            }
            train_offsets, offset = {}, 0
            for key in state_keys:
                if plan["states"][key]["split"] == "train":
                    train_offsets[key] = offset
                    offset += plan["states"][key]["tokens"]
            state_output = args.output / "smoke-train-states.npy"
            query_output = args.output / "smoke-queries.npy"
            state_bank = np.lib.format.open_memmap(
                state_output, mode="w+", dtype=np.float32, shape=tuple(shapes[state_output.name])
            )
            query_bank = np.lib.format.open_memmap(
                query_output, mode="w+", dtype=np.float32, shape=tuple(shapes[query_output.name])
            )
        else:
            shapes = plan["shapes"]
            for name, shape in shapes.items():
                banks[name] = np.lib.format.open_memmap(
                    args.output / name, mode="w+", dtype=np.float32, shape=tuple(shape)
                )

        write(args.output / "state-order.json", state_keys)
        write(args.output / "query-order.json", query_keys)
        phase_start = time.perf_counter()
        segments = {}

        def guard():
            memory = psutil.Process().memory_info()
            result["peak_working_set_bytes"] = memory.peak_wset
            if (
                time.perf_counter() - started > runtime["max_wall_seconds"]
                or shutil.disk_usage(output).free < runtime["disk_reserve_bytes"]
                or torch.cuda.max_memory_allocated() > runtime["max_vram_bytes"]
                or memory.peak_wset > runtime["max_rss_bytes"]
            ):
                raise RuntimeError("registered feature-capture resource budget exceeded")

        for key in state_keys:
            if args.phase == "cache":
                meta = plan["states"][key]
                target = banks[meta["file"]]
                selection = slice(meta["offset"], meta["offset"] + meta["tokens"])
            else:
                if plan["states"][key]["split"] != "train":
                    continue
                meta = plan["states"][key]
                target = state_bank
                selection = slice(train_offsets[key], train_offsets[key] + meta["tokens"])
            hidden, spans = observation_tokens(model, tokenizer, index["states"][key])
            value = hidden.detach().cpu().numpy()
            if value.shape != (meta["tokens"], config["hidden_width"]):
                raise ValueError("state feature token coverage differs")
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError("nonfinite or non-FP32 state features")
            target[selection] = value
            segments[key] = spans
            result["completed_states"] += 1
            if result["completed_states"] % 128 == 0:
                guard()
                write(args.output / "result.json", result)
                print("states", result["completed_states"], "/", len(state_keys), flush=True)
        result["state_encoding_seconds"] = time.perf_counter() - phase_start
        phase_start = time.perf_counter()

        for position, key in enumerate(query_keys):
            if args.phase == "cache":
                offset = plan["queries"][key]["offset"]
                target = banks["queries.npy"]
            else:
                offset = position
                target = query_bank
            vector = candidate_vector(model, tokenizer, *pairs[key]).detach().cpu().numpy()
            if (
                vector.shape != (config["hidden_width"],)
                or vector.dtype != np.float32
                or not np.isfinite(vector).all()
            ):
                raise ValueError("invalid query feature")
            target[offset] = vector
            result["completed_queries"] += 1
            if result["completed_queries"] % 256 == 0:
                guard()
                write(args.output / "result.json", result)
                print("queries", result["completed_queries"], "/", len(query_keys), flush=True)
        result["query_encoding_seconds"] = time.perf_counter() - phase_start

        if args.phase == "smoke":
            state_bank.flush()
            query_bank.flush()
            del state_bank, query_bank
            for name in ("smoke-train-states.npy", "smoke-queries.npy"):
                path = args.output / name
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                if not np.isfinite(array).all():
                    raise ValueError("smoke reload contained nonfinite values")
            result["smoke_reload_pass"] = True
            write(args.output / "segments.json", segments)
        else:
            for bank in banks.values():
                bank.flush()
            banks.clear()
            for name, shape in shapes.items():
                array = np.load(args.output / name, mmap_mode="r", allow_pickle=False)
                if array.shape != tuple(shape) or array.dtype != np.float32:
                    raise ValueError("saved feature-bank shape or dtype differs")
                for offset in range(0, len(array), 1024):
                    if not np.isfinite(array[offset : offset + 1024]).all():
                        raise ValueError("saved feature bank contains nonfinite values")
            write(args.output / "segments.json", segments)

        after = digest_model(model)
        if before != after:
            raise ValueError("frozen BERT weights changed during feature capture")
        for path, expected in pins.items():
            if sha(Path(path)) != expected:
                raise ValueError("pinned input changed during run: " + path)
        guard()
        result.update(
            status="PASS_NATIVE_BERT_FEATURES_" + args.phase.upper(),
            final_model_state_sha256=after,
            state_encoding_seconds=result.get("state_encoding_seconds"),
            query_encoding_seconds=result.get("query_encoding_seconds"),
            peak_vram_bytes=torch.cuda.max_memory_allocated(),
            peak_vram_reserved_bytes=torch.cuda.max_memory_reserved(),
            finished_utc=datetime.now(UTC).isoformat(),
            seconds=time.perf_counter() - started,
            artifact_hashes={
                path.name: sha(path)
                for path in args.output.iterdir()
                if path.is_file() and path.name != "result.json"
            },
        )
        write(args.output / "result.json", result)
    except BaseException as error:
        result.update(
            status="INCOMPLETE",
            error=repr(error),
            traceback=traceback.format_exc(),
            finished_utc=datetime.now(UTC).isoformat(),
            seconds=time.perf_counter() - started,
        )
        write(args.output / "result.json", result)
        raise
    print(result["status"], result["seconds"], flush=True)


if __name__ == "__main__":
    main()
