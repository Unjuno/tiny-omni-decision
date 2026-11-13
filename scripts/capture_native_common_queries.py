"""Encode only missing native question/candidate pairs with the pinned language profile."""

import argparse
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scripts.capture_native_language import state_digest, verify, write
from scripts.native_common_queries import query_schedule, validate_vectors
from scripts.native_video_cache import sha
from tiny_omni_decision.native_language import candidate_vector

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/pretrained_reuse/native_common_queries_v1.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["smoke", "cache", "replay"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-sha256")
    args = parser.parse_args()
    if (args.phase != "smoke") != bool(args.reference) or bool(args.reference) != bool(
        args.reference_sha256
    ):
        raise ValueError("cache needs smoke receipt, replay needs cache receipt and SHA")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO).strip():
        raise ValueError("clean committed source required")
    usage = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    memory, utilization = [int(v.strip()) for v in usage.split(",")]
    if memory > 512 or utilization > 5:
        raise RuntimeError("local GPU occupied; do not compete")
    args.output.mkdir(exist_ok=False)
    started = time.perf_counter()
    result = dict(
        status="RUNNING",
        phase=args.phase,
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        config_sha256=sha(CONFIG),
        command=sys.argv,
        started_utc=datetime.now(UTC).isoformat(),
        completed=0,
        optimizer_updates=0,
        quality_evaluation=False,
        calibration_loaded=False,
        sealed_audit_loaded=False,
        validation_question_metadata_loaded=True,
        validation_queries_encoded=args.phase != "smoke",
        cloud_cost=0,
        gpu_preflight=usage,
    )
    write(args.output / "result.json", result)
    try:
        config = json.loads(CONFIG.read_text())
        pins = dict(config["input_pins"])
        pins[str(CONFIG)] = sha(CONFIG)
        names = subprocess.check_output(
            ["git", "ls-files", "-z", "--", "*.py"], cwd=REPO, text=True
        ).split("\0")
        pins.update({str(REPO / name): sha(REPO / name) for name in names if name})
        verify(pins)
        root = Path(config["corpus"])
        corpus = json.loads((root / "result.json").read_text())
        if corpus["status"] != "PASS_COMPLETE_NATIVE_COMMON_CORPUS":
            raise ValueError("complete source corpus required")
        pins.update({str(root / name): digest for name, digest in corpus["artifacts"].items()})
        verify(pins)
        pairs = json.loads((root / "pairs.json").read_text(encoding="utf-8"))
        mapping = json.loads((root / "query-map.json").read_text())
        train_keys = set()
        with (root / "train.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["split"] != "train":
                    raise ValueError("TRAIN membership differs")
                train_keys.update(row["queries"])
        schedule = query_schedule(pairs, mapping, train_keys)
        if len(schedule["all"]) != config["expected_new_pairs"]:
            raise ValueError("new query count differs")
        reference = None
        if args.reference:
            if sha(args.reference / "result.json") != args.reference_sha256:
                raise ValueError("reference hash changed")
            reference = json.loads((args.reference / "result.json").read_text())
            expected = "PASS_NATIVE_COMMON_QUERIES_" + (
                "SMOKE" if args.phase == "cache" else "CACHE"
            )
            if (
                reference["status"] != expected
                or reference["config_sha256"] != sha(CONFIG)
                or reference["validated_hashes"] != pins
            ):
                raise ValueError("same source/config and complete reference required")
            verify(
                {
                    str(args.reference / name): digest
                    for name, digest in reference["artifacts"].items()
                }
            )
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.manual_seed(17)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()
        prior = json.loads(Path(config["language_receipt"]).read_text())
        tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True)
        language, loading = AutoModelForSequenceClassification.from_pretrained(
            config["model_path"],
            local_files_only=True,
            dtype=torch.float32,
            attn_implementation="eager",
            output_loading_info=True,
        )
        if any(loading.get(k) for k in ["missing_keys", "unexpected_keys", "mismatched_keys"]):
            raise ValueError("nonexact language model load")
        language.eval().requires_grad_(False).cuda()
        before = state_digest(language)
        runtime = {
            inspect.getfile(type(obj)): sha(Path(inspect.getfile(type(obj))))
            for obj in (language.deberta, tokenizer)
        }
        versions = {
            p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "numpy", "safetensors", "tokenizers")
        }
        if (
            before != prior["initial_model_state_sha256"]
            or runtime != prior["runtime_implementations"]
            or versions != prior["versions"]
        ):
            raise ValueError("original frozen language state/runtime differs")
        result.update(
            validated_hashes=pins,
            initial_model_state_sha256=before,
            runtime_implementations=runtime,
            versions=versions,
            device=torch.cuda.get_device_name(),
            python=sys.version,
            cuda=torch.version.cuda,
            dtype="float32",
            attention="eager",
            batch_size=1,
            tf32=False,
        )
        old_key = schedule["reference"]
        old_bank = np.load(config["language_queries"], mmap_mode="r", allow_pickle=False)
        encoded = candidate_vector(language, tokenizer, *pairs[old_key]).cpu().numpy()
        np.testing.assert_array_equal(encoded, old_bank[mapping[old_key]["offset"]])
        result["existing_train_pair_max_abs"] = 0.0
        keys = schedule["smoke"] if args.phase == "smoke" else schedule["all"]
        write(args.output / "keys.json", keys)
        write(args.output / "effective-config.json", config)
        output = np.empty((len(keys), 384), dtype=np.float32)
        saved = (
            None
            if args.phase == "smoke"
            else np.load(args.reference / "queries.npy", allow_pickle=False)
        )
        prior_keys = (
            [] if args.phase == "smoke" else json.loads((args.reference / "keys.json").read_text())
        )
        fixed = {key: i for i, key in enumerate(prior_keys)}

        def guard():
            memory = psutil.Process().memory_info()
            result["peak_working_set_bytes"] = memory.peak_wset
            if (
                time.perf_counter() - started > 3600
                or memory.peak_wset > 6 * 2**30
                or torch.cuda.max_memory_allocated() > 3 * 2**30
                or shutil.disk_usage(args.output).free < 2 * 2**30
            ):
                raise RuntimeError("registered resource budget exceeded")

        guard()
        for i, key in enumerate(keys):
            value = candidate_vector(language, tokenizer, *pairs[key]).cpu().numpy()
            if value.shape != (384,) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError("invalid encoded query vector")
            if key in fixed:
                np.testing.assert_array_equal(value, saved[fixed[key]])
            elif args.phase == "replay":
                raise ValueError("missing full replay query")
            output[i] = value
            result["completed"] += 1
            if (i + 1) % 256 == 0:
                guard()
                write(args.output / "result.json", result)
                print("queries", i + 1, "/", len(keys), flush=True)
        validate_vectors(output, len(keys))
        np.save(args.output / "queries.npy", output, allow_pickle=False)
        np.testing.assert_array_equal(
            output, np.load(args.output / "queries.npy", allow_pickle=False)
        )
        after = state_digest(language)
        if before != after:
            raise ValueError("frozen language weights changed")
        verify(pins)
        guard()
        result.update(
            status="PASS_NATIVE_COMMON_QUERIES_" + args.phase.upper(),
            expected=len(keys),
            final_model_state_sha256=after,
            reference_sha256=args.reference_sha256,
            replay_max_abs=0.0 if args.phase == "replay" else None,
            peak_vram_bytes=torch.cuda.max_memory_allocated(),
            peak_vram_reserved_bytes=torch.cuda.max_memory_reserved(),
            artifacts={p.name: sha(p) for p in args.output.iterdir() if p.name != "result.json"},
        )
    except BaseException as exc:
        result.update(status="INCOMPLETE", failure=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        result.update(
            seconds=time.perf_counter() - started, finished_utc=datetime.now(UTC).isoformat()
        )
        write(args.output / "result.json", result)
    print(result["status"], result["seconds"], flush=True)


if __name__ == "__main__":
    main()
