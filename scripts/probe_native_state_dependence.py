"""Prepare fixed TRAIN input contrasts, run pinned inference, and replay exactly."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from scripts.native_state_dependence import make_plan, summarize
from scripts.native_video_cache import sha

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/pretrained_reuse/native_state_dependence_v1.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def verify(pins):
    for path, expected in pins.items():
        if sha(Path(path)) != expected:
            raise ValueError("changed input/source: " + path)


def model_digest(model):
    value = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["prepare", "smoke", "probe", "replay"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-sha256")
    args = parser.parse_args()
    if (args.phase != "prepare") != bool(args.reference) or bool(args.reference) != bool(
        args.reference_sha256
    ):
        raise ValueError("smoke/probe/replay require a completed prior receipt and SHA")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO).strip():
        raise ValueError("clean committed source required")
    args.output.mkdir(exist_ok=False)
    started = time.perf_counter()
    report = dict(
        status="RUNNING",
        phase=args.phase,
        pid=os.getpid(),
        command=sys.argv,
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        config_sha256=sha(CONFIG),
        started_utc=datetime.now(UTC).isoformat(),
        model_updates=0,
        validation_examples_loaded=False,
        sealed_audit_loaded=False,
        calibration_loaded=False,
        cloud_cost=0,
    )
    write(args.output / "result.json", report)
    try:
        config = read(CONFIG)
        if config["pairs_per_task_and_stratum"] != 8 or config["seed"] != 17:
            raise ValueError("fixed diagnostic selection required")
        pins = dict(config["pins"])
        pins[str(CONFIG)] = sha(CONFIG)
        names = subprocess.check_output(
            ["git", "ls-files", "-z", "--", "*.py"], cwd=REPO, text=True
        ).split("\0")
        pins.update({str(REPO / name): sha(REPO / name) for name in names if name})
        verify(pins)
        fit = Path(config["fit"])
        corpus = Path(config["corpus"])
        queries = Path(config["queries"])
        fit_receipt, corpus_receipt, query_receipt = (
            read(p / "result.json") for p in (fit, corpus, queries)
        )
        if (
            fit_receipt["status"] != "PASS_NATIVE_FIT_TRAIN"
            or fit_receipt["variant"] != "baseline"
            or fit_receipt["step"] != 2048
            or corpus_receipt["status"] != "PASS_COMPLETE_NATIVE_COMMON_CORPUS"
            or query_receipt["status"] != "PASS_NATIVE_COMMON_QUERIES_CACHE"
        ):
            raise ValueError("completed pinned baseline/corpus/query artifacts required")
        for name in ("initial.pt", "final.pt", "training-history.jsonl"):
            pins[str(fit / name)] = fit_receipt["artifacts"][name]
        # Only training rows are read. Shared query/observation metadata can describe
        # development records; no validation row or validation feature bank is read.
        for name in ("train.jsonl", "observations.json", "banks.json", "query-map.json"):
            pins[str(corpus / name)] = corpus_receipt["artifacts"][name]
        for name in (
            "scripts/native_fit_runtime.py",
            "scripts/native_candidate_inputs.py",
            "src/tiny_omni_decision/native_candidate.py",
            "scripts/semantic_state_diagnostic.py",
        ):
            path = str(REPO / name)
            if pins[path] != fit_receipt["validated_hashes"][path]:
                raise ValueError("original readout/feature/metric implementation changed")
        verify(pins)
        reference = None
        if args.reference:
            if sha(args.reference / "result.json") != args.reference_sha256:
                raise ValueError("reference receipt hash differs")
            reference = read(args.reference / "result.json")
            expected = {"smoke": "PREPARE", "probe": "SMOKE", "replay": "PROBE"}[args.phase]
            if reference["status"] != "PASS_NATIVE_STATE_" + expected or reference[
                "config_sha256"
            ] != sha(CONFIG):
                raise ValueError("matching completed prior phase required")
            verify({str(args.reference / name): h for name, h in reference["artifacts"].items()})
        if args.phase == "prepare":
            rows = [json.loads(line) for line in (corpus / "train.jsonl").open(encoding="utf-8")]
            trajectory = [json.loads(line) for line in (fit / "training-history.jsonl").open()]
            if [h["step"] for h in trajectory] != list(range(1, 2049)):
                raise ValueError("complete fixed optimizer history required")
            consumed = {ident for h in trajectory for ident in h["sample_ids"]}
            plan = make_plan(rows, consumed, limit=8, seed=17)
            if not plan["pairs"]:
                raise ValueError("no eligible input contrasts; diagnostic cannot execute")
            report.update(train_rows=len(rows), consumed_questions=len(consumed))
        else:
            plan = read(args.reference / "plan.json")
        write(args.output / "plan.json", plan)
        if reference and sha(args.output / "plan.json") != reference["artifacts"]["plan.json"]:
            raise ValueError("frozen contrast plan differs")
        all_observations = read(corpus / "observations.json")
        used_observations = {key for p in plan["pairs"] for key in p["observations"]}
        observations = {key: all_observations[key] for key in sorted(used_observations)}
        if any(o["split"] != "train" for o in observations.values()):
            raise ValueError("TRAIN observations only")
        bank_names = {f["bank"] for o in observations.values() for f in o["features"]}
        all_banks = read(corpus / "banks.json")
        banks = {key: all_banks[key] for key in sorted(bank_names)}
        all_mapping = read(corpus / "query-map.json")
        used_queries = {key for p in plan["pairs"] for key in p["queries"]}
        mapping = {key: all_mapping[key] for key in sorted(used_queries)}
        for name in {v["bank"] for v in mapping.values()}:
            banks[name] = (
                dict(
                    path=str(queries / "queries.npy"),
                    sha256=query_receipt["artifacts"]["queries.npy"],
                )
                if name == "new/queries.npy"
                else all_banks[name]
            )
        pins.update({v["path"]: v["sha256"] for v in banks.values()})
        verify(pins)
        if reference and reference["validated_hashes"] != pins:
            raise ValueError("reference source/input bindings differ")
        report.update(
            validated_hashes=pins,
            pairs=len(plan["pairs"]),
            observations=len(observations),
            query_pairs=len(mapping),
        )
        write(args.output / "effective-config.json", config)
        if args.phase != "prepare":
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            import psutil
            import torch

            from scripts.native_fit_runtime import FeatureStore
            from tiny_omni_decision.native_candidate import NativeCandidateReadout

            gpu = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
            memory, utilization = [int(v.strip()) for v in gpu.split(",")]
            if memory > 512 or utilization > 5:
                raise RuntimeError("GPU occupied; do not compete")
            torch.set_num_threads(4)
            torch.set_num_interop_threads(1)
            torch.manual_seed(17)
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.cuda.reset_peak_memory_stats()
            model = NativeCandidateReadout().cuda().eval().requires_grad_(False)
            store = FeatureStore(observations, banks, mapping)
            pairs = plan["pairs"]
            if args.phase == "smoke":
                seen = set()
                pairs = [p for p in pairs if p["source"] not in seen and not seen.add(p["source"])]
            write(args.output / "executed-pairs.json", pairs)
            report.update(
                device=torch.cuda.get_device_name(),
                gpu_preflight=gpu,
                versions={k: importlib.metadata.version(k) for k in ("torch", "numpy", "psutil")},
                python=sys.version,
                cuda=torch.version.cuda,
                dtype="float32",
                executed_pairs=len(pairs),
                checkpoints={},
                total_parameters=196254874,
            )
            if (
                reference
                and "versions" in reference
                and reference["versions"] != report["versions"]
            ):
                raise ValueError("reference runtime versions differ")

            def guard():
                report["peak_working_set_bytes"] = psutil.Process().memory_info().peak_wset
                if (
                    time.perf_counter() - started > 1200
                    or torch.cuda.max_memory_allocated() > 2 * 2**30
                    or report["peak_working_set_bytes"] > 8 * 2**30
                    or shutil.disk_usage(args.output).free < 2 * 2**30
                ):
                    raise RuntimeError("registered diagnostic resource budget exceeded")

            for label, filename in (("initial", "initial.pt"), ("endpoint2048", "final.pt")):
                data = torch.load(fit / filename, map_location="cpu", weights_only=True)
                if label == "endpoint2048":
                    if data["step"] != 2048:
                        raise ValueError("fixed final checkpoint required")
                    data = data["model"]
                model.load_state_dict(data, strict=True)
                before = model_digest(model)
                tick = time.perf_counter()
                scores = np.full((len(pairs), 2, 40), -1e30, dtype=np.float32)
                with torch.no_grad():
                    for i, pair in enumerate(pairs):
                        for j, observation in enumerate(pair["observations"]):
                            # Target, task, group and source are deliberately excluded.
                            blocks, q = store.get(
                                dict(observation=observation, queries=pair["queries"])
                            )
                            value = model(blocks, q).cpu().numpy()
                            scores[i, j, : len(value)] = value
                        if (i + 1) % 128 == 0:
                            guard()
                            report.update(checkpoint=label, completed_pairs=i + 1)
                            write(args.output / "result.json", report)
                            print(label, i + 1, "/", len(pairs), flush=True)
                torch.cuda.synchronize()
                after = model_digest(model)
                if before != after:
                    raise ValueError("frozen readout changed")
                if args.phase == "replay":
                    np.testing.assert_array_equal(
                        scores, np.load(args.reference / (label + ".npy"), allow_pickle=False)
                    )
                np.save(args.output / (label + ".npy"), scores, allow_pickle=False)
                report["checkpoints"][label] = dict(
                    sha256=fit_receipt["artifacts"][filename],
                    state_digest=before,
                    state_unchanged=True,
                    seconds=time.perf_counter() - tick,
                    metrics=summarize(dict(plan, pairs=pairs), scores),
                )
            guard()
            report.update(
                peak_vram_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
            if args.phase == "replay":
                report.update(replay_max_abs=0, reference_sha256=args.reference_sha256)
        verify(pins)
        report.update(
            status="PASS_NATIVE_STATE_" + args.phase.upper(),
            artifacts={p.name: sha(p) for p in args.output.iterdir() if p.name != "result.json"},
        )
    except BaseException as exc:
        report.update(status="INCOMPLETE", failure=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report.update(
            seconds=time.perf_counter() - started, finished_utc=datetime.now(UTC).isoformat()
        )
        write(args.output / "result.json", report)
    print(report["status"], report["seconds"], flush=True)


if __name__ == "__main__":
    main()
