"""All actually consumed TRAIN questions, immutable endpoints, no optimizer."""

import argparse
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

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import psutil
import torch

from scripts.native_consumed_evaluation import consumed_plan
from scripts.native_fit import summarize
from scripts.native_fit_runtime import FeatureStore
from scripts.native_video_cache import sha
from scripts.probe_native_state_dependence import model_digest, read, verify, write
from tiny_omni_decision.native_candidate import NativeCandidateReadout
from tiny_omni_decision.native_temporal import TemporalNativeCandidate

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/pretrained_reuse/native_consumed_evaluation_v1.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["smoke", "probe", "replay"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-sha256")
    args = parser.parse_args()
    if (args.phase != "smoke") != bool(args.reference) or bool(args.reference) != bool(
        args.reference_sha256
    ):
        raise ValueError("probe/replay require completed prior receipt and SHA")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO).strip():
        raise ValueError("clean committed source required")
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    memory, use = [int(s.strip()) for s in gpu.split(",")]
    if memory > 512 or use > 5:
        raise RuntimeError("GPU occupied; do not compete")
    args.output.mkdir(exist_ok=False)
    started = time.perf_counter()
    report = dict(
        status="RUNNING",
        phase=args.phase,
        command=sys.argv,
        pid=os.getpid(),
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        started_utc=datetime.now(UTC).isoformat(),
        config_sha256=sha(CONFIG),
        model_updates=0,
        validation_inference=False,
        sealed_audit_loaded=False,
        cloud_cost=0,
        gpu_preflight=gpu,
    )
    write(args.output / "result.json", report)
    try:
        config = read(CONFIG)
        shutil.copyfile(CONFIG, args.output / "effective-config.json")
        pins = dict(config["pins"])
        pins[str(CONFIG)] = sha(CONFIG)
        names = subprocess.check_output(
            ["git", "ls-files", "-z", "--", "*.py"], cwd=REPO, text=True
        ).split("\0")
        pins.update({str(REPO / name): sha(REPO / name) for name in names if name})
        verify(pins)
        corpus, queries = Path(config["corpus"]), Path(config["queries"])
        cr, qr = read(corpus / "result.json"), read(queries / "result.json")
        if (
            cr["status"] != "PASS_COMPLETE_NATIVE_COMMON_CORPUS"
            or qr["status"] != "PASS_NATIVE_COMMON_QUERIES_CACHE"
        ):
            raise ValueError("complete source caches required")
        for name in ("train.jsonl", "observations.json", "banks.json", "query-map.json"):
            pins[str(corpus / name)] = cr["artifacts"][name]
        histories, models = {}, config["models"]
        for name, entry in models.items():
            fit = Path(entry["fit"])
            receipt = read(fit / "result.json")
            for runtime in (
                "scripts/native_fit_runtime.py",
                "scripts/native_candidate_inputs.py",
                "src/tiny_omni_decision/native_candidate.py",
                "src/tiny_omni_decision/native_temporal.py",
            ):
                path = str(REPO / runtime)
                if pins[path] != receipt["validated_hashes"][path]:
                    raise ValueError("original model/feature implementation changed")
            if receipt["status"] not in (
                "PASS_NATIVE_FIT_TRAIN",
                "PASS_NATIVE_MATCHED_BATCH_TRAIN",
            ):
                raise ValueError("completed immutable fit required")
            for file in ("training-history.jsonl", entry["checkpoint"]):
                pins[str(fit / file)] = receipt["artifacts"][file]
            history = [
                json.loads(s) for s in (fit / "training-history.jsonl").read_text().splitlines()
            ]
            if len(history) != entry["fit_steps"] or receipt["step"] != entry["fit_steps"]:
                raise ValueError("full fixed-budget histories required")
            histories[name] = history
        verify(pins)
        all_rows = [
            json.loads(s) for s in (corpus / "train.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        plan = consumed_plan(all_rows, histories)
        if (len(all_rows), plan["occurrence_count"], plan["unique_count"]) != (100580, 8192, 6656):
            raise ValueError("registered complete consumption differs")
        row_map = {r["id"]: r for r in all_rows}
        rows = [row_map[i] for i in plan["ids"]]
        del row_map, all_rows
        write(args.output / "plan.json", plan)
        selected = list(range(len(rows)))
        if args.phase == "smoke":
            selected = [
                i
                for m in ("text", "image", "audio", "video")
                for i in [j for j, r in enumerate(rows) if r["modality"] == m][:8]
            ]
        evaluated = [rows[i] for i in selected]
        write(args.output / "evaluated-ids.json", [r["id"] for r in evaluated])
        obs = read(corpus / "observations.json")
        obs = {k: obs[k] for k in sorted({r["observation"] for r in evaluated})}
        if any(v["split"] != "train" for v in obs.values()):
            raise ValueError("TRAIN native observations required")
        all_banks, mapping = read(corpus / "banks.json"), read(corpus / "query-map.json")
        mapping = {k: mapping[k] for k in sorted({k for r in evaluated for k in r["queries"]})}
        names = {f["bank"] for o in obs.values() for f in o["features"]} | {
            v["bank"] for v in mapping.values()
        }
        banks = {
            n: dict(path=str(queries / "queries.npy"), sha256=qr["artifacts"]["queries.npy"])
            if n == "new/queries.npy"
            else all_banks[n]
            for n in names
        }
        pins.update({v["path"]: v["sha256"] for v in banks.values()})
        verify(pins)
        reference = None
        if args.reference:
            if sha(args.reference / "result.json") != args.reference_sha256:
                raise ValueError("reference hash differs")
            reference = read(args.reference / "result.json")
            expected = "PASS_NATIVE_CONSUMED_" + ("SMOKE" if args.phase == "probe" else "PROBE")
            if reference["status"] != expected or reference["config_sha256"] != sha(CONFIG):
                raise ValueError("matching completed reference required")
            verify({str(args.reference / n): h for n, h in reference["artifacts"].items()})
            if read(args.reference / "plan.json") != plan:
                raise ValueError("complete frozen TRAIN plan changed")
            # Smoke has a subset of banks; its shared pins must still match.
            if any(pins.get(k) != v for k, v in reference["validated_hashes"].items()):
                raise ValueError("source/input pins changed")
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.manual_seed(17)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.reset_peak_memory_stats()
        store = FeatureStore(obs, banks, mapping)
        report.update(
            validated_hashes=pins,
            evaluated_questions=len(evaluated),
            full_plan_questions=len(rows),
            versions={p: importlib.metadata.version(p) for p in ("torch", "numpy", "psutil")},
            python=sys.version,
            cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(),
            dtype="float32",
            tf32=False,
            models={},
        )

        def guard():
            memory = psutil.Process().memory_info().peak_wset
            report["peak_working_set_bytes"] = memory
            if (
                time.perf_counter() - started > 1800
                or memory > 10 * 2**30
                or torch.cuda.max_memory_allocated() > 4 * 2**30
                or shutil.disk_usage(args.output).free < 2 * 2**30
            ):
                raise RuntimeError("fixed inference resource guard exceeded")

        for name, entry in models.items():
            tick = time.perf_counter()
            model = NativeCandidateReadout()
            if entry["variant"] == "lstm":
                model = TemporalNativeCandidate(model)
            checkpoint = torch.load(
                Path(entry["fit"]) / entry["checkpoint"], map_location="cpu", weights_only=True
            )
            if entry["checkpoint"] == "initial.pt":
                model.load_state_dict(checkpoint, strict=True)
            else:
                if checkpoint["step"] != entry["fit_steps"]:
                    raise ValueError("fixed endpoint required, not selected weights")
                model.load_state_dict(checkpoint["model"], strict=True)
            del checkpoint
            model = model.cuda().eval().requires_grad_(False)
            before = model_digest(model)
            parameters = sum(p.numel() for p in model.parameters())
            if parameters + 195365529 > 200000000:
                raise ValueError("full prequantization parameter cap exceeded")
            scores = np.full((len(evaluated), 40), -1e30, dtype=np.float32)
            with torch.no_grad():
                for i, row in enumerate(evaluated):
                    blocks, q = store.get(row)
                    scores[i, : len(row["options"])] = model(blocks, q).cpu().numpy()
                    if (i + 1) % 1024 == 0:
                        guard()
                        print(name, i + 1, "/", len(evaluated), flush=True)
            if before != model_digest(model):
                raise ValueError("evaluation changed model tensors")
            if args.phase == "replay":
                np.testing.assert_array_equal(
                    scores, np.load(args.reference / (name + ".npy"), allow_pickle=False)
                )
            np.save(args.output / (name + ".npy"), scores, allow_pickle=False)
            metrics = dict(unique=summarize(evaluated, scores))
            if args.phase != "smoke":
                expanded = np.repeat(np.arange(len(rows)), plan["multiplicities"])
                metrics["occurrence_weighted"] = summarize(
                    [rows[i] for i in expanded], scores[expanded]
                )
            write(args.output / (name + "-metrics.json"), metrics)
            report["models"][name] = dict(
                seconds=time.perf_counter() - tick,
                state_digest=before,
                total_parameters=parameters + 195365529,
                readout_parameters=parameters,
                completed_questions=len(evaluated),
                replay_exact=args.phase == "replay",
            )
            write(args.output / "result.json", report)
            print(name, "complete", metrics["unique"]["macro"], flush=True)
            del model
        guard()
        verify(pins)
        report.update(
            status="PASS_NATIVE_CONSUMED_" + args.phase.upper(),
            peak_vram_bytes=torch.cuda.max_memory_allocated(),
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
