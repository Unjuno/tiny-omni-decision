"""TRAIN feature plumbing only: no labels, quality scores, or encoder execution."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch

from scripts.native_candidate_inputs import positions
from tiny_omni_decision.native_candidate import NativeCandidateReadout, NativeTokenBlock
from tiny_omni_decision.native_temporal import TemporalNativeCandidate

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/pretrained_reuse/native_temporal_probe_v1.json"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--replay-sha256")
    args = parser.parse_args()
    if bool(args.replay) != bool(args.replay_sha256):
        raise ValueError("paired replay arguments required")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO).strip():
        raise ValueError("commit source before physical probe")
    config = json.loads(CONFIG.read_text())
    source = Path(config["feature_directory"])
    if sha(source / "result.json") != config["feature_receipt_sha256"]:
        raise ValueError("source feature receipt changed")
    previous = json.loads((source / "result.json").read_text())
    if previous["status"] != "PASS_NATIVE_CANDIDATE_INTEGRATION":
        raise ValueError("passing native source required")
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                            text=True).strip()
    replay = None
    if args.replay:
        if sha(args.replay / "result.json") != args.replay_sha256:
            raise ValueError("replay receipt mismatch")
        replay = json.loads((args.replay / "result.json").read_text())
        if (replay["status"] != "PASS_TEMPORAL_PLUMBING_ONLY"
                or replay["source_commit"] != source_commit
                or replay["config_sha256"] != sha(CONFIG)):
            raise ValueError("same source/config and completed reference required")
        for name, digest in replay["artifacts"].items():
            if sha(args.replay / name) != digest:
                raise ValueError("replay artifact changed")
    if not torch.cuda.is_available():
        raise RuntimeError("local CUDA required by this probe")
    usage = subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"
    ], text=True).strip()
    memory, utilization = [int(x.strip()) for x in usage.split(",")]
    if memory > 512 or utilization > 5:
        raise RuntimeError("GPU occupied; do not compete with another process")
    args.output.mkdir(exist_ok=False)
    started = time.perf_counter()
    result = dict(status="RUNNING", source_commit=source_commit, config_sha256=sha(CONFIG),
                  config=config, command=sys.argv, started_utc=datetime.now(UTC).isoformat(),
                  quality_evaluation=False, validation_loaded=False, calibration_loaded=False,
                  sealed_audit_loaded=False, encoder_execution=False, real_training_updates=0,
                  artificial_target_updates=0, cloud_cost=0, gpu_preflight=usage)
    write(args.output / "result.json", result)
    try:
        torch.set_num_threads(4)
        torch.manual_seed(config["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        checked = {str(CONFIG): sha(CONFIG), str(source / "result.json"):
                   config["feature_receipt_sha256"]}

        def check(name):
            path = source / name
            if sha(path) != previous["artifact_hashes"][name]:
                raise ValueError(f"changed input {name}")
            checked[str(path)] = sha(path)
            return path

        blocks = {}
        for kind in ("text", "image", "speech", "event_audio", "video"):
            array = np.load(check(kind + "-tokens.npy"), allow_pickle=False)
            if array.dtype != np.float32:
                raise ValueError("original FP32 features required")
            blocks[kind] = NativeTokenBlock(kind, torch.from_numpy(array).cuda(),
                                            positions(kind, len(array)).cuda(),
                                            torch.ones(len(array), dtype=torch.bool, device="cuda"))
        baseline = NativeCandidateReadout()
        baseline.load_state_dict(torch.load(check("readout.pt"), weights_only=True))
        model = TemporalNativeCandidate(baseline).cuda().eval()
        if (config["hidden_size"] != 192 or config["layers"] != 1
                or config["bidirectional"] or not config["zero_output_initialization"]
                or config["artificial_target_updates"] != 2
                or config["quality_evaluation"] or config["task_training"]):
            raise ValueError("probe configuration differs from implemented profile")
        baseline.requires_grad_(False)
        before = {k: v.detach().clone() for k, v in baseline.state_dict().items()}
        result["readout_parameters"] = sum(p.numel() for p in baseline.parameters())
        result["adapter_parameters"] = {k: sum(p.numel() for p in m.parameters())
                                        for k, m in model.temporal.items()}
        result["candidate_parameters"] = (config["frozen_encoder_parameters"]
                                            + sum(p.numel() for p in model.parameters()))
        if result["candidate_parameters"] > 200000000:
            raise ValueError("prequantization cap exceeded")
        torch.save(model.state_dict(), args.output / "initial.pt")
        cases = []
        for modality in ("text", "image", "audio", "video"):
            native = [blocks[k] for k in ("speech", "event_audio")] if modality == "audio" else [
                blocks[modality]]
            for i in range(2):
                key = f"{modality}-{i}"
                q = torch.from_numpy(np.load(check(key + "-queries.npy"),
                                              allow_pickle=False)).cuda()
                cases.append((key, native, q))
        initial = []
        timing = {}
        with torch.no_grad():
            for key, native, q in cases:
                expected, actual = baseline(native, q), model(native, q)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                torch.testing.assert_close(actual.flip(0), model(native, q.flip(0)),
                                           atol=1e-6, rtol=0)
                initial.append(actual.cpu().numpy())
                timing[key] = {}
                for name, candidate in (("baseline", baseline), ("lstm", model)):
                    for _ in range(config["warmup_calls"]):
                        candidate(native, q)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    spans = []
                    for _ in range(config["timed_calls"]):
                        tick = time.perf_counter()
                        candidate(native, q)
                        torch.cuda.synchronize()
                        spans.append((time.perf_counter() - tick) * 1000)
                    timing[key][name] = dict(
                        p50_ms=float(np.percentile(spans, 50)),
                        p95_ms=float(np.percentile(spans, 95)), samples_ms=spans,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())
        np.save(args.output / "initial-scores.npy", np.stack(initial), allow_pickle=False)
        optimizer = torch.optim.SGD(model.temporal.parameters(), lr=config["diagnostic_lr"])
        # Artificial target 0 exercises gradients only. Never use for quality or selection.
        mixed = [blocks[k] for k in ("speech", "event_audio", "video")]
        q = cases[4][2]
        norms, losses = [], []
        model.train()
        baseline.eval()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(2):
            optimizer.zero_grad()
            loss = -model(mixed, q).log_softmax(0)[0]
            if not torch.isfinite(loss):
                raise ValueError("nonfinite artificial loss")
            loss.backward()
            grad = {}
            for name, p in model.temporal.named_parameters():
                if p.grad is None or not torch.isfinite(p.grad).all():
                    raise ValueError("missing/nonfinite adapter gradient")
                grad[name] = float(p.grad.norm())
            norms.append(grad)
            losses.append(float(loss.detach()))
            optimizer.step()
            result["artificial_target_updates"] += 1
        if not all(v > 0 for v in norms[1].values()):
            raise ValueError("recurrent gradient reach failed on second update")
        torch.cuda.synchronize()
        result["gradient_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        if not all(torch.equal(v, before[k]) for k, v in baseline.state_dict().items()):
            raise ValueError("fixed baseline changed")
        torch.save(model.state_dict(), args.output / "diagnostic-only.pt")
        model.eval()
        loaded = TemporalNativeCandidate().cuda().eval()
        loaded.load_state_dict(torch.load(args.output / "diagnostic-only.pt", weights_only=True))
        after = []
        with torch.no_grad():
            for _key, native, q in cases:
                scores = model(native, q)
                torch.testing.assert_close(scores, loaded(native, q), atol=0, rtol=0)
                torch.testing.assert_close(scores, model(native, q), atol=0, rtol=0)
                after.append(scores.cpu().numpy())
        np.save(args.output / "diagnostic-scores.npy", np.stack(after), allow_pickle=False)
        if replay:
            for name in ("initial-scores.npy", "diagnostic-scores.npy"):
                np.testing.assert_array_equal(np.load(args.output / name),
                                              np.load(args.replay / name))
            for name in ("initial.pt", "diagnostic-only.pt"):
                current = torch.load(args.output / name, weights_only=True, map_location="cpu")
                expected = torch.load(args.replay / name, weights_only=True, map_location="cpu")
                if current.keys() != expected.keys() or any(
                        not torch.equal(v, expected[k]) for k, v in current.items()):
                    raise ValueError("fresh-process weight replay mismatch")
            if losses != replay["artificial_losses"] or norms != replay["gradient_norms"]:
                raise ValueError("fresh-process gradient replay mismatch")
        for path, digest in checked.items():
            if sha(path) != digest:
                raise ValueError("input changed during probe")
        result.update(status="PASS_TEMPORAL_PLUMBING_ONLY", timings=timing,
                      artificial_losses=losses, gradient_norms=norms, checked=checked,
                      baseline_unchanged=True, initial_baseline_max_difference=0,
                      checkpoint_reload_max_difference=0, replay_exact=bool(replay),
                      replay_reference_sha256=args.replay_sha256,
                      device=torch.cuda.get_device_name(), python=platform.python_version(),
                      torch=str(torch.__version__), numpy=np.__version__, cuda=torch.version.cuda,
                      cudnn=torch.backends.cudnn.version(), dtype="float32", tf32=False,
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      artifacts={p.name: sha(p) for p in args.output.iterdir()
                                 if p.name != "result.json"})
    except Exception as exc:
        result.update(status="FAILED", failure=repr(exc))
        raise
    finally:
        result.update(finished_utc=datetime.now(UTC).isoformat(),
                      seconds=time.perf_counter() - started)
        write(args.output / "result.json", result)
    print(json.dumps({k: result[k] for k in ["status", "candidate_parameters", "adapter_parameters",
                                           "seconds", "replay_exact"]}))


if __name__ == "__main__":
    main()
