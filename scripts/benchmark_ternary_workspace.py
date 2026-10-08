"""Opt-in CUDA microbenchmark; never loads a model, corpus, or checkpoint.

Compare with a saved, unmodified ternary.py from the reference commit.
Measures the no-grad fake-quantization used by QAT caches, not whole-run speed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tiny_omni_decision import ternary  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-improvement", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("requires an idle local CUDA GPU")
    spec = importlib.util.spec_from_file_location("reference_ternary", args.reference)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    methods = {
        "reference": reference.fake_quantize_ternary,
        "candidate": ternary.fake_quantize_ternary,
    }
    torch.manual_seed(17)
    ordering = random.Random(17)
    results = []
    with torch.no_grad():
        for size in (65536, 1048576, 16777216, 16777219):
            weights = torch.randn(size, device="cuda", dtype=torch.bfloat16)
            old_codes, old_scales = reference.quantize_groupwise_ternary(weights)
            new_codes, new_scales = ternary.quantize_groupwise_ternary(weights)
            torch.testing.assert_close(new_codes, old_codes, rtol=0, atol=0)
            torch.testing.assert_close(new_scales, old_scales, rtol=0, atol=0)
            torch.testing.assert_close(
                ternary.dequantize_groupwise_ternary(new_codes, new_scales),
                reference.dequantize_groupwise_ternary(old_codes, old_scales),
                rtol=0, atol=0,
            )
            del old_codes, old_scales, new_codes, new_scales
            expected = methods["reference"](weights)
            actual = methods["candidate"](weights)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            del actual, expected
            for method in methods.values():
                for _ in range(2):
                    result = method(weights)
                    del result
            samples = {name: [] for name in methods}
            for _ in range(5):
                names = list(methods)
                ordering.shuffle(names)
                for name in names:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    baseline = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    result = methods[name](weights)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    peak = torch.cuda.max_memory_allocated() - baseline
                    del result
                    samples[name].append({"seconds": elapsed, "extra_peak_bytes": peak})
            results.append({"elements": size, "exact_bf16_parity": True, "methods": {
                name: {"median_seconds": statistics.median(s["seconds"] for s in values),
                       "extra_peak_bytes": max(s["extra_peak_bytes"] for s in values),
                       "samples": values}
                for name, values in samples.items()
            }})
            del weights
    report = {
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "seed": 17, "dtype": "bfloat16",
        "group_size": 256, "threshold_factor": 0.7,
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(Path(ternary.__file__).read_bytes()).hexdigest(),
        "scope": "synthetic no-grad fake quantization only; two warmups, five alternating trials",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.require_improvement:
        for row in results[-2:]:
            methods = row["methods"]
            assert (
                methods["candidate"]["extra_peak_bytes"]
                < 0.8 * methods["reference"]["extra_peak_bytes"]
            ), "workspace reduction below 20%"


if __name__ == "__main__":
    main()
