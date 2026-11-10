"""Run the frozen AVMIT SAVC smoke or mean-vs-GRU readout experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tiny_omni_decision.avmit_training import (
    cache_fingerprint,
    resolve_device,
    run_smoke,
    train_and_evaluate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "train"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke-report", type=Path)
    args = parser.parse_args()

    if args.mode == "smoke":
        result = run_smoke(args.cache, args.pairs, args.output, device_name=args.device)
    else:
        if args.smoke_report is None:
            parser.error("train mode requires --smoke-report from a passing separate smoke run")
        smoke = json.loads(args.smoke_report.read_text(encoding="utf-8"))
        if smoke.get("status") != "pass":
            parser.error("train mode requires a passing smoke report")
        current = cache_fingerprint(args.cache, args.pairs)
        for key in (
            "cache_root",
            "pair_manifest_dir",
            "cache_report_sha256",
            "cache_file_sha256",
            "pair_manifest_sha256",
            "config_sha256",
        ):
            if smoke.get(key) != current[key]:
                parser.error(f"smoke report does not match current frozen input: {key}")
        if smoke.get("device") != str(resolve_device(args.device)):
            parser.error("training device differs from the measured smoke device")
        result = train_and_evaluate(args.cache, args.pairs, args.output, device_name=args.device)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
