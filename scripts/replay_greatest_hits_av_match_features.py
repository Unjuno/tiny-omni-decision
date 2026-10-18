"""Fresh-process replay of frozen AV features against the saved feature cache."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.extract_greatest_hits_av_match_features import (
    _load_encoder,
    read_groups,
    sha256_file,
)


def _ordered_feature_rows(features_dir: Path, expected_rows: list[dict[str, Any]]):
    result_path = features_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest_path = features_dir / "features.jsonl"
    if result.get("status") != "PASS_FROZEN_FEATURE_EXTRACTION_REPLAY":
        raise ValueError("complete frozen feature receipt required")
    if result.get("features_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("saved feature manifest hash mismatch")
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != len(expected_rows):
        raise ValueError("saved feature count differs from input group count")
    for input_row, feature_row in zip(expected_rows, rows, strict=True):
        if (
            feature_row["source_group"] != input_row["source_group"]
            or feature_row["split"] != input_row["split"]
            or feature_row["input_sha256"] != input_row["input_sha256"]
        ):
            raise ValueError("saved feature manifest order/source identity differs")
    return result, rows


def replay(modality: str, inputs_dir: Path, features_dir: Path, output_dir: Path) -> dict[str, Any]:
    import numpy as np

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    _, input_rows = read_groups(inputs_dir)
    feature_result, feature_rows = _ordered_feature_rows(features_dir, input_rows)
    output_dir.mkdir(parents=True)
    report: dict[str, Any] = {
        "status": "RUNNING",
        "modality": modality,
        "device": "cpu",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "started_utc": datetime.now(UTC).isoformat(),
        "inputs_result_sha256": sha256_file(inputs_dir / "result.json"),
        "groups_manifest_sha256": sha256_file(inputs_dir / "groups.jsonl"),
        "features_result_sha256": sha256_file(features_dir / "result.json"),
        "features_manifest_sha256": sha256_file(features_dir / "features.jsonl"),
        "feature_extraction_runner_sha256": feature_result["runner_sha256"],
        "training": False,
        "evaluation": False,
        "sealed_audit_opened": False,
        "optimizer_updates": 0,
        "groups": [],
    }
    start = time.perf_counter()
    try:
        encode, device_fn = _load_encoder(modality, "cpu", report)
        report["actual_device"] = device_fn()
        for index, (input_row, feature_row) in enumerate(
            zip(input_rows, feature_rows, strict=True)
        ):
            input_path = inputs_dir / input_row["input_file"]
            feature_path = features_dir / feature_row["feature_file"]
            if sha256_file(input_path) != input_row["input_sha256"]:
                raise ValueError("frozen prepared input changed")
            if sha256_file(feature_path) != feature_row["feature_sha256"]:
                raise ValueError("saved feature file hash mismatch")
            saved = np.load(feature_path, allow_pickle=False)
            if saved.dtype != np.float32 or not np.isfinite(saved).all():
                raise ValueError("saved feature is not finite float32")
            with np.load(input_path, allow_pickle=False) as data:
                computed = encode(data).astype(np.float32, copy=False)
            if computed.shape != saved.shape:
                raise ValueError("fresh-process feature shape differs")
            delta = float(np.max(np.abs(computed - saved)))
            if not np.allclose(computed, saved, rtol=1e-5, atol=1e-6):
                raise ValueError("fresh-process feature replay mismatch")
            report["groups"].append(
                {
                    "source_group": input_row["source_group"],
                    "split": input_row["split"],
                    "input_sha256": input_row["input_sha256"],
                    "feature_sha256": feature_row["feature_sha256"],
                    "feature_shape": list(saved.shape),
                    "max_abs_delta": delta,
                }
            )
            if index % 16 == 0 or index + 1 == len(input_rows):
                (output_dir / "progress.json").write_text(
                    json.dumps(
                        {
                            "completed": index + 1,
                            "expected": len(input_rows),
                            "seconds": time.perf_counter() - start,
                            "last_source_group": input_row["source_group"],
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
        report["status"] = "PASS_FRESH_PROCESS_FEATURE_REPLAY"
    except Exception as error:
        report.update(status="FAILED", error=repr(error))
        raise
    finally:
        report.update(
            finished_utc=datetime.now(UTC).isoformat(),
            elapsed_seconds=time.perf_counter() - start,
            completed=len(report["groups"]),
            max_abs_delta=max((row["max_abs_delta"] for row in report["groups"]), default=None),
        )
        (output_dir / "audit.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        digest = sha256_file(output_dir / "audit.json")
        (output_dir / "audit.sha256").write_text(digest + "\n", encoding="ascii")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=("audio", "video"), required=True)
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.modality, args.inputs_dir, args.features_dir, args.output_dir)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "modality", "completed", "elapsed_seconds", "max_abs_delta")
            }
        )
    )


if __name__ == "__main__":
    main()
