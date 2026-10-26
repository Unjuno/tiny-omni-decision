"""Build an immutable, metadata-only DROID early-outcome task manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs/pretrained_reuse/droid_early_outcome_video_sensor_v0.json"
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tiny_omni_decision.droid_early_outcome import build_early_outcome_rows  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> tuple[dict[str, Any], str]:
    data = path.read_bytes()
    return json.loads(data), sha256(data)


def write_json(path: Path, value: Any) -> str:
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)
    return sha256(data)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT
        / "manifests/pretrained_reuse/droid-selected-shards-seed17-20261026.json",
    )
    parser.add_argument("--receipt-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = _args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing task output: {output_dir}")
    if output_dir.is_relative_to(REPOSITORY_ROOT.resolve()):
        raise ValueError("task outputs must be outside the repository")

    config, config_sha = read_json(args.config.resolve(strict=True))
    manifest, manifest_sha = read_json(args.manifest.resolve(strict=True))
    audit, audit_sha = read_json(args.audit_json.resolve(strict=True))
    split, split_sha = read_json(args.split_json.resolve(strict=True))
    receipt, receipt_sha = read_json(args.receipt_json.resolve(strict=True))
    expected = config["source"]
    for name, observed in (
        ("selection_manifest_sha256", manifest_sha),
        ("download_receipt_sha256", receipt_sha),
        ("audit_sha256", audit_sha),
        ("development_split_sha256", split_sha),
    ):
        if expected[name] != observed:
            raise ValueError(f"{name} differs from the frozen task config")
    if receipt.get("manifest_sha256") != manifest_sha:
        raise ValueError("download receipt belongs to another selected-shard manifest")
    if split.get("selection_manifest_sha256") != manifest_sha:
        raise ValueError("development split belongs to another selected-shard manifest")
    if audit.get("selection_manifest_sha256") != manifest_sha:
        raise ValueError("serialized-field audit belongs to another selected-shard manifest")
    if (audit.get("episode_count"), audit.get("unique_recording_groups")) != (539, 539):
        raise ValueError("audit does not match the fixed 539-episode source selection")
    if config["model"].get("training_started") is not False:
        raise ValueError("this preparation step must not start model training")

    task = config["task"]
    result = build_early_outcome_rows(
        audit,
        split,
        observation_count=int(task["observation_count"]),
        observed_fraction=float(task["observed_fraction"]),
    )
    if result["options"] != task["options"]:
        raise ValueError("task option order differs from the frozen config")
    if result["input_contract"]["modalities"] != task["input_modalities"]:
        raise ValueError("task input modalities differ from the frozen config")
    if result["input_contract"]["video_views"] != task["video_views"]:
        raise ValueError("task video views differ from the frozen config")
    if result["input_contract"]["sensor_fields"] != task["sensor_fields"]:
        raise ValueError("task sensor fields differ from the frozen config")
    if result["input_contract"]["future_observations_in_input"] is not False:
        raise ValueError("future observations must not be task inputs")

    output_dir.mkdir(parents=True, exist_ok=False)
    for name in ("train", "validation"):
        rows = result["splits"][name]["rows"]
        lines = "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
        )
        (output_dir / f"{name}.jsonl").write_text(lines, encoding="utf-8", newline="\n")
    result_sha = write_json(output_dir / "task-manifest.json", result)
    effective_config_path = output_dir / "effective-config.json"
    shutil.copyfile(args.config.resolve(strict=True), effective_config_path)
    run = {
        "status": "PREPARED_NOT_TRAINED",
        "created_utc": datetime.now(UTC).isoformat(),
        "config_sha256": config_sha,
        "effective_config_sha256": sha256(effective_config_path.read_bytes()),
        "selection_manifest_sha256": manifest_sha,
        "download_receipt_sha256": receipt_sha,
        "audit_sha256": audit_sha,
        "development_split_sha256": split_sha,
        "task_manifest_sha256": result_sha,
        "train_jsonl_sha256": sha256((output_dir / "train.jsonl").read_bytes()),
        "validation_jsonl_sha256": sha256((output_dir / "validation.jsonl").read_bytes()),
        "sealed_audit_loaded": False,
        "model_training_started": False,
    }
    write_json(output_dir / "run-metadata.json", run)
    print(
        json.dumps(
            {
                "status": run["status"],
                "output_dir": str(output_dir),
                "train": result["splits"]["train"]["target_counts"],
                "validation": result["splits"]["validation"]["target_counts"],
                "excluded_short": {
                    name: result["splits"][name]["excluded_short_episode_counts"]
                    for name in ("train", "validation")
                },
                "task_manifest_sha256": result_sha,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
