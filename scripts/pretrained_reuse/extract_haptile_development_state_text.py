"""Extract causal state/instruction features for validation or calibration only."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.pretrained_reuse.analyze_haptile_development_preflight import (  # noqa: E402
    sha256_file,
    validate_development_receipt,
)
from scripts.pretrained_reuse.extract_haptile_train_state_text import (  # noqa: E402
    _extract_one,
)
from tiny_omni_decision.haptile_state import STREAM_WIDTHS  # noqa: E402


def validate_development_action(
    *, action: dict[str, object], receipt: dict[str, object], receipt_path: Path,
    manifest_path: Path, split_name: str, file_rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    expected_format = f"haptile_{split_name}_action_preflight_v1"
    if (action.get("format") != expected_format
            or action.get("input_receipt_sha256") != sha256_file(receipt_path)
            or action.get("split_manifest_sha256") != sha256_file(manifest_path)
            or action.get("split_accessed") != split_name
            or action.get("sealed_audit_opened") is not False):
        raise ValueError("development target preflight provenance is inconsistent")
    rows = action.get("samples")
    if not isinstance(rows, list) or len(rows) != len(file_rows):
        raise ValueError("development target rows do not align one-to-one with media")
    targets: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("development target preflight has a malformed row")
        path = str(row["path"])
        if path in targets:
            raise ValueError("development target preflight has duplicate paths")
        targets[path] = row
    if set(targets) != {str(item["path"]) for item in file_rows}:
        raise ValueError("development targets and media receipt differ")
    return targets


def extract_development_state_text(
    *, receipt_path: Path, action_path: Path, split_manifest_path: Path,
    split_name: str, output_dir: Path,
) -> dict[str, object]:
    if split_name not in {"validation", "calibration"}:
        raise ValueError("only validation or calibration may be extracted")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    files = validate_development_receipt(
        receipt, manifest, sha256_file(split_manifest_path), split_name
    )
    action = json.loads(action_path.read_text(encoding="utf-8"))
    target_by_path = validate_development_action(
        action=action, receipt=receipt, receipt_path=receipt_path,
        manifest_path=split_manifest_path, split_name=split_name, file_rows=files,
    )

    output_dir.mkdir(parents=True)
    shutil.copy2(Path(__file__), output_dir / "extract_haptile_development_state_text.executed.py")
    records: list[dict[str, object]] = []
    sequences: list[np.ndarray] = []
    for index, item in enumerate(files):
        relative = Path(str(item["path"]))
        media = receipt_path.parent.joinpath(*relative.parts)
        actual_sha = sha256_file(media)
        row = target_by_path[str(relative).replace("\\", "/")]
        if actual_sha != item["sha256"] or row.get("sha256") != actual_sha:
            raise ValueError(f"media hash or target row mismatch: {relative}")
        instruction, state = _extract_one(media, float(row["query_time_log_seconds"]))
        sequences.append(state)
        records.append({
            "path": str(relative).replace("\\", "/"),
            "sha256": actual_sha,
            "task_group": item.get("task", relative.parts[1]),
            "instruction": instruction,
            "query_time_log_seconds": row["query_time_log_seconds"],
            "target": row["target"],
        })
        if (index + 1) % 50 == 0:
            print(f"extracted {index + 1}/{len(files)} {split_name} state rows", flush=True)

    states_path = output_dir / "state-sequences.npy"
    np.save(states_path, np.stack(sequences).astype(np.float32, copy=False))
    examples_path = output_dir / "examples.json"
    examples_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    result = {
        "format": f"haptile_{split_name}_state_text_features_v1",
        "status": f"complete_{split_name}_only_state_text_extraction",
        "input_receipt_sha256": sha256_file(receipt_path),
        "action_preflight_sha256": sha256_file(action_path),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "split": split_name,
        "examples": len(records),
        "task_groups": sorted({
            str(item.get("task", Path(str(item["path"])).parts[1])) for item in files
        }),
        "state_shape": list(np.load(states_path, mmap_mode="r").shape),
        "state_channels": list(STREAM_WIDTHS),
        "state_sequence_sha256": sha256_file(states_path),
        "examples_sha256": sha256_file(examples_path),
        "sealed_audit_opened": False,
        "audio_present_in_source": False,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--action-preflight", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "calibration"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(extract_development_state_text(
        receipt_path=args.receipt, action_path=args.action_preflight,
        split_manifest_path=args.split_manifest, split_name=args.split,
        output_dir=args.output_dir,
    ), indent=2), flush=True)


if __name__ == "__main__":
    main()
