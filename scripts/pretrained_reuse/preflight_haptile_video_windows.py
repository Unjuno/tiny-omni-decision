"""Audit fixed one-second/eight-frame video eligibility without reading labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.pretrained_reuse.analyze_haptile_development_preflight import (  # noqa: E402
    validate_development_receipt,
)
from scripts.pretrained_reuse.analyze_haptile_train_preflight import (  # noqa: E402
    validate_train_receipt,
)
from tiny_omni_decision.haptile_video import (  # noqa: E402
    select_causal_frame_indices,
)

PINNED_REVISION = "c8593da98d3facb7a9293842514f0ee4ad3f2318"
TOPICS = ("/front-camera", "/wrist-camera", "/tactile-left", "/tactile-right")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _audit_media(
    media: Path, query: float
) -> tuple[dict[str, list[float]] | None, list[dict[str, object]]]:
    try:
        from mcap.reader import make_reader
        from mcap_protobuf.decoder import DecoderFactory
    except ImportError as exc:
        raise RuntimeError("install pinned MCAP decoder dependencies") from exc
    times_by_topic: dict[str, list[float]] = {topic: [] for topic in TOPICS}
    with media.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()], validate_crcs=True)
        for schema, channel, message, compressed in reader.iter_decoded_messages(
            topics=list(TOPICS)
        ):
            topic = channel.topic
            if (topic not in times_by_topic or schema is None
                    or schema.name != "foxglove.CompressedVideo"
                    or compressed.format != "h264"):
                raise ValueError(f"unexpected HapTile video schema/codec: {media} {topic}")
            timestamp = message.log_time / 1_000_000_000
            if times_by_topic[topic] and timestamp <= times_by_topic[topic][-1]:
                raise ValueError(f"nonmonotonic video timestamps: {media} {topic}")
            if query - 1.1 <= timestamp <= query:
                times_by_topic[topic].append(timestamp)
    failures: list[dict[str, object]] = []
    selected_by_topic: dict[str, list[float]] = {}
    for topic in TOPICS:
        try:
            indices = select_causal_frame_indices(times_by_topic[topic], query)
            selected_by_topic[topic] = [times_by_topic[topic][i] for i in indices]
        except ValueError:
            failures.append({
                "topic": topic,
                "available_frames_in_window": len(times_by_topic[topic]),
                "reason": "no_8_distinct_causal_frames_within_100ms_target_tolerance",
            })
    return (selected_by_topic if not failures else None), failures


def preflight_video_windows(
    *, receipt_path: Path, action_path: Path, split_manifest_path: Path,
    split_name: str, output_path: Path,
) -> dict[str, Any]:
    if split_name not in {"train", "validation", "calibration"}:
        raise ValueError("sealed audit is not eligible for this preflight")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    action = json.loads(action_path.read_text(encoding="utf-8"))
    manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    manifest_hash = sha256_file(split_manifest_path)
    if split_name == "train":
        files = validate_train_receipt(receipt, manifest, manifest_hash)
        expected_format = "haptile_train_only_action_preflight_v1"
    else:
        files = validate_development_receipt(
            receipt, manifest, manifest_hash, split_name
        )
        expected_format = f"haptile_{split_name}_action_preflight_v1"
    if action.get("format") != expected_format:
        raise ValueError("action preflight format differs from the requested split")
    if action.get("input_receipt_sha256") != sha256_file(receipt_path):
        raise ValueError("action preflight points to a different media receipt")
    expected_action_split_hash = action.get("split_manifest_sha256")
    if expected_action_split_hash != manifest_hash:
        raise ValueError("action preflight uses a different frozen split")
    rows = action.get("samples")
    if not isinstance(rows, list) or len(rows) != len(files):
        raise ValueError("action preflight rows do not cover the requested corpus")
    action_by_path = {str(row.get("path")): row for row in rows if isinstance(row, dict)}
    if len(action_by_path) != len(rows):
        raise ValueError("action preflight contains duplicate or malformed paths")

    eligible: list[dict[str, str]] = []
    excluded: list[dict[str, object]] = []
    groups: Counter[str] = Counter()
    for index, item in enumerate(files, 1):
        relative = Path(str(item["path"]))
        path = receipt_path.parent.joinpath(*relative.parts)
        if (not path.is_file() or path.stat().st_size != int(item["size"])
                or sha256_file(path) != item.get("sha256")):
            raise ValueError(f"media integrity failure during video preflight: {relative}")
        row = action_by_path.get(relative.as_posix())
        if row is None or row.get("sha256") != item.get("sha256"):
            raise ValueError(f"media and action preflight differ: {relative}")
        query = float(row["query_time_log_seconds"])
        selected_times, failures = _audit_media(path, query)
        task = str(item.get("task", row.get("task", relative.parts[1])))
        if selected_times is not None:
            eligible.append({
                "path": relative.as_posix(),
                "sha256": str(item["sha256"]),
                "task": task,
                "query_time_log_seconds": query,
                "selected_frame_times_log_seconds": selected_times,
            })
            groups[task] += 1
        else:
            excluded.append({
                "path": relative.as_posix(),
                "sha256": str(item["sha256"]),
                "task": task,
                "video_window_failures": failures,
            })
        if index % 50 == 0:
            print(
                f"preflighted {index}/{len(files)} {split_name} media; "
                f"eligible={len(eligible)} excluded={len(excluded)}",
                flush=True,
            )
    payload: dict[str, Any] = {
        "format": f"haptile_{split_name}_video_window_eligibility_v2",
        "split": split_name,
        "source_revision": manifest["source_revision"],
        "conversion_revision": PINNED_REVISION,
        "input_receipt_sha256": sha256_file(receipt_path),
        "action_preflight_sha256": sha256_file(action_path),
        "split_manifest_sha256": manifest_hash,
        "selection_uses_labels": False,
        "criteria": {
            "context_seconds": 1.0,
            "frames_per_stream": 8,
            "max_frame_time_error_seconds": 0.1,
            "causal_cutoff_inclusive": True,
            "streams": list(TOPICS),
            "excluded_only_for": "insufficient distinct causal frames in fixed window",
        },
        "corpus_examples": len(files),
        "eligible_examples": len(eligible),
        "excluded_examples": len(excluded),
        "eligible_task_groups": len(groups),
        "eligible_examples_by_task": dict(sorted(groups.items())),
        "eligible_samples": eligible,
        "excluded_samples": excluded,
        "sealed_audit_opened": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite video-window audit: {output_path}")
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--action-preflight", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "calibration"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = preflight_video_windows(
        receipt_path=args.receipt, action_path=args.action_preflight,
        split_manifest_path=args.split_manifest, split_name=args.split,
        output_path=args.output,
    )
    print(json.dumps({
        "split": args.split,
        "eligible": result["eligible_examples"],
        "excluded": result["excluded_examples"],
        "eligible_task_groups": result["eligible_task_groups"],
        "sealed_audit_opened": result["sealed_audit_opened"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
