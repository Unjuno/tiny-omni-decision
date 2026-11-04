"""Verify content-based observation replacement with real frozen Path A workers."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.benchmark_path_a_dispatcher import (  # noqa: E402
    _compare_output,
    _git,
    _make_manifested_view,
)
from scripts.benchmark_path_a_latency_distribution import (  # noqa: E402
    _choose_images,
    _choose_videos,
)
from tiny_omni_decision.dataset import sha256_file  # noqa: E402
from tiny_omni_decision.runtime_dispatch import PathARuntimeDispatcher  # noqa: E402


def verify_updates(
    dispatcher: Any, specs: list[dict[str, Any]], output: Path
) -> list[dict[str, Any]]:
    """Mutate only independent media copies; check each response before proceeding."""
    first, second = specs
    task = first["task"]
    path_key = f"{task}_path"
    active = output / f"active-{task}{Path(first['payload'][path_key]).suffix}"
    alias = output / f"alias-{task}{active.suffix}"
    events: list[dict[str, Any]] = []

    def record(event: dict[str, Any]) -> None:
        events.append(event)
        with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def decide(label: str, spec: dict[str, Any], path: Path, hit: bool) -> None:
        payload = {**spec["payload"], path_key: str(path)}
        response = dispatcher.dispatch(task, payload)
        comparison = _compare_output(spec, response)
        metrics = response["video_encoder" if task == "video" else "encoder_metrics"]
        passed = (
            metrics["cache_hit"] is hit
            and comparison["prediction_mismatches"] == 0
            and comparison["max_probability_absolute_delta"] <= 1e-5
        )
        record({
            "task": task, "event": label, "passed": passed,
            "media_sha256": sha256_file(path), "expected_cache_hit": hit,
            "comparison": comparison, "response": response,
        })
        if not passed:
            raise AssertionError(f"{task}/{label}: cache or reference parity failed")

    shutil.copyfile(first["payload"][path_key], active)
    decide("initial_observation", first, active, False)
    alternate = {**first, "payload": first["warm_payload"], "reference": first["warm_reference"]}
    decide("new_question_same_observation", alternate, active, True)
    shutil.copyfile(active, alias)
    decide("same_bytes_new_path", first, alias, True)

    # Keep the original timestamp to rule out path/mtime-only cache identity.
    import os

    original_stat = active.stat()
    shutil.copyfile(second["payload"][path_key], active)
    os.utime(active, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    try:
        dispatcher.dispatch(task, {**first["payload"], path_key: str(active)})
    except RuntimeError as error:
        passed = "hash differs from the supplied frozen identity" in str(error)
        record({"task": task, "event": "stale_hash_rejected", "passed": passed,
                "error": str(error)})
        if not passed:
            raise
    else:
        record({"task": task, "event": "stale_hash_rejected", "passed": False})
        raise AssertionError(f"{task}: stale media hash was accepted")
    decide("replacement_same_path_and_mtime", second, active, False)
    decide("repeat_replacement", second, active, True)
    shutil.copyfile(first["payload"][path_key], active)
    decide("restore_original_content", first, active, True)
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--videoprism-python", type=Path, required=True)
    parser.add_argument("--videoprism-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    specs = [_choose_images(args.artifact_root, 2), _choose_videos(args.artifact_root, 2)]
    paths = {path for group in specs for spec in group for path in spec["source_files"]}
    source_hashes = {str(path): sha256_file(path) for path in sorted(paths)}
    report = {
        "started_utc": datetime.now(UTC).isoformat(),
        "source_commit": _git(REPO, "rev-parse", "HEAD"),
        "source_status": _git(REPO, "status", "--porcelain"),
        "runner_sha256": sha256_file(Path(__file__)),
        "command": sys.argv,
        "source_hashes_before": source_hashes,
        "scope": "two Image assets and two Video scenes; reused development predictions",
        "tolerance": 1e-5,
        "package": _make_manifested_view(args.assembly.resolve(), output / "package-view"),
        "events": [],
        "status": "RUNNING",
    }
    result_path = output / "result.json"

    def save() -> None:
        result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        with PathARuntimeDispatcher(
            output / "package-view", videoprism_python=args.videoprism_python,
            videoprism_source=args.videoprism_source,
            worker_log_dir=output / "worker-logs", device=args.device,
        ) as dispatcher:
            for group in specs:
                report["events"].extend(verify_updates(dispatcher, group, output))
                save()
            report["workers"] = dispatcher.worker_metadata
        after = {path: sha256_file(Path(path)) for path in source_hashes}
        report["source_hashes_after"] = after
        if source_hashes != after:
            raise AssertionError("source files changed during verification")
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        report["finished_utc"] = datetime.now(UTC).isoformat()
        save()
    print(json.dumps({"status": report["status"], "result": str(result_path),
                      "sha256": sha256_file(result_path)}))


if __name__ == "__main__":
    main()
