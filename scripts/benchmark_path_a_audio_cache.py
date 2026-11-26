"""Compare fixed single-event Audio inference with and without feature reuse."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.benchmark_path_a_dispatcher import (  # noqa: E402
    _compare_output,
    _git,
    _make_manifested_view,
)
from scripts.benchmark_path_a_latency_distribution import _choose_audio  # noqa: E402
from scripts.verify_path_a_observation_updates import verify_updates  # noqa: E402
from tiny_omni_decision.dataset import sha256_file  # noqa: E402
from tiny_omni_decision.modular_bundle import verify_modular_bundle  # noqa: E402
from tiny_omni_decision.runtime_dispatch import JsonLineWorker  # noqa: E402


class AudioClient:
    def __init__(self, worker):
        self.worker = worker

    def dispatch(self, task, payload):
        started = time.perf_counter()
        response = self.worker.request({"task": task, "payload": payload})
        return {**response, "dispatch_wall_seconds": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    specs = _choose_audio(args.artifact_root, 10)
    files = {p for spec in specs for p in spec["source_files"]}
    sources = {str(p): sha256_file(p) for p in sorted(files)}
    report = {
        "source_commit": _git(REPO, "rev-parse", "HEAD"),
        "source_status": _git(REPO, "status", "--porcelain"),
        "started_utc": datetime.now(UTC).isoformat(),
        "command": sys.argv,
        "code_sha256": {str(p): sha256_file(REPO / p) for p in [
            Path(__file__).relative_to(REPO), Path("scripts/path_a_torch_worker.py"),
            Path("src/tiny_omni_decision/audio_runtime.py"),
            Path("src/tiny_omni_decision/feature_cache.py"),
        ]},
        "source_hashes_before": sources,
        "package": _make_manifested_view(args.assembly.resolve(), output / "package-view"),
        "runs": {}, "status": "RUNNING",
        "scope": "10 reused validation audio assets; runtime parity, not new quality",
    }
    verify_modular_bundle(output / "package-view")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src"), str(REPO)])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    started = time.perf_counter()
    baseline = []
    try:
        for capacity in (0, 64):
            worker = JsonLineWorker(
                [sys.executable, "-B", "-m", "scripts.path_a_torch_worker",
                 "--assembly", str(output / "package-view"), "--device", "cuda",
                 "--audio-cache-entries", str(capacity)],
                cwd=REPO, env=env, log_path=output / f"worker-{capacity}.log",
            )
            client = AudioClient(worker)
            run = {"worker": worker.ready, "requests": []}
            report["runs"][str(capacity)] = run
            try:
                for index, spec in enumerate(specs if capacity == 0 else specs[:8]):
                    for repeat in range(2 if index < 8 else 1):
                        response = client.dispatch("audio", spec["payload"])
                        if capacity == 0 and repeat == 0:
                            baseline.append({**spec, "reference": {
                                "sample_id": spec["reference"]["sample_id"],
                                "decisions": [
                                    {**old, **new} for old, new in zip(
                                        spec["reference"]["decisions"],
                                        response["decisions"], strict=True,
                                    )
                                ],
                            }})
                        comparison = _compare_output(baseline[index], response)
                        expected_hit = capacity > 0 and repeat == 1
                        passed = (
                            comparison["prediction_mismatches"] == 0
                            and comparison["max_probability_absolute_delta"] <= 1e-6
                            and response["encoder_metrics"]["cache_hit"] is expected_hit
                        )
                        run["requests"].append({
                            "index": index, "repeat": repeat, "passed": passed,
                            "comparison": comparison, "response": response,
                        })
                        if not passed:
                            raise AssertionError("Audio cache output or hit/miss differed")
                if capacity:
                    updates = []
                    for spec in baseline[8:]:
                        payload = {**spec["payload"], "expected_media_sha256":
                                   sha256_file(Path(spec["payload"]["audio_path"]))}
                        updates.append({
                            **spec,
                            "payload": {**payload, "queries": payload["queries"][:1]},
                            "reference": {**spec["reference"], "decisions":
                                          spec["reference"]["decisions"][:1]},
                            "warm_payload": {**payload, "queries": payload["queries"][1:2]},
                            "warm_reference": {**spec["reference"], "decisions":
                                               spec["reference"]["decisions"][1:2]},
                        })
                    run["updates"] = verify_updates(client, updates, output)
            finally:
                worker.close()
        report["source_hashes_after"] = {p: sha256_file(Path(p)) for p in sources}
        if report["source_hashes_after"] != sources:
            raise AssertionError("source hashes changed")
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        path = output / "result.json"
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "result": str(path),
                      "sha256": sha256_file(path)}))


if __name__ == "__main__":
    main()
