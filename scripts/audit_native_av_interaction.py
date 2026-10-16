"""CPU parameter/observation-metadata audit; no training or quality evaluation."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from tiny_omni_decision.native_candidate import WIDTHS


def observation_counts(observations):
    result = {s: {"observations": 0, "audio_video_present": 0, "kind_combinations": Counter()}
              for s in ("train", "validation")}
    for item in observations.values():
        split = item.get("split")
        features = item.get("features", [])
        kinds = {f.get("kind") for f in features}
        if split not in result or not kinds or not kinds.issubset(WIDTHS):
            raise ValueError("only known train/validation observation metadata permitted")
        row = result[split]
        row["observations"] += 1
        row["kind_combinations"]["+".join(sorted(kinds))] += 1
        row["audio_video_present"] += int(
            bool(kinds & {"speech", "event_audio"}) and "video" in kinds
        )
    # Presence does not prove source rights, alignment or complementary labels.
    result["joint_training_admitted"] = False
    return result


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--observations-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("explicit CPU-only environment required")
    if sha(args.observations) != args.observations_sha256:
        raise ValueError("observation metadata hash differs")
    repo = Path(__file__).resolve().parents[1]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo).strip():
        raise RuntimeError("clean committed source required")
    import torch

    from tiny_omni_decision.native_av_interaction import ConditionalAVReadout

    args.output.mkdir(parents=True, exist_ok=False)
    counts = observation_counts(json.loads(args.observations.read_text(encoding="utf-8")))
    ledgers = {}
    for exchange in (False, True):
        for gated in (False, True):
            torch.manual_seed(17)
            model = ConditionalAVReadout(exchange=exchange, gated=gated)
            if not all(p.device.type == "cpu" for p in model.parameters()):
                raise RuntimeError("CPU-only module construction required")
            ledger = model.parameter_ledger()
            if ledger["reference_total_parameters"] > 200_000_000:
                raise ValueError("reference total exceeds200M")
            ledgers[f"exchange={exchange},gated={gated}"] = ledger
    if sha(args.observations) != args.observations_sha256:
        raise ValueError("observation metadata changed during audit")
    paths = ["src/tiny_omni_decision/native_candidate.py",
             "src/tiny_omni_decision/native_av_interaction.py",
             "scripts/audit_native_av_interaction.py"]
    record = {
        "status": "PASS_CPU_PARAMETER_AND_METADATA_AUDIT",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "command": [sys.executable, "-m", "scripts.audit_native_av_interaction", *sys.argv[1:]],
        "finished_utc": datetime.now(UTC).isoformat(),
        "source_hashes": {p: sha(repo / p) for p in paths},
        "observations_path": str(args.observations),
        "observations_sha256": args.observations_sha256,
        "observation_counts": counts, "parameter_ledgers": ledgers,
        "python": sys.version, "torch": torch.__version__, "platform": platform.platform(),
        "device": "cpu", "training_updates": 0, "media_loaded": False,
        "sealed_audit_loaded": False, "quality_evaluated": False, "cloud_cost": 0,
        "encoder_count_scope": "existing frozen reference ledger; encoders not loaded here",
    }
    (args.output / "result.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "counts": counts, "ledgers": ledgers}))


if __name__ == "__main__":
    main()
