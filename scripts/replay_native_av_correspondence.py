"""Strict provenance wrapper around the immutable actual raw-media producer.

Verify physical dependencies in both processes before forwards, then require
the complete primary backend profile at runtime construction. Numeric equality
alone never admits a changed backend. The old producer/source stay unchanged.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def verify(pins):
    if not pins or any(sha(p) != h for p, h in pins.items()):
        raise ValueError("physical parent/worker dependency or artifact hash differs")


def verified_primary(folder, expected):
    pins = {str(folder / "result.json"): expected}
    verify(pins)
    primary = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    if primary["status"] != "PASS_NATIVE_AV_CORRESPONDENCE_RUNTIME":
        raise ValueError("complete primary required")
    pins.update(primary["validated_hashes"])
    for name, h in primary["artifact_hashes"].items():
        pins[str(folder / name)] = h
    child = folder / "video-worker/result.json"
    if pins.get(str(child)) != primary["video_worker_receipt_sha256"]:
        raise ValueError("worker receipt authority differs")
    verify(pins)
    worker = json.loads(child.read_text(encoding="utf-8"))
    if worker["status"] != "PASS_NATIVE_VIDEO_WORKER" or not worker["validated_hashes"]:
        raise ValueError("complete pinned worker required")
    for p, h in worker["validated_hashes"].items():
        if p in pins and pins[p] != h:
            raise ValueError("conflicting parent/worker hashes")
        pins[p] = h
    verify(pins)
    return primary, pins


def require_profile(current, expected):
    if current != expected or not isinstance(current, str) or len(current) != 64:
        raise ValueError("complete backend profile differs before native forwards")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-reference", type=Path, required=True)
    parser.add_argument("--replay-sha256", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new replay output required")
    tick = time.perf_counter()
    started = datetime.now(UTC).isoformat()
    primary, pins = verified_primary(args.replay_reference, args.replay_sha256)
    pins[str(Path(__file__).resolve())] = sha(__file__)
    receipt = dict(
        status="RUNNING_STRICT_REPLAY_PROVENANCE",
        validated_hashes=pins,
        expected_profile_sha256=primary["profile_sha256"],
        profile_checked_before_forward=False,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        command=sys.argv,
        started_utc=started,
    )
    from scripts import verify_native_av_correspondence as producer

    original = producer.NativeAVCorrespondence

    def factory(*positional, **keywords):
        require_profile(positional[5], primary["profile_sha256"])
        verify(pins)
        receipt["profile_checked_before_forward"] = True
        return original(*positional, **keywords)

    producer.NativeAVCorrespondence = factory
    try:
        producer.main()
        current = json.loads((args.output / "result.json").read_text(encoding="utf-8"))
        require_profile(current["profile_sha256"], primary["profile_sha256"])
        verify(pins)
        if (
            current["status"] != "PASS_NATIVE_AV_CORRESPONDENCE_RUNTIME"
            or not receipt["profile_checked_before_forward"]
        ):
            raise ValueError("complete profile-bound replay required")
        receipt.update(
            status="PASS_STRICT_RAW_AV_REPLAY_PROVENANCE",
            replay_receipt_sha256=sha(args.output / "result.json"),
            input_pin_count=len(pins),
        )
    except Exception:
        receipt.update(status="FAIL_INCOMPLETE", error=traceback.format_exc())
        raise
    finally:
        producer.NativeAVCorrespondence = original
        receipt.update(seconds=time.perf_counter() - tick, ended_utc=datetime.now(UTC).isoformat())
        if args.output.exists():
            (args.output / "replay-provenance.json").write_text(
                json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
            )
    print(receipt["status"], flush=True)


if __name__ == "__main__":
    main()
