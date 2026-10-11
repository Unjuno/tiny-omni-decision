"""Replay two pinned saved tensors through the JAX Python; synthetic RAM capacity check."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from tiny_omni_decision.feature_cache import BoundedFeatureCache
from tiny_omni_decision.token_transport import read_tokens, write_tokens

SHAPE = (8, 16, 16, 768)
HASHES = [
    "e1e2d5307da6a7af6447ae14d9b81924ac4e1a5434ebae4f9009954168bcca17",
    "53cd5745bb1706668796893eb3256507816915a8cb43418245866d4c5ec39ddf",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--worker-python", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    paths = [args.audit / f"tokens-{i}.npy" for i in range(2)]
    if [sha(p) for p in paths] != HASHES:
        raise ValueError("pinned saved tensors changed")
    if args.worker:
        for i, path in enumerate(paths):
            write_tokens(sys.stdout.buffer, str(i), np.load(path, allow_pickle=False))
        return
    import psutil

    if args.output is None or args.output.exists() or args.worker_python is None:
        raise ValueError("new output and existing worker Python required")
    available = psutil.virtual_memory().available
    required = 456 * int(np.prod(SHAPE)) * 4
    if available < required + 8 * 1024**3:
        raise ValueError("insufficient RAM with 8 GiB reserve")
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    config = dict(
        argv=sys.argv,
        started_utc=datetime.now(UTC).isoformat(),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        script_sha256=sha(Path(__file__)),
        input_file_sha256=HASHES,
        worker_python=str(args.worker_python),
        available_ram_bytes=available,
        training=False,
        quality_evaluation=False,
        live_encoder_extraction=False,
    )
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    bank = np.empty((456, *SHAPE), dtype=np.float32)
    bank.fill(0)  # Touch every page: capacity measurement, NOT 456 extracted scenes.
    cache = BoundedFeatureCache(456)
    command = [
        str(args.worker_python),
        "-m",
        "scripts.audit_token_ram_transport",
        "--audit",
        str(args.audit),
        "--worker",
    ]
    records = []
    with (args.output / "worker-stderr.log").open("wb") as stderr:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr)
        try:
            for index in range(2):
                record = read_tokens(child.stdout, SHAPE)
                if record is None or record[0] != str(index):
                    raise ValueError("missing/out-of-order asset")
                bank[index] = record[1]
                original = np.load(paths[index], allow_pickle=False)
                np.testing.assert_array_equal(bank[index], original)
                feature = bank[index].view()
                feature.flags.writeable = False
                digest = hashlib.sha256(feature.tobytes()).hexdigest()
                cache.put(record[0], feature, digest)
                first, _ = cache.get(record[0])
                second, _ = cache.get(record[0])
                if first is not second or not np.shares_memory(first, bank):
                    raise ValueError("per-question copy detected")
                records.append(
                    dict(
                        key=record[0],
                        exact=True,
                        payload_sha256=digest,
                        repeated_lookup_same_object=True,
                    )
                )
            if read_tokens(child.stdout, SHAPE) is not None or child.wait(timeout=30) != 0:
                raise ValueError("unexpected worker output/exit")
        finally:
            if child.poll() is None:
                child.kill()  # This script owns this replay worker only.
                child.wait()
            child.stdout.close()
    result = dict(
        status="PASS_TRANSPORT_CAPACITY_ONLY",
        records=records,
        allocated_slots=456,
        real_saved_assets_loaded=2,
        synthetic_zero_slots=454,
        bank_bytes=bank.nbytes,
        process_memory=psutil.Process().memory_info()._asdict(),
        available_ram_after_bytes=psutil.virtual_memory().available,
        seconds=time.perf_counter() - started,
        worker_command=command,
        finished_utc=datetime.now(UTC).isoformat(),
        cost_usd=0,
        limitation="Saved-token replay only; no full extraction, training, or quality result.",
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
