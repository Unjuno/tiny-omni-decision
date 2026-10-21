"""Hash-gated publication wrapper; preserve the already executed original renderer."""

import hashlib
import json
import subprocess
import time
from pathlib import Path

from scripts import publish_lombard_temporal_results as original


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(pins):
    for path, expected in pins.items():
        if sha(path) != expected:
            raise ValueError("publication input hash differs: " + str(path))


def main():
    tick = time.perf_counter()
    destination = original.ROOT / "lombard-temporal-checked-publication-01.json"
    if destination.exists():
        raise ValueError("new checked-publication receipt required")
    pins = {
        str(original.ROOT / name / "result.json"): digest
        for name, digest in original.EXPECTED.items()
    }
    verify(pins)
    audit = original.read("lombard-temporal-evaluation-01")
    fit = original.read("lombard-temporal-full-01")
    pins.update(
        {
            str(original.ROOT / "lombard-temporal-evaluation-01" / name): digest
            for name, digest in audit["artifact_hashes"].items()
        }
    )
    pins.update(fit["validated_hashes"])
    pins.update(fit["full_execution_hashes"])
    for path in [
        Path(original.__file__),
        Path(__file__),
        Path("tests/test_lombard_temporal_publication.py"),
    ]:
        pins[str(path.resolve())] = sha(path)
    verify(pins)
    original.main()
    verify(pins)
    outputs = [original.DOCS / (original.PREFIX + suffix) for suffix in [".json", ".md"]]
    outputs.append(original.DOCS / "R2_LOMBARD_TEMPORAL_CURVES_20261015.csv")
    receipt = dict(
        status="PASS_HASH_BOUND_CONTROL_AND_RECEIPT_PUBLICATION",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        input_files_verified_before_and_after=len(pins),
        input_hashes=pins,
        output_hashes={str(p.resolve()): sha(p) for p in outputs},
        seconds=time.perf_counter() - tick,
        model_updates=0,
        original_renderer_unchanged=True,
        legacy_training_unchanged=True,
        goal="active unmet",
    )
    destination.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(receipt["status"], flush=True)


if __name__ == "__main__":
    main()
