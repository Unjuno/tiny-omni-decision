"""Re-extract pinned video tokens into RAM without changing existing artifacts."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import psutil

from scripts.range_generalization import sha, write
from scripts.train_shared_text_video import (
    TOKEN_REFERENCE,
    TOKEN_REFERENCE_HASH,
    worker_environment,
)
from tiny_omni_decision.token_transport import read_tokens


def extract(rows, pooled, output):
    keys = list(dict.fromkeys(r["media_sha256"] for data in rows.values() for r in data))
    if sha(TOKEN_REFERENCE) != TOKEN_REFERENCE_HASH:
        raise ValueError("token reference changed")
    expected = {
        r["media_sha256"]: r["token_sha256"] for r in json.loads(TOKEN_REFERENCE.read_text())
    }
    if psutil.virtual_memory().available < len(keys) * 2048 * 768 * 4 + 8 * 1024**3:
        raise ValueError("insufficient RAM reserve")
    write(output / "request.json", keys)
    bank = np.empty((len(keys), 2048, 768), dtype=np.float32)
    repo = Path(__file__).resolve().parents[1]
    command = [
        "C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-r1/venv/Scripts/python.exe",
        "-m",
        "scripts.audit_token_ram_transport",
        "--audit",
        "C:/CodexArtifacts/pretrained-reuse-r2/videoprism-token-audit-02",
        "--request",
        str(output / "request.json"),
        "--worker",
        "--live",
    ]
    records = []
    with (output / "worker.log").open("wb") as log:
        child = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=log, cwd=repo, env=worker_environment(repo)
        )
        try:
            for i, key in enumerate(keys):
                record = read_tokens(child.stdout, (8, 16, 16, 768))
                if record is None or record[0] != str(i):
                    raise ValueError("worker token order mismatch")
                bank[i] = record[1].reshape(2048, 768)
                delta = float(np.abs(bank[i].mean(0) - pooled[key]).max())
                digest = hashlib.sha256(bank[i].tobytes()).hexdigest()
                if delta > 1e-5 or digest != expected.get(key):
                    raise ValueError("token identity or pooled parity failed")
                records.append(dict(media_sha256=key, token_sha256=digest, pooled_max_abs=delta))
                write(output / "extraction.json", records)
                if i % 16 == 0:
                    print("extracted", i + 1, "/", len(keys), flush=True)
            if read_tokens(child.stdout, (8, 16, 16, 768)) is not None or child.wait(30) != 0:
                raise ValueError("worker did not complete cleanly")
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
            child.stdout.close()
    return keys, bank
