"""Read only registered development audio; preserve a complete physical-input receipt."""

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import psutil

from scripts.native_audio_sources import audio_plan, iter_audio

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
REPO = Path("C:/CodexWorktrees/pretrained-reuse-omni")
PINNED = {
    ROOT
    / "lombard-grid-layout-02-media-gated/train.json": (
        "259f3ee36aa956618a89819338c2ecb67b194c82dbf636df70b551e00a5d78b0"
    ),
    ROOT
    / "lombard-grid-layout-02-media-gated/validation.json": (
        "98d81114e73dd4924c087df08873d21648a2ae34a9cc289887eb0767ded4658d"
    ),
    ROOT
    / "lombard-grid-media-integrity-01/rows.json": (
        "af7217e9323cc4a2d261baec65ccec23177e1889ab0adde25575036f9cfddce4"
    ),
    ROOT
    / "starss23-all-windows-v1/train.jsonl": (
        "2d06c0e94ea68d9b5372eb942f37cd87fcf4d9ba71f05ae71bb3be836ae73347"
    ),
    ROOT
    / "starss23-all-windows-v1/validation.jsonl": (
        "20f43eeb0e123b19809a1bf4e9f639f5af850481ba4164c77a3a4b22c603fbc9"
    ),
    ROOT
    / "lombard-grid-archives-01/audio.zip": (
        "e36707602fdbeaa2e2284acb29bba883261d5e86857f694ac40cbb56cdcc5e10"
    ),
    REPO
    / "manifests/pretrained_reuse/lombard-grid-source-20261014.json": (
        "5876e57a4c71108fe3483fde6e18f3977121942c9208567488f042247626577d"
    ),
    ROOT
    / "starss23-source-audit-01/LICENSE": (
        "af4138d4fe3ccd91970f7ac1f22ae3f513ac570468a26f0f516540b5443b8745"
    ),
    ROOT
    / "starss23-source-audit-01/README.md": (
        "9401015b18d0bbb2214668f0fc5a49f82005124d036808b284ccd27678c21716"
    ),
}


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    OUTPUT = parser.parse_args().output
    OUTPUT.mkdir(exist_ok=False)
    tick = time.perf_counter()
    result = dict(
        status="RUNNING",
        started_utc=datetime.now(UTC).isoformat(),
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        exact_runner_sha256=sha(Path(__file__)),
        completed=0,
        gpu_used=False,
        models_loaded=False,
        optimizer_updates=0,
        calibration_payloads_loaded=False,
        sealed_audit_loaded=False,
        media_metadata_includes_reserved_ids=True,
        max_seconds=600,
        max_rss_bytes=2 * 2**30,
        disk_reserve_bytes=2 * 2**30,
    )
    pins = PINNED.copy()
    for name in ["native_audio_sources.py", "lombard_grid_corpus.py", "starss23_stream.py"]:
        path = REPO / "scripts" / name
        pins[path] = sha(path)
    try:
        for path, expected in pins.items():
            if sha(path) != expected:
                raise ValueError("input/source hash differs: " + str(path))
        source = load(REPO / "manifests/pretrained_reuse/lombard-grid-source-20261014.json")
        if source["policy"] != "ALLOW" or not source["derivative_model_training_allowed"]:
            raise ValueError("Lombard source no longer admitted")
        lombard = {
            s: load(ROOT / "lombard-grid-layout-02-media-gated" / f"{s}.json")
            for s in ("train", "validation")
        }
        starss = {
            s: [
                json.loads(line)
                for line in (ROOT / "starss23-all-windows-v1" / f"{s}.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            for s in ("train", "validation")
        }
        plan = audio_plan(lombard, load(ROOT / "lombard-grid-media-integrity-01/rows.json"), starss)
        if len(plan["rows"]) != 7410:
            raise ValueError("frozen example count differs")
        write(OUTPUT / "plan.json", plan)
        result.update(
            counts=plan["counts"],
            shapes=plan["shapes"],
            tensor_bytes=plan["tensor_bytes"],
            example_order_sha256=hashlib.sha256(
                "\n".join(r["id"] for r in plan["rows"]).encode()
            ).hexdigest(),
        )
        physical = {
            Path(r["path"]): r["media_sha256"] for r in plan["rows"] if r["source"] == "starss"
        }
        pins.update(physical)
        pcm_sets = {s: set() for s in ("train", "validation")}
        silent = {s: 0 for s in ("train", "validation")}
        with (OUTPUT / "decoded.jsonl").open("x", encoding="utf-8") as stream:
            for row, pcm, digest in iter_audio(
                plan["rows"], ROOT / "lombard-grid-archives-01/audio.zip"
            ):
                pcm_sets[row["split"]].add((row["rate"], digest))
                silent[row["split"]] += int(not np.any(pcm))
                stream.write(
                    json.dumps(
                        dict(
                            id=row["id"],
                            split=row["split"],
                            rate=row["rate"],
                            samples=len(pcm),
                            pcm_sha256=digest,
                            minimum=float(pcm.min()),
                            maximum=float(pcm.max()),
                            rms=float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2))),
                        ),
                        allow_nan=False,
                    )
                    + "\n"
                )
                result["completed"] += 1
                peak = psutil.Process().memory_info().peak_wset
                if (
                    peak > result["max_rss_bytes"]
                    or time.perf_counter() - tick > result["max_seconds"]
                    or shutil.disk_usage(OUTPUT).free < result["disk_reserve_bytes"]
                ):
                    raise RuntimeError("registered resource cap exceeded")
                result["os_peak_working_set_bytes"] = peak
                if result["completed"] % 512 == 0:
                    print("decoded", result["completed"], "/7410", flush=True)
                    write(OUTPUT / "result.json", result)
        for path, expected in pins.items():
            if sha(path) != expected:
                raise ValueError("input/source changed: " + str(path))
        result.update(
            status="PASS_NATIVE_AUDIO_CORPUS",
            unique_decoded_pcm={s: len(v) for s, v in pcm_sets.items()},
            cross_split_decoded_pcm_overlap=len(pcm_sets["train"] & pcm_sets["validation"]),
            silent_examples=silent,
            unique_physical_assets=sum(
                v["assets"] for d in plan["counts"].values() for v in d.values()
            ),
            validated_hashes={str(k): v for k, v in pins.items()},
            artifact_hashes={
                p.name: sha(p) for p in OUTPUT.iterdir() if p.is_file() and p.name != "result.json"
            },
        )
    except BaseException as error:
        result.update(status="INCOMPLETE", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        result.update(
            finished_utc=datetime.now(UTC).isoformat(), seconds=time.perf_counter() - tick
        )
        write(OUTPUT / "result.json", result)
        print(result["status"], result["completed"], result["seconds"], flush=True)


if __name__ == "__main__":
    main()
