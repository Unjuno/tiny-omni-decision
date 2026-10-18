"""Bind complete admitted train/development questions to immutable native banks."""

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from scripts.native_common_corpus import build_corpus
from scripts.native_video_cache import sha

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/pretrained_reuse/native_common_corpus_v1.json"


def write(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO).strip():
        raise ValueError("clean committed source required")
    args.output.mkdir(exist_ok=False)
    started = time.perf_counter()
    config = json.loads(CONFIG.read_text())
    pins = dict(config["input_pins"])
    report = dict(
        status="RUNNING",
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        command=sys.argv,
        config_sha256=sha(CONFIG),
        started_utc=datetime.now(UTC).isoformat(),
        optimizer_updates=0,
        quality_evaluation=False,
        calibration_loaded=False,
        sealed_audit_loaded=False,
        cloud_cost=0,
    )

    def check(path, digest=None):
        path = Path(path)
        expected = digest or pins[str(path)]
        if sha(path) != expected:
            raise ValueError("changed input: " + str(path))
        pins[str(path)] = expected
        return path

    def read(path, digest=None):
        p = check(path, digest)
        return (
            [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
            if p.suffix == ".jsonl"
            else json.loads(p.read_text(encoding="utf-8"))
        )

    try:
        for path in list(pins):
            check(path)
        roots = {k: Path(v) for k, v in config["feature_roots"].items()}
        receipts = {k: read(root / "result.json") for k, root in roots.items()}
        if any(receipts[k]["status"] != v for k, v in config["receipt_status"].items()):
            raise ValueError("all complete native feature caches required")
        language = read(
            roots["language"] / "index.json", receipts["language"]["artifact_hashes"]["index.json"]
        )
        audio = read(
            roots["audio"] / "plan.json", receipts["audio"]["artifact_hashes"]["plan.json"]
        )
        images = read(roots["image"] / "records.json", receipts["image"]["records_sha256"])
        files = config["files"]
        result = build_corpus(
            text_files={k: read(v) for k, v in files["text"].items()},
            language=language,
            images={k: read(v) for k, v in files["image"].items()},
            image_records=images,
            lombard={k: read(v) for k, v in files["lombard"].items()},
            starss={k: read(v) for k, v in files["starss"].items()},
            audio=audio,
            videos={k: read(v) for k, v in files["video"].items()},
            video_cache=receipts["video"],
        )
        for split, expected in config["expected_questions"].items():
            if len(result["rows"][split]) != expected:
                raise ValueError("complete registered question count differs")
        bank_digests = {}
        for kind in ("language", "audio"):
            bank_digests.update(
                {
                    kind + "/" + k: v
                    for k, v in receipts[kind]["artifact_hashes"].items()
                    if k.endswith(".npy")
                }
            )
        bank_digests.update(
            {f"image/{s}-cells.npy": v for s, v in receipts["image"]["feature_hashes"].items()}
        )
        bank_digests.update(
            {"video/" + r["file"]: r["file_sha256"] for r in receipts["video"]["rows"]}
        )
        banks = {}
        for name, digest in bank_digests.items():
            kind, filename = name.split("/", 1)
            path = check(roots[kind] / filename, digest)
            bank = np.load(path, mmap_mode="r", allow_pickle=False)
            if bank.dtype != np.float32:
                raise ValueError("native bank dtype differs")
            banks[name] = dict(path=str(path), sha256=digest, shape=list(bank.shape))
        for obs in result["observations"].values():
            for feature in obs["features"]:
                shape = banks[feature["bank"]]["shape"]
                if "index" in feature:
                    if len(shape) != 3 or not 0 <= feature["index"] < shape[0]:
                        raise ValueError("invalid per-asset bank slice")
                    shape = shape[1:]
                width = 768 if feature["kind"] in ("image", "video") else 384
                if (
                    len(shape) != 2
                    or shape[1] != width
                    or feature["offset"] + feature["tokens"] > shape[0]
                ):
                    raise ValueError("invalid native token offset/shape")
        old = language["queries"]
        new = sorted(set(result["pairs"]) - set(old))
        query_map = {}
        for key in result["pairs"]:
            if key in old:
                query_map[key] = dict(bank="language/queries.npy", offset=old[key]["offset"])
        query_map.update({k: dict(bank="new/queries.npy", offset=i) for i, k in enumerate(new)})
        write(args.output / "observations.json", result["observations"])
        write(args.output / "pairs.json", result["pairs"])
        write(args.output / "query-map.json", query_map)
        write(args.output / "banks.json", banks)
        for split, rows in result["rows"].items():
            with (args.output / (split + ".jsonl")).open("x", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        for path, digest in pins.items():
            check(path, digest)
        report.update(
            status="PASS_COMPLETE_NATIVE_COMMON_CORPUS",
            counts=result["counts"],
            observations=len(result["observations"]),
            pairs=len(result["pairs"]),
            reused_pairs=len(query_map) - len(new),
            new_pairs=len(new),
            validated_hashes=pins,
            artifacts={p.name: sha(p) for p in args.output.iterdir()},
            feature_banks=len(banks),
            feature_roots=config["feature_roots"],
        )
    except Exception as exc:
        report.update(status="FAILED", failure=repr(exc))
        raise
    finally:
        report.update(
            seconds=time.perf_counter() - started, finished_utc=datetime.now(UTC).isoformat()
        )
        write(args.output / "result.json", report)
    print(
        json.dumps(
            {k: report[k] for k in ["status", "observations", "pairs", "new_pairs", "seconds"]}
        )
    )


if __name__ == "__main__":
    main()
