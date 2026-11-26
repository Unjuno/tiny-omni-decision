"""Read official split lists and three fixed TRAIN annotation files plus one AV clip."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from scripts.greatest_hits_source import CurlRangeSource, member_bytes, parse_ids, select_train

URL = "https://web.eecs.umich.edu/~ahowens/vis/vis-data-256.zip"
SIZE = 21_142_601_241
ETAG = '"4ec327e19-5cc7fa3687c80"'
DIRECTORY_SHA = "98673a0101020606537d2850f825ebe08219c496928002f2cf09dcaa83b9824a"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo).strip():
        raise ValueError("clean committed source required")
    if sha(args.directory) != DIRECTORY_SHA:
        raise ValueError("directory evidence changed")
    entries = json.loads(args.directory.read_text())
    lookup = {e["name"]: e for e in entries}
    if len(lookup) != len(entries):
        raise ValueError("duplicate ZIP names")
    args.output.mkdir(exist_ok=False)
    source = CurlRangeSource(URL, SIZE, ETAG, args.output / "ranges")
    started = time.monotonic()
    result = dict(status="RUNNING", command=[sys.executable, *sys.argv],
                  source_commit=subprocess.check_output(
                      ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                  url=URL, archive_size=SIZE, etag=ETAG,
                  directory_sha256=DIRECTORY_SHA, selected=[], fetched={},
                  test_payload_opened=False, sealed_audit_opened=False,
                  optimizer_updates=0, model_loaded=False)
    try:
        def fetch(name):
            entry = lookup["vis-data-256/" + name]
            data = member_bytes(entry, source.read)
            (args.output / name).write_bytes(data)
            result["fetched"][name] = dict(sha256=hashlib.sha256(data).hexdigest(),
                                          bytes=len(data), crc32=entry["crc32"])
            return data

        train, test = parse_ids(fetch("train.txt")), parse_ids(fetch("test.txt"))
        parents = {Path(n).name.removesuffix("_times.txt")
                   for n in lookup if n.endswith("_times.txt")}
        selected = select_train(train, test, parents)
        result.update(train_parents=len(train), test_parents=len(test), selected=selected,
                      selection="first3 SHA256('17:' + official TRAIN ID)")
        for ident in selected:
            fetch(ident + "_times.txt")
        fetch(selected[0] + "_denoised.mp4")
        if sha(args.directory) != DIRECTORY_SHA:
            raise ValueError("directory changed during probe")
        result["status"] = "PASS_FIXED_TRAIN_MEMBER_ACQUISITION"
    except Exception as exc:
        result.update(status="FAILED", error=repr(exc))
        raise
    finally:
        result.update(seconds=time.monotonic() - started, utc=datetime.now(UTC).isoformat(),
                      requested_bytes=source.requested, requests=source.requests,
                      source_hashes={p: sha(repo / p) for p in [
                          "scripts/greatest_hits_source.py",
                          "scripts/probe_greatest_hits_train.py"]})
        (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k not in {"requests", "fetched"}}))


if __name__ == "__main__":
    main()
