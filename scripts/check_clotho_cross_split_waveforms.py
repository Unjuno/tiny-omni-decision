"""All cross-split whole-template correlation pairs, development media only."""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from scripts.audio_overlap_probe import PreparedWaveformMatcher
from scripts.clotho_acquisition_gate import complete_acquisition


def main():
    started = time.perf_counter()
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    media = root / "clotho-aqa-development-media-01"
    records = complete_acquisition(media, root / "clotho-aqa-development-split-01")
    decoded = root / "clotho-aqa-decoded-audit-01"
    data = (decoded / "result.json").read_bytes()
    previous = json.loads(data)
    media_sha = hashlib.sha256((media / "result.json").read_bytes()).hexdigest()
    if (previous["status"] != "PASS" or previous["media_report_sha256"] != media_sha
            or previous["sample_rate"] != 2000 or previous["duplicate_pairs"]):
        raise ValueError("decode audit is incomplete or differs from acquisition")
    if {(r["source_id"], r["split"]) for r in records} != {
            (r["source_id"], r["split"]) for r in previous["rows"]}:
        raise ValueError("decoded identities differ")
    waveforms = {}
    for row in previous["rows"]:
        a = np.load(decoded / (row["source_id"] + ".npy"), allow_pickle=False)
        if hashlib.sha256(a.astype("<f4").tobytes()).hexdigest() != row["waveform_sha256"]:
            raise ValueError("waveform cache changed")
        waveforms[row["source_id"]] = a
    output = root / "clotho-aqa-cross-split-waveform-screen-01"
    output.mkdir(exist_ok=False)
    pairs = [(left, right) for i, left in enumerate(records) for right in records[i + 1:]
             if left["split"] != right["split"]]
    prepare_started = time.perf_counter()
    matcher = PreparedWaveformMatcher(waveforms, memory_limit_bytes=3 * 1024**3)
    prepare_seconds = time.perf_counter() - prepare_started
    report = dict(status="RUNNING", scope="all cross-split whole-template pairs",
                  review_threshold=0.99, expected_pairs=len(pairs), pairs=0,
                  flagged_pairs=0, maximum_correlation=0.0,
                  media_report_sha256=media_sha,
                  decoded_report_sha256=hashlib.sha256(data).hexdigest(), workers=4,
                  prepared_cache_bytes=matcher.cache_bytes, cache_limit_bytes=3 * 1024**3,
                  preparation_seconds=prepare_seconds,
                  limitation="Does not exclude edits, mixing, time-stretch or shorter reuse")

    def score(pair):
        left, right = pair
        a, b = waveforms[left["source_id"]], waveforms[right["source_id"]]
        if len(a) > len(b):
            left, right, a, b = right, left, b, a
        return dict(template=left["source_id"], recording=right["source_id"],
                    **matcher.match(left["source_id"], right["source_id"]))

    def save():
        report["seconds"] = time.perf_counter() - started
        temporary = output / "progress.tmp"
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(output / "progress.json")

    save()
    try:
        with (output / "pairs.jsonl").open("w", encoding="utf-8") as stream:
            with ThreadPoolExecutor(4) as pool:
                # Bounded submissions prevent retaining a Future for every pair.
                for start in range(0, len(pairs), 512):
                    for result in pool.map(score, pairs[start:start + 512]):
                        stream.write(json.dumps(result) + "\n")
                        value = result["absolute_correlation"]
                        report["pairs"] += 1
                        report["flagged_pairs"] += int(value is None or value >= 0.99)
                        if value is not None:
                            report["maximum_correlation"] = max(
                                value, report["maximum_correlation"])
                    stream.flush()
                    save()
        if report["pairs"] != report["expected_pairs"]:
            raise ValueError("pair screen incomplete")
        report["status"] = "COMPLETE"
    except Exception as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        save()
        report["pairs_sha256"] = hashlib.sha256((output / "pairs.jsonl").read_bytes()).hexdigest()
        (output / "result.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
