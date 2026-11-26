"""Screen preregistered Clotho publisher-validation clips against used audio."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import av
import numpy as np

from scripts.audio_overlap_probe import PreparedWaveformMatcher
from scripts.clotho_acquisition_gate import complete_acquisition
from scripts.extract_frozen_videoprism_features import sha256_file
from scripts.fetch_clotho_publisher_val_media import MANIFEST_SHA256, load_manifest

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
ATTEMPT = ROOT / "clotho-aqa-publisher-val-20261126-attempt-01"
MEDIA = ATTEMPT / "media"
OUTPUT = ATTEMPT / "waveform-overlap"
RATE = 2_000
THRESHOLD = 0.99
MEMORY_LIMIT = 1536 * 1024**2
REFERENCE_BLOCK = 16


def decode_mono(path: Path) -> np.ndarray:
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        converter = av.AudioResampler(format="fltp", layout="mono", rate=RATE)
        chunks = []
        for frame in container.decode(stream):
            chunks.extend(item.to_ndarray().ravel() for item in converter.resample(frame))
        chunks.extend(item.to_ndarray().ravel() for item in converter.resample(None))
    waveform = np.concatenate(chunks).astype("<f4", copy=False) if chunks else np.array([])
    if waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
        raise ValueError(f"invalid decoded waveform: {path.name}")
    return waveform


def wave_sha256(waveform: np.ndarray) -> str:
    canonical = np.asarray(waveform, dtype="<f4")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def oriented_match(matcher, left_id, left, right_id, right):
    if len(left) <= len(right):
        return matcher.match(left_id, right_id)
    return matcher.match(right_id, left_id)


def load_corpus() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    acquisition_path = MEDIA / "result.json"
    acquisition_bytes = acquisition_path.read_bytes()
    acquisition = json.loads(acquisition_bytes)
    manifest = load_manifest()
    manifest_sources = {row["source_id"] for row in manifest}
    acquired_sources = {row["source_id"] for row in acquisition["files"]}
    if (
        acquisition["status"] != "PASS"
        or len(acquisition["files"]) != 61
        or acquired_sources != manifest_sources
        or acquisition["manifest_sha256"] != MANIFEST_SHA256
    ):
        raise ValueError("publisher validation media acquisition is incomplete")
    candidate = {}
    for row in acquisition["files"]:
        path = MEDIA / f"{row['source_id']}.wav"
        if sha256_file(path) != row["sha256"]:
            raise ValueError("publisher validation audio hash mismatch")
        candidate[f"val:{row['source_id']}"] = decode_mono(path)

    used = {}
    dev_media = ROOT / "clotho-aqa-development-media-02"
    dev_records = complete_acquisition(dev_media, ROOT / "clotho-aqa-development-split-01")
    dev_decoded = ROOT / "clotho-aqa-decoded-audit-01"
    dev_report_bytes = (dev_decoded / "result.json").read_bytes()
    dev_report = json.loads(dev_report_bytes)
    if dev_report["status"] != "PASS" or dev_report["sample_rate"] != RATE:
        raise ValueError("existing Clotho decoded audit is incomplete or incompatible")
    dev_rows = {row["source_id"]: row for row in dev_report["rows"]}
    for row in dev_records:
        source_id = row["source_id"]
        item = dev_rows[source_id]
        wave = np.load(dev_decoded / f"{source_id}.npy", allow_pickle=False)
        if wave_sha256(wave) != item["waveform_sha256"]:
            raise ValueError("existing Clotho waveform cache hash mismatch")
        used[f"clotho:{source_id}"] = wave

    fsd = ROOT / "fsd50k-training-128-02"
    fsd_report = json.loads((fsd / "result.json").read_text(encoding="utf-8"))
    if fsd_report["status"] != "PASS" or len(fsd_report["files"]) != 128:
        raise ValueError("existing FSD training acquisition is incomplete")
    for row in fsd_report["files"]:
        path = fsd / f"{row['id']}.wav"
        if sha256_file(path) != row["sha256"]:
            raise ValueError("existing FSD waveform file hash mismatch")
        used[f"fsd:{row['id']}"] = decode_mono(path)

    provenance = {
        "publisher_media_report_sha256": hashlib.sha256(acquisition_bytes).hexdigest(),
        "clotho_media_report_sha256": sha256_file(dev_media / "result.json"),
        "clotho_decoded_report_sha256": hashlib.sha256(dev_report_bytes).hexdigest(),
        "fsd_media_report_sha256": sha256_file(fsd / "result.json"),
        "candidate_count": len(candidate),
        "used_count": len(used),
    }
    return candidate, used, provenance


def run(output: Path = OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    candidate, used, provenance = load_corpus()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    candidate_hashes = {key: wave_sha256(value) for key, value in candidate.items()}
    used_hashes = {key: wave_sha256(value) for key, value in used.items()}
    exact_pairs = []
    used_by_hash = {}
    for key, digest in used_hashes.items():
        used_by_hash.setdefault(digest, []).append(key)
    for key, digest in candidate_hashes.items():
        for other in used_by_hash.get(digest, []):
            exact_pairs.append({"candidate": key, "used": other, "wave_sha256": digest})
    if len(set(candidate_hashes.values())) != len(candidate_hashes):
        raise ValueError("duplicate canonical waveforms within publisher-validation candidates")

    report = {
        "status": "RUNNING",
        "scope": "all candidate-vs-used and candidate-vs-candidate whole-template pairs",
        "sample_rate": RATE,
        "threshold": THRESHOLD,
        "groups": provenance,
        "expected_cross_pairs": len(candidate) * len(used),
        "completed_cross_pairs": 0,
        "expected_candidate_pairs": len(candidate) * (len(candidate) - 1) // 2,
        "completed_candidate_pairs": 0,
        "flagged_pairs": len(exact_pairs),
        "exact_waveform_pairs": exact_pairs,
        "max_correlations": [],
        "memory_limit_bytes": MEMORY_LIMIT,
        "limitations": [
            "2-kHz mono whole-template matching; edits and short reuse may escape.",
            "EfficientAT pretraining item IDs are unavailable, so pretraining overlap is unknown.",
        ],
        "started_epoch": time.time(),
    }
    pairs_path = output / "pairs.jsonl"
    progress_path = output / "progress.json"

    def save_progress() -> None:
        report["seconds"] = time.perf_counter() - started
        temporary = output / "progress.tmp"
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(progress_path)

    try:
        with pairs_path.open("w", encoding="utf-8") as stream:
            # The small validation side is held fixed; used recordings are prepared in
            # bounded blocks so the FFT cache stays below the host-memory budget.
            nearest = {
                key: {"candidate": key, "absolute_correlation": -1.0}
                for key in candidate
            }
            candidate_matcher = PreparedWaveformMatcher(
                candidate, memory_limit_bytes=MEMORY_LIMIT
            )
            candidate_ids = list(candidate)
            for index, left_id in enumerate(candidate_ids):
                for right_id in candidate_ids[index + 1 :]:
                    if len(candidate[left_id]) <= len(candidate[right_id]):
                        score = oriented_match(
                            candidate_matcher,
                            left_id,
                            candidate[left_id],
                            right_id,
                            candidate[right_id],
                        )
                    else:
                        score = oriented_match(
                            candidate_matcher,
                            right_id,
                            candidate[right_id],
                            left_id,
                            candidate[left_id],
                        )
                    report["completed_candidate_pairs"] += 1
                    correlation = score["absolute_correlation"]
                    if correlation is None or correlation >= THRESHOLD:
                        report["flagged_pairs"] += 1
                        stream.write(
                            json.dumps(
                                {"candidate": left_id, "used": right_id, **score}
                            )
                            + "\n"
                        )
                    if correlation is not None:
                        for key, other in ((left_id, right_id), (right_id, left_id)):
                            if correlation > nearest[key]["absolute_correlation"]:
                                nearest[key] = {
                                    "candidate": key,
                                    "used": other,
                                    **score,
                                }

            used_items = list(used.items())
            for block_start in range(0, len(used_items), REFERENCE_BLOCK):
                reference_block = dict(used_items[block_start : block_start + REFERENCE_BLOCK])
                matcher = PreparedWaveformMatcher(
                    {**candidate, **reference_block},
                    memory_limit_bytes=MEMORY_LIMIT,
                )
                for candidate_id, waveform in candidate.items():
                    for used_id, reference in reference_block.items():
                        score = oriented_match(
                            matcher, candidate_id, waveform, used_id, reference
                        )
                        report["completed_cross_pairs"] += 1
                        correlation = score["absolute_correlation"]
                        if correlation is None or correlation >= THRESHOLD:
                            report["flagged_pairs"] += 1
                            stream.write(
                                json.dumps(
                                    {"candidate": candidate_id, "used": used_id, **score}
                                )
                                + "\n"
                            )
                        if (
                            correlation is not None
                            and correlation > nearest[candidate_id]["absolute_correlation"]
                        ):
                            nearest[candidate_id] = {
                                "candidate": candidate_id,
                                "used": used_id,
                                **score,
                            }
                stream.flush()
                save_progress()
            report["max_correlations"] = list(nearest.values())
        if (
            report["completed_cross_pairs"] != report["expected_cross_pairs"]
            or report["completed_candidate_pairs"] != report["expected_candidate_pairs"]
        ):
            raise ValueError("waveform pair screen incomplete")
        report["status"] = "COMPLETE"
    except Exception as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        report["ended_epoch"] = time.time()
        report["seconds"] = time.perf_counter() - started
        report["pairs_sha256"] = sha256_file(pairs_path)
        (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
