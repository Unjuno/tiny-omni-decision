"""Extract frozen DyMN features independently for all four STARSS23 FOA channels."""

from __future__ import annotations

import argparse
import json
import platform
import time
from itertools import groupby
from pathlib import Path

import numpy as np

from scripts.extract_frozen_videoprism_features import sha256_file
from scripts.extract_starss23_corpus_features import read_manifest
from scripts.extract_starss23_features import AUDIO_SHA, AUDIO_SOURCE, load_encoder
from scripts.fetch_starss23_frozen_media import write_json
from scripts.starss23_stream import audio_windows

MANIFEST_DIR = Path("C:/CodexArtifacts/pretrained-reuse-r2/starss23-all-windows-v1")
REFERENCE_RESULT = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2/starss23-audio-corpus-features-v1/result.json"
)
REFERENCE_FEATURES = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2/starss23-audio-corpus-features-v1/features.npy"
)
EXPECTED_FEATURE_ROWS = 2720
FOA_CHANNEL_ORDER = ("W", "Y", "Z", "X")


def encode_foa_window(audio: np.ndarray, encode) -> np.ndarray:
    """Encode each FOA channel with the same frozen mono frontend, in source order."""
    values = np.asarray(audio, dtype=np.float32)
    if values.shape != (120000, 4) or not np.isfinite(values).all():
        raise ValueError("expected finite five-second 24-kHz four-channel FOA audio")
    channels = [encode({"audio": values[:, index : index + 1]}) for index in range(4)]
    features = np.stack(channels).astype(np.float32)
    if features.shape != (4, 384) or not np.isfinite(features).all():
        raise ValueError("unexpected per-channel DyMN feature shape or values")
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    rows = read_manifest(MANIFEST_DIR)
    if len(rows) != EXPECTED_FEATURE_ROWS:
        raise ValueError(f"expected {EXPECTED_FEATURE_ROWS} windows; found {len(rows)}")
    if sha256_file(args.checkpoint) != AUDIO_SHA:
        raise ValueError("pinned DyMN checkpoint hash mismatch")
    import subprocess

    revision = subprocess.check_output(
        ["git", "-C", str(args.source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != AUDIO_SOURCE:
        raise ValueError("pinned audio model source revision mismatch")

    reference = json.loads(REFERENCE_RESULT.read_text(encoding="utf-8"))
    if reference.get("status") != "PASS" or reference.get("feature_shape") != [2720, 384]:
        raise ValueError("existing channel-0 audio reference is not complete")
    if len(reference.get("features", [])) != len(rows):
        raise ValueError("reference audio feature index row count mismatch")
    reference_ids = [row["id"] for row in reference["features"]]
    manifest_ids = [row["id"] for row in rows]
    if reference_ids != manifest_ids:
        raise ValueError("reference audio feature order differs from fixed window manifest")
    reference_matrix = np.load(REFERENCE_FEATURES, mmap_mode="r", allow_pickle=False)
    if reference_matrix.shape != (len(rows), 384):
        raise ValueError("reference audio matrix has an unexpected shape")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {
        "status": "RUNNING",
        "modality": "audio",
        "representation": "four independently encoded FOA channels",
        "channel_order": list(FOA_CHANNEL_ORDER),
        "checkpoint_sha256": AUDIO_SHA,
        "source_commit": AUDIO_SOURCE,
        "manifest_result_sha256": sha256_file(MANIFEST_DIR / "result.json"),
        "reference_result_sha256": sha256_file(REFERENCE_RESULT),
        "reference_features_sha256": sha256_file(REFERENCE_FEATURES),
        "runner_sha256": sha256_file(Path(__file__)),
        "python": platform.python_version(),
        "device": "cpu",
        "precision": "float32",
        "feature_shape": [len(rows), 4, 384],
        "windows": [],
        "recording_shards": [],
        "training": False,
        "annotation_labels_read": False,
        "dev_test_or_evaluation_or_sealed_audit_media_read": False,
    }
    (args.output_dir / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
    try:
        encode, mono_shape = load_encoder(
            argparse.Namespace(
                modality="audio",
                source_root=args.source_root,
                checkpoint=args.checkpoint,
            ),
            report,
        )
        if mono_shape != (384,):
            raise ValueError("unexpected pinned mono DyMN feature shape")
        report["preprocessing"] = (
            "each original PCM16 FOA channel separately; 24kHz to 32kHz sinc resample; "
            "official evaluation mel and frozen DyMN"
        )
        vectors = []
        for recording, group in groupby(rows, key=lambda row: row["recording"]):
            group = list(group)
            first_index = len(vectors)
            media = group[0]["media"]["audio"]
            if any(row["media"]["audio"] != media for row in group):
                raise ValueError("recording maps to inconsistent audio media")
            media_path = Path(media["path"])
            if sha256_file(media_path) != media["sha256"]:
                raise ValueError(f"audio media hash mismatch: {recording}")
            windows = [(row["start_seconds"], row["end_seconds"]) for row in group]
            for row, audio in zip(group, audio_windows(media_path, windows), strict=True):
                tick = time.perf_counter()
                features = encode_foa_window(audio, encode)
                if not np.allclose(
                    features[0], reference_matrix[len(vectors)], rtol=1e-5, atol=1e-6
                ):
                    raise ValueError(f"channel-0 reference parity failed for {row['id']}")
                vectors.append(features)
                report["windows"].append(
                    {
                        "id": row["id"],
                        "recording": recording,
                        "room": row["room"],
                        "split": row["split"],
                        "media_sha256": media["sha256"],
                        "channel0_reference_max_abs_diff": float(
                            np.max(np.abs(features[0] - reference_matrix[len(vectors) - 1]))
                        ),
                        "encoder_seconds": time.perf_counter() - tick,
                    }
                )
            shard = np.stack(vectors[first_index:]).astype(np.float32)
            shard_path = args.output_dir / f"recording-{len(report['recording_shards']):03d}.npy"
            np.save(shard_path, shard, allow_pickle=False)
            if not np.array_equal(shard, np.load(shard_path, allow_pickle=False)):
                raise ValueError("recording shard reload mismatch")
            report["recording_shards"].append(
                {
                    "recording": recording,
                    "first_index": first_index,
                    "count": len(group),
                    "path": shard_path.name,
                    "sha256": sha256_file(shard_path),
                }
            )
            write_json(args.output_dir / "progress.json", report)
            print(
                json.dumps({"recording": recording, "features": len(vectors), "total": len(rows)}),
                flush=True,
            )

        features = np.stack(vectors).astype(np.float32)
        if features.shape != (len(rows), 4, 384) or not np.isfinite(features).all():
            raise ValueError("full audio feature matrix shape or values are invalid")
        feature_path = args.output_dir / "features.npy"
        np.save(feature_path, features, allow_pickle=False)
        if not np.array_equal(features, np.load(feature_path, allow_pickle=False)):
            raise ValueError("feature matrix reload mismatch")
        report.update(
            status="PASS",
            feature_sha256=sha256_file(feature_path),
            wall_seconds=time.perf_counter() - started,
        )
    except Exception as error:
        report.update(
            status="FAIL",
            error=f"{type(error).__name__}: {error}",
            wall_seconds=time.perf_counter() - started,
        )
        raise
    finally:
        write_json(args.output_dir / "result.json", report)
        write_json(args.output_dir / "progress.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
