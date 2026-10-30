"""Measure train-only embedded-audio / paired-WAV alignment for FAIR-Play."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

_SOURCE = "facebookresearch/FAIR-Play@2666d6caf2d126ae43a84841e94e0c9ae24b3e2d"
_SPLIT = "PseudoBinaural new_splits/split1 train only"
_MEDIA_SPLIT = "PseudoBinaural split1 train"
_SAMPLE_RATE = 48_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _locate(roots: list[Path], asset_id: str, suffix: str) -> Path:
    matches = [root / f"{asset_id}.{suffix}" for root in roots]
    matches = [path for path in matches if path.is_file()]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {asset_id}.{suffix} across media roots; "
            f"found {len(matches)}"
        )
    return matches[0]


def _decode_mono(path: Path, stream_index: int | None = None) -> tuple[Any, dict[str, Any]]:
    import av
    import numpy as np

    with av.open(str(path)) as container:
        streams = [stream for stream in container.streams if stream.type == "audio"]
        if stream_index is None:
            if len(streams) != 1:
                raise ValueError(f"expected one audio stream in {path.name}; found {len(streams)}")
            stream = streams[0]
        else:
            selected = [stream for stream in streams if stream.index == stream_index]
            if len(selected) != 1:
                raise ValueError(f"audio stream index {stream_index} missing in {path.name}")
            stream = selected[0]

        stream_meta = {
            "stream_index": stream.index,
            "codec": stream.codec_context.name,
            "source_rate_hz": stream.rate,
            "channels": len(stream.layout.channels) if stream.layout else None,
        }

        resampler = av.AudioResampler(format="fltp", layout="mono", rate=_SAMPLE_RATE)
        chunks = []
        first_pts = None
        first_time_base = None
        decoded_frames = 0
        for frame in container.decode(stream):
            if first_pts is None:
                first_pts = frame.pts
                first_time_base = frame.time_base
            decoded_frames += 1
            for output in resampler.resample(frame):
                chunks.append(output.to_ndarray().reshape(-1).astype(np.float32, copy=True))
        for output in resampler.resample(None):
            chunks.append(output.to_ndarray().reshape(-1).astype(np.float32, copy=True))

    if not chunks:
        raise ValueError(f"no audio samples decoded from {path.name}")
    samples = np.concatenate(chunks)
    if not np.isfinite(samples).all():
        raise ValueError(f"nonfinite decoded audio in {path.name}")
    start_seconds = None
    if first_pts is not None and first_time_base is not None:
        start_seconds = float(first_pts * first_time_base)
    return samples, {
        **stream_meta,
        "decoded_frames": decoded_frames,
        "resampled_rate_hz": _SAMPLE_RATE,
        "resampled_samples": int(samples.size),
        "first_pts": first_pts,
        "start_seconds": start_seconds,
        "rms": float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))),
        "peak_abs": float(np.max(np.abs(samples))),
    }


def _prefix(values: Any) -> Any:
    import numpy as np

    return np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(values, dtype=np.float64)))


def peak_normalized_correlation(
    embedded: Any, paired: Any, max_lag_samples: int
) -> dict[str, float | int]:
    """Find maximum Pearson correlation; positive lag means paired audio is delayed."""
    import numpy as np

    x = np.asarray(embedded, dtype=np.float64).reshape(-1)
    y = np.asarray(paired, dtype=np.float64).reshape(-1)
    if x.size < 2 or y.size < 2 or max_lag_samples < 0:
        raise ValueError("signals must have at least two samples and lag must be nonnegative")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("signals must be finite")

    nfft = 1 << (x.size + y.size - 2).bit_length()
    cross = np.fft.irfft(np.conj(np.fft.rfft(x, nfft)) * np.fft.rfft(y, nfft), nfft)
    xsum, ysum = _prefix(x), _prefix(y)
    xsq, ysq = _prefix(np.square(x)), _prefix(np.square(y))
    limit = min(max_lag_samples, max(x.size, y.size) - 1)
    best: tuple[float, int] | None = None
    zero_lag = float("nan")

    for lag in range(-limit, limit + 1):
        x_start = max(0, -lag)
        y_start = max(0, lag)
        count = min(x.size - x_start, y.size - y_start)
        if count < 2:
            continue
        x_end, y_end = x_start + count, y_start + count
        cross_sum = float(cross[lag if lag >= 0 else nfft + lag])
        sx = float(xsum[x_end] - xsum[x_start])
        sy = float(ysum[y_end] - ysum[y_start])
        vx = float(xsq[x_end] - xsq[x_start] - sx * sx / count)
        vy = float(ysq[y_end] - ysq[y_start] - sy * sy / count)
        denominator = math.sqrt(max(0.0, vx) * max(0.0, vy))
        correlation = (cross_sum - sx * sy / count) / denominator if denominator else 0.0
        if lag == 0:
            zero_lag = correlation
        if best is None or abs(correlation) > abs(best[0]):
            best = (correlation, lag)

    if best is None:
        raise ValueError("no valid lag overlap")
    return {
        "zero_lag_correlation": float(zero_lag),
        "best_correlation": float(best[0]),
        "best_lag_samples": int(best[1]),
        "best_lag_seconds": float(best[1] / _SAMPLE_RATE),
        "max_lag_samples": int(limit),
    }


def audit_receipt(receipt: dict[str, Any], roots: list[Path], max_lag_ms: int) -> dict[str, Any]:
    if receipt.get("source") != _SOURCE or receipt.get("split") != _SPLIT:
        raise ValueError("receipt source and split must match pinned PseudoBinaural split1 train")
    if (
        receipt.get("gpu_used") is not False
        or receipt.get("sealed_audit_loaded") is not False
        or receipt.get("cpu_only") is not True
    ):
        raise ValueError("receipt must prove CPU-only train-split preflight")
    ids = receipt.get("ids")
    media = receipt.get("media")
    if not isinstance(ids, list) or len(ids) != 8 or len(set(ids)) != len(ids):
        raise ValueError("expected exactly eight unique prespecified train IDs")
    if not isinstance(media, list) or {row.get("id") for row in media} != set(ids):
        raise ValueError("receipt media rows must match the eight train IDs")
    if any(not re.fullmatch(r"\d{6}", asset_id) for asset_id in ids):
        raise ValueError("invalid asset ID")

    rows = []
    max_lag_samples = round(max_lag_ms * _SAMPLE_RATE / 1000)
    for item in media:
        asset_id = item["id"]
        if item.get("split") != _MEDIA_SPLIT:
            raise ValueError(f"media {asset_id} is not explicitly marked split1 train")
        mp4 = _locate(roots, asset_id, "mp4")
        wav = _locate(roots, asset_id, "wav")
        mp4_hash, wav_hash = _sha256(mp4), _sha256(wav)
        if mp4.stat().st_size != item["mp4_bytes"] or mp4_hash != item["mp4_sha256"]:
            raise ValueError(f"MP4 size/hash mismatch for train asset {asset_id}")
        if wav.stat().st_size != item["wav_bytes"] or wav_hash != item["wav_sha256"]:
            raise ValueError(f"WAV size/hash mismatch for train asset {asset_id}")
        audio_streams = item.get("embedded_audio")
        if not isinstance(audio_streams, list) or len(audio_streams) != 1:
            raise ValueError(f"expected exactly one embedded audio stream for {asset_id}")
        embedded, embedded_meta = _decode_mono(mp4, audio_streams[0]["stream_index"])
        paired, paired_meta = _decode_mono(wav)
        rows.append(
            {
                "id": asset_id,
                "split": _SPLIT,
                "mp4_sha256": mp4_hash,
                "wav_sha256": wav_hash,
                "embedded_audio": embedded_meta,
                "paired_wav": paired_meta,
                "waveform_comparison": peak_normalized_correlation(
                    embedded, paired, max_lag_samples
                ),
            }
        )

    return {
        "schema_version": 1,
        "source": _SOURCE,
        "receipt_sha256": hashlib.sha256(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "split": _SPLIT,
        "scope": {
            "train_assets_only": True,
            "asset_count": len(rows),
            "gpu_used": False,
            "labels_read": False,
            "validation_or_test_media_read": False,
            "model_loaded": False,
            "features_or_training": False,
            "raw_waveforms_saved": False,
        },
        "method": {
            "decoder": "PyAV",
            "resample": "float planar mono at 48 kHz",
            "max_lag_ms": max_lag_ms,
            "correlation": (
                "Pearson coefficient on overlapping samples; select largest absolute value"
            ),
            "positive_lag": "paired WAV is delayed relative to MP4 embedded audio",
        },
        "assets": rows,
        "interpretation_limit": (
            "This compares paired audio waveforms only; it does not measure physical "
            "audio-video sync, "
            "scene independence, task quality, or both-modality necessity."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-lag-ms", type=int, default=500)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.max_lag_ms < 0 or args.max_lag_ms > 1000:
        raise ValueError("max lag must be between 0 and 1000 ms")
    receipt_bytes = args.receipt.read_bytes()
    report = audit_receipt(json.loads(receipt_bytes), args.media_root, args.max_lag_ms)
    report["input_hashes"] = {
        "receipt_file_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "receipt_media_manifest_canonical_sha256": report.pop("receipt_sha256"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
