"""Strict native-input decoding helpers; no targets, model or remote execution."""

import hashlib
import json
import wave
from pathlib import Path

import numpy as np


def sample_indices(count):
    if not isinstance(count, int) or count < 1:
        raise ValueError("positive actual decoded frame count required")
    return sorted({0, (count - 1) // 2, count - 1})


def parse_csi(value):
    required = {"module_number", "time_stamp_ns", "RSS", "complex_csi"}
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("three native CSI modules required")
    arrays, metrics, seen = {}, [], set()
    for m in value:
        if not isinstance(m, dict) or not required <= set(m):
            raise ValueError("native CSI fields missing")
        number = m["module_number"]
        if type(number) is not int or number not in [1, 2, 3] or number in seen:
            raise ValueError("three unique integer CSI module numbers required")
        seen.add(number)
        t = m["time_stamp_ns"]
        if (
            not isinstance(t, list)
            or not 2 <= len(t) <= 4096
            or any(type(v) is not int or not 0 <= v < 2**63 for v in t)
        ):
            raise ValueError("bounded native integer timestamps required, no silent cast")
        ts = np.array(t, np.int64)
        if np.any(np.diff(ts) <= 0):
            raise ValueError("strictly increasing CSI timestamps required")
        rows = m["complex_csi"]
        if (
            not isinstance(rows, list)
            or len(rows) != len(t)
            or any(
                not isinstance(row, list)
                or len(row) != 256
                or any(not isinstance(s, str) for s in row)
                for row in rows
            )
        ):
            raise ValueError("aligned256 native complex-string subcarriers required")
        try:
            z = np.array([[complex(s) for s in row] for row in rows], np.complex128)
            rss = np.asarray(m["RSS"], np.float64)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError("native numeric CSI/RSS required; no eval/imputation") from exc
        if rss.shape != (len(t),) or not np.isfinite(z).all() or not np.isfinite(rss).all():
            raise ValueError("finite aligned CSI/RSS required")
        prefix = f"module_{number}"
        arrays.update(
            {
                f"{prefix}_real": z.real.copy(),
                f"{prefix}_imag": z.imag.copy(),
                f"{prefix}_rssi": rss,
                f"{prefix}_timestamps_ns": ts,
            }
        )
        dt = np.diff(ts).astype(np.float64) / 1e9
        metrics.append(
            dict(
                module_number=number,
                samples=len(t),
                subcarriers=256,
                first_timestamp_ns=int(ts[0]),
                last_timestamp_ns=int(ts[-1]),
                duration_seconds=float((ts[-1] - ts[0]) / 1e9),
                median_interval_seconds=float(np.median(dt)),
                min_interval_seconds=float(dt.min()),
                max_interval_seconds=float(dt.max()),
                excluded_metadata_keys=sorted(set(m) - required),
                native_timestamp_role="alignment audit only; not a learned absolute-time feature",
            )
        )
    return arrays, sorted(metrics, key=lambda m: m["module_number"])


def pcm_samples(raw, width, channels):
    if (
        width not in [1, 2, 3, 4]
        or channels not in [1, 2]
        or not raw
        or len(raw) % (width * channels)
    ):
        raise ValueError("complete native PCM frames/channels required")
    if width == 1:
        a = np.frombuffer(raw, np.uint8).astype(np.float32) - 128
        scale = 128
    elif width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        a = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        a = np.where(a & 0x800000, a - 0x1000000, a).astype(np.float32)
        scale = 2**23
    else:
        a = np.frombuffer(raw, f"<i{width}").astype(np.float32)
        scale = 2 ** (8 * width - 1)
    return (a / scale).astype(np.float32).reshape(-1, channels)


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        channels, width, rate, count = (
            w.getnchannels(),
            w.getsampwidth(),
            w.getframerate(),
            w.getnframes(),
        )
        if w.getcomptype() != "NONE" or not 8000 <= rate <= 192000 or not 1 <= count <= rate * 60:
            raise ValueError("bounded uncompressed native WAV required")
        raw = w.readframes(count)
        if len(raw) != count * width * channels:
            raise ValueError("native WAV payload truncated")
    a = pcm_samples(raw, width, channels)
    return a, dict(
        channels=channels,
        sample_width_bytes=width,
        sample_rate=rate,
        sample_frames=count,
        duration_seconds=count / rate,
        rms=float(np.sqrt(np.mean(a.astype(np.float64) ** 2))),
        absolute_start_clock="NOT_AVAILABLE_IN_WAV",
    )


def read_video(path, check):
    import av

    hashes, times, pts = [], [], []
    dims = None
    with av.open(str(path)) as c:
        streams = c.streams.video
        if len(streams) != 1:
            raise ValueError("one native video stream required")
        stream = streams[0]
        stream.codec_context.thread_count = 1
        info = dict(
            codec=stream.codec_context.name,
            time_base=str(stream.time_base),
            declared_frames=stream.frames,
            average_rate=str(stream.average_rate),
            container_metadata=dict(c.metadata),
            stream_metadata=dict(stream.metadata),
        )
        for frame in c.decode(stream):
            check()
            if len(hashes) >= 1024 or frame.pts is None or frame.time is None:
                raise ValueError("bounded frames with native PTS required")
            if frame.width > 4096 or frame.height > 2160:
                raise ValueError("native video dimensions exceed smoke budget")
            a = frame.to_ndarray(format="rgb24")
            if dims is None:
                dims = a.shape
            if a.shape != dims or a.dtype != np.uint8:
                raise ValueError("native decoded dimensions/dtype changed")
            hashes.append(hashlib.sha256(a.tobytes()).hexdigest())
            times.append(float(frame.time))
            pts.append(frame.pts)
    indices = sample_indices(len(hashes))
    if not np.isfinite(times).all() or (len(times) > 1 and np.any(np.diff(times) <= 0)):
        raise ValueError("finite strictly increasing native video times required")
    selected = []
    with av.open(str(path)) as c:
        c.streams.video[0].codec_context.thread_count = 1
        for i, frame in enumerate(c.decode(video=0)):
            check()
            if i in indices:
                a = frame.to_ndarray(format="rgb24")
                if hashlib.sha256(a.tobytes()).hexdigest() != hashes[i] or frame.pts != pts[i]:
                    raise ValueError("second CPU decoder pass differs at fixed sample")
                selected.append(a)
            if i == indices[-1]:
                break
    if len(selected) != len(indices):
        raise ValueError("all fixed sampled frames required")
    arrays = dict(
        rgb_samples=np.stack(selected),
        sample_indices=np.array(indices, np.int64),
        frame_times_seconds=np.array(times, np.float64),
        frame_pts=np.array(pts, np.int64),
    )
    info.update(
        decoded_frames=len(hashes),
        rgb_shape=list(dims),
        sample_indices=indices,
        first_frame_time_seconds=times[0],
        last_frame_time_seconds=times[-1],
        median_interval_seconds=float(np.median(np.diff(times))) if len(times) > 1 else None,
        rgb_stream_sha256=hashlib.sha256("".join(hashes).encode()).hexdigest(),
        raw_frame_sha256=hashes,
        absolute_clock_alignment="NOT_CERTIFIED_BY_PTS",
    )
    return arrays, info


def load_csi(path):
    return parse_csi(json.loads(Path(path).read_bytes()))
