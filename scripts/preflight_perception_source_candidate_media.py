"""CPU media-integrity smoke for a seeded train-only sample."""

import argparse
import hashlib
import json
from pathlib import Path


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def select_smoke_records(records, media_video_ids, *, limit=8, seed=17):
    """Choose at most one train candidate per video in deterministic hash order."""
    if type(limit) is not int or limit <= 0:
        raise ValueError("limit must be a positive integer")
    per_video = {}
    for record in records:
        if record.get("status") != "candidate":
            continue
        video_id = record.get("group_id")
        if video_id not in media_video_ids:
            continue
        current = per_video.get(video_id)
        rank = _digest(f"{seed}:{record['record_id']}")
        if current is None or rank < current[0]:
            per_video[video_id] = (rank, record)
    ordered = sorted(
        (record for _, record in per_video.values()),
        key=lambda record: _digest(f"{seed}:{record['group_id']}"),
    )
    return ordered[:limit]


def audio_interval_coverage(interval_us, decoded_intervals):
    """Return the union coverage in seconds for decoded audio-frame intervals."""
    start, end = (float(value) / 1e6 for value in interval_us)
    if start < 0 or end <= start:
        raise ValueError("audio interval must be a positive half-open range")
    clipped = sorted(
        (max(start, left), min(end, right))
        for left, right in decoded_intervals
        if right > start and left < end
    )
    coverage = 0.0
    current_start = current_end = None
    for left, right in clipped:
        if right <= left:
            continue
        if current_start is None:
            current_start, current_end = left, right
        elif left <= current_end:
            current_end = max(current_end, right)
        else:
            coverage += current_end - current_start
            current_start, current_end = left, right
    if current_start is not None:
        coverage += current_end - current_start
    return coverage


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_seconds(frame):
    if frame.time is not None:
        return float(frame.time)
    if frame.pts is None or frame.time_base is None:
        return None
    return float(frame.pts * frame.time_base)


def inspect_record(record, clip, media_dir):
    import av

    video_id = record["media_reference"]["video_id"]
    path = Path(media_dir) / f"{video_id}.mp4"
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != clip["bytes"]:
        raise ValueError(f"file size differs from acquisition receipt: {video_id}")
    actual_hash = sha256_file(path)
    if actual_hash != clip["sha256"]:
        raise ValueError(f"file SHA-256 differs from acquisition receipt: {video_id}")
    metadata = clip["metadata"]

    wanted_frames = sorted(
        {
            box["frame_id"]
            for candidate in record["request"]["candidates"]
            for box in candidate["frame_boxes"]
        }
    )
    frame_times = {}
    with av.open(str(path)) as container:
        if not container.streams.video or not container.streams.audio:
            raise ValueError(f"audio/video stream missing: {video_id}")
        video_stream = container.streams.video[0]
        max_frame = max(wanted_frames)
        decoded_frames = {}
        for index, frame in enumerate(container.decode(video_stream)):
            if index in wanted_frames:
                frame_time = _frame_seconds(frame)
                decoded_frames[index] = {
                    "time_s": frame_time,
                    "width": frame.width,
                    "height": frame.height,
                }
                if frame_time is not None:
                    frame_times[index] = frame_time
            if index >= max_frame:
                break
    missing = sorted(set(wanted_frames) - decoded_frames.keys())
    if missing:
        raise ValueError(f"requested bbox frame IDs absent in decoded media: {video_id} {missing}")
    if len(frame_times) != len(decoded_frames):
        raise ValueError(f"decoded candidate frame has no presentation timestamp: {video_id}")

    interval = record["request"]["audio_interval_us"]
    interval_start_s, interval_end_s = (value / 1e6 for value in interval)
    frame_tolerance_s = 2 / metadata["frame_rate"]
    out_of_interval_frames = [
        frame_id
        for frame_id, frame_time in frame_times.items()
        if not interval_start_s - frame_tolerance_s
        <= frame_time
        <= interval_end_s + frame_tolerance_s
    ]
    if out_of_interval_frames:
        raise ValueError(
            f"decoded candidate frame timestamps fall outside event interval: "
            f"{video_id} {out_of_interval_frames}"
        )

    with av.open(str(path)) as container:
        audio_stream = container.streams.audio[0]
        audio_intervals = []
        event_end_s = interval[1] / 1e6
        for frame in container.decode(audio_stream):
            frame_start = _frame_seconds(frame)
            if frame_start is None or not frame.sample_rate:
                raise ValueError(f"audio frame has no timing/rate: {video_id}")
            frame_end = frame_start + frame.samples / frame.sample_rate
            audio_intervals.append((frame_start, frame_end))
            if frame_end >= event_end_s:
                break
    coverage = audio_interval_coverage(interval, audio_intervals)
    target_duration = (interval[1] - interval[0]) / 1e6
    if coverage < target_duration - 0.001:
        raise ValueError(
            f"decoded audio does not cover full annotated interval: {video_id} "
            f"{coverage:.6f}/{target_duration:.6f}s"
        )

    return {
        "record_id": record["record_id"],
        "video_id": video_id,
        "source_split": metadata["split"],
        "media_bytes": clip["bytes"],
        "media_sha256": actual_hash,
        "candidate_count": len(record["request"]["candidates"]),
        "requested_frame_count": len(wanted_frames),
        "decoded_frame_count": len(decoded_frames),
        "decoded_frame_ids": wanted_frames,
        "decoded_frame_times_s": {
            str(frame_id): frame_times[frame_id] for frame_id in sorted(frame_times)
        },
        "decoded_frame_shapes": sorted(
            {(item["height"], item["width"]) for item in decoded_frames.values()}
        ),
        "audio_interval_us": interval,
        "decoded_audio_coverage_s": coverage,
        "audio_interval_duration_s": target_duration,
    }


def preflight(corpus_path, acquisition_manifest_path, media_dir, *, limit=8, seed=17):
    corpus_path = Path(corpus_path)
    acquisition_manifest_path = Path(acquisition_manifest_path)
    manifest = json.loads(acquisition_manifest_path.read_text(encoding="utf-8"))
    clips = {}
    for clip in manifest["clips"]:
        video_id = clip["metadata"]["video_id"]
        if clip["metadata"].get("split") != "train":
            raise ValueError("only official train media is allowed")
        if video_id in clips:
            raise ValueError(f"duplicate acquisition clip ID: {video_id}")
        clips[video_id] = clip

    records = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()]
    selected = select_smoke_records(records, clips, limit=limit, seed=seed)
    if len(selected) < limit:
        raise ValueError(f"only {len(selected)} media groups available for requested smoke size")
    results = [inspect_record(record, clips[record["group_id"]], media_dir) for record in selected]
    return {
        "protocol": "perception_test_source_candidate_cpu_media_smoke_v1",
        "source_split": "official train only",
        "seed": seed,
        "video_group_count": len(results),
        "unique_video_count": len({row["video_id"] for row in results}),
        "source_corpus_sha256": sha256_file(corpus_path),
        "acquisition_manifest_sha256": sha256_file(acquisition_manifest_path),
        "gpu_used": False,
        "model_loaded": False,
        "metrics_computed": False,
        "records": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--acquisition-manifest", required=True)
    parser.add_argument("--media-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite prior preflight: {output}")
    result = preflight(
        args.corpus,
        args.acquisition_manifest,
        args.media_dir,
        limit=args.limit,
        seed=args.seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
