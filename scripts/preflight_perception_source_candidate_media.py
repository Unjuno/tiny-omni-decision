"""CPU media-integrity smoke for a seeded train-only sample."""

import argparse
import hashlib
import json
from pathlib import Path


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def select_smoke_records(
    records,
    media_video_ids,
    *,
    limit=8,
    seed=17,
    require_none_visible=0,
    require_cut_frame=0,
):
    """Choose at most one train candidate per video in deterministic hash order."""
    if type(limit) is not int or limit <= 0:
        raise ValueError("limit must be a positive integer")
    if type(require_none_visible) is not int or require_none_visible < 0:
        raise ValueError("required none-visible group count must be nonnegative")
    if type(require_cut_frame) is not int or require_cut_frame < 0:
        raise ValueError("required cut-frame group count must be nonnegative")
    if require_none_visible + require_cut_frame > limit:
        raise ValueError("required strata exceed smoke size")
    per_video = {}
    records_by_video = {}
    for record in records:
        if record.get("status") != "candidate":
            continue
        video_id = record.get("group_id")
        if video_id not in media_video_ids:
            continue
        records_by_video.setdefault(video_id, []).append(record)
        current = per_video.get(video_id)
        rank = _digest(f"{seed}:{record['record_id']}")
        if current is None or rank < current[0]:
            per_video[video_id] = (rank, record)
    selected = []
    selected_video_ids = set()

    def add_required(predicate, count, tag):
        available = []
        for video_id, group_records in records_by_video.items():
            if video_id in selected_video_ids:
                continue
            matching = [record for record in group_records if predicate(record)]
            if matching:
                chosen = min(matching, key=lambda item: _digest(f"{seed}:{item['record_id']}"))
                available.append((video_id, chosen))
        available.sort(key=lambda item: _digest(f"{seed}:{tag}:{item[0]}"))
        if len(available) < count:
            raise ValueError(f"only {len(available)} groups satisfy required stratum {tag}")
        for video_id, record in available[:count]:
            selected.append(record)
            selected_video_ids.add(video_id)

    add_required(
        lambda record: record.get("supervision", {}).get("none_visible") is True,
        require_none_visible,
        "none-visible",
    )
    add_required(
        lambda record: record.get("provenance", {}).get("upstream_cut_frame", -1) >= 0,
        require_cut_frame,
        "cut-frame",
    )
    remaining = sorted(
        (
            record
            for video_id, (_, record) in per_video.items()
            if video_id not in selected_video_ids
        ),
        key=lambda record: _digest(f"{seed}:{record['group_id']}"),
    )
    return selected + remaining[: limit - len(selected)]


def select_records_by_id(records, clips, record_ids):
    """Select an explicit ordered record cohort without collapsing video groups."""
    ordered_ids = list(record_ids)
    if len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError("explicit record ID list contains duplicates")
    by_id = {
        record["record_id"]: record
        for record in records
        if record.get("status") == "candidate" and record.get("group_id") in clips
    }
    missing = [record_id for record_id in ordered_ids if record_id not in by_id]
    if missing:
        raise ValueError(
            f"explicit record IDs are absent from local candidate media: {missing[:5]}"
        )
    return [by_id[record_id] for record_id in ordered_ids]


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


def describe_terminal_frame_repair(
    record, *, missing_frame_ids, declared_frame_count, decoded_frame_count
):
    """Describe a one-frame tail repair without dropping a candidate or target."""
    missing = sorted(set(missing_frame_ids))
    if (
        len(missing) != 1
        or type(decoded_frame_count) is not int
        or type(declared_frame_count) is not int
        or decoded_frame_count < 1
        or declared_frame_count != decoded_frame_count + 1
        or missing[0] != decoded_frame_count
    ):
        raise ValueError("repair requires one missing declared terminal frame")

    candidate_ids = []
    retained_counts = {}
    for candidate in record["request"]["candidates"]:
        candidate_id = candidate["candidate_id"]
        frame_ids = [box["frame_id"] for box in candidate["frame_boxes"]]
        retained = sum(frame_id not in missing for frame_id in frame_ids)
        if retained < 1:
            raise ValueError("terminal frame repair leaves a candidate with no usable frames")
        retained_counts[candidate_id] = retained
        if missing[0] in frame_ids:
            candidate_ids.append(candidate_id)
    if not candidate_ids:
        raise ValueError("terminal frame repair does not affect any candidate")
    return {
        "record_id": record["record_id"],
        "frame_id": missing[0],
        "candidate_ids": candidate_ids,
        "retained_frame_count_by_candidate": retained_counts,
        "declared_frame_count": declared_frame_count,
        "decoded_frame_count": decoded_frame_count,
    }


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
    cut_frame = record.get("provenance", {}).get("upstream_cut_frame")
    if type(cut_frame) is not int or cut_frame != clip.get("cut_frame"):
        raise ValueError(f"record and media receipt cut-frame mapping differ: {video_id}")

    wanted_frames = sorted(
        {
            box["frame_id"]
            for candidate in record["request"]["candidates"]
            for box in candidate["frame_boxes"]
        }
    )
    requested_frame_ids = list(wanted_frames)
    frame_times = {}
    decoded_frame_count = 0
    with av.open(str(path)) as container:
        if not container.streams.video or not container.streams.audio:
            raise ValueError(f"audio/video stream missing: {video_id}")
        video_stream = container.streams.video[0]
        max_frame = max(wanted_frames)
        decoded_frames = {}
        for index, frame in enumerate(container.decode(video_stream)):
            decoded_frame_count = index + 1
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
    frame_repair = None
    if missing:
        frame_repair = describe_terminal_frame_repair(
            record,
            missing_frame_ids=missing,
            declared_frame_count=metadata["num_frames"],
            decoded_frame_count=decoded_frame_count,
        )
        wanted_frames = [frame_id for frame_id in wanted_frames if frame_id not in missing]
        frame_times = {frame_id: frame_times[frame_id] for frame_id in wanted_frames}
        decoded_frames = {frame_id: decoded_frames[frame_id] for frame_id in wanted_frames}
    if len(frame_times) != len(decoded_frames):
        raise ValueError(f"decoded candidate frame has no presentation timestamp: {video_id}")

    interval = record["request"]["audio_interval_us"]
    interval_start_s, interval_end_s = (value / 1e6 for value in interval)
    cut_time_s = (
        metadata["num_frames"] / metadata["frame_rate"]
        if cut_frame == -1
        else cut_frame / metadata["frame_rate"]
    )
    if interval_end_s > cut_time_s:
        raise ValueError(f"audio interval exceeds upstream cut frame: {video_id}")
    if cut_frame >= 0 and any(frame_id >= cut_frame for frame_id in wanted_frames):
        raise ValueError(f"candidate frame ID reaches upstream cut frame: {video_id}")
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
        "upstream_cut_frame": cut_frame,
        "none_visible": record["supervision"]["none_visible"],
        "media_bytes": clip["bytes"],
        "media_sha256": actual_hash,
        "candidate_count": len(record["request"]["candidates"]),
        "requested_frame_count": len(requested_frame_ids),
        "decoded_frame_count": len(decoded_frames),
        "decoded_frame_ids": wanted_frames,
        "requested_frame_ids": requested_frame_ids,
        "media_actual_frame_count": decoded_frame_count if frame_repair else None,
        "terminal_frame_repair": frame_repair,
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


def inspect_video_group(records, clip, media_dir):
    """Decode one acquisition video once for all fixed-cohort records using it."""
    import av

    if not records:
        return []
    video_id = clip["metadata"]["video_id"]
    path = Path(media_dir) / f"{video_id}.mp4"
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != clip["bytes"]:
        raise ValueError(f"file size differs from acquisition receipt: {video_id}")
    actual_hash = sha256_file(path)
    if actual_hash != clip["sha256"]:
        raise ValueError(f"file SHA-256 differs from acquisition receipt: {video_id}")

    requested_by_record = {}
    all_frame_ids = set()
    max_audio_end_s = 0.0
    for record in records:
        cut_frame = record.get("provenance", {}).get("upstream_cut_frame")
        if type(cut_frame) is not int or cut_frame != clip.get("cut_frame"):
            raise ValueError(f"record and media receipt cut-frame mapping differ: {video_id}")
        if record.get("media_reference", {}).get("video_id") != video_id:
            raise ValueError(f"record media reference mismatch: {record['record_id']}")
        frames = sorted(
            {
                box["frame_id"]
                for candidate in record["request"]["candidates"]
                for box in candidate["frame_boxes"]
            }
        )
        if not frames:
            raise ValueError(f"record has no candidate frames: {record['record_id']}")
        requested_by_record[record["record_id"]] = frames
        all_frame_ids.update(frames)
        max_audio_end_s = max(
            max_audio_end_s, record["request"]["audio_interval_us"][1] / 1e6
        )

    decoded_frames = {}
    decoded_frame_count = 0
    max_frame = max(all_frame_ids)
    with av.open(str(path)) as container:
        if not container.streams.video or not container.streams.audio:
            raise ValueError(f"audio/video stream missing: {video_id}")
        for index, frame in enumerate(container.decode(container.streams.video[0])):
            decoded_frame_count = index + 1
            if index in all_frame_ids:
                decoded_frames[index] = {
                    "time_s": _frame_seconds(frame),
                    "width": frame.width,
                    "height": frame.height,
                }
            if index >= max_frame:
                break

    audio_intervals = []
    with av.open(str(path)) as container:
        for frame in container.decode(container.streams.audio[0]):
            frame_start = _frame_seconds(frame)
            if frame_start is None or not frame.sample_rate:
                raise ValueError(f"audio frame has no timing/rate: {video_id}")
            frame_end = frame_start + frame.samples / frame.sample_rate
            audio_intervals.append((frame_start, frame_end))
            if frame_end >= max_audio_end_s:
                break

    metadata = clip["metadata"]
    results = []
    for record in records:
        cut_frame = record["provenance"]["upstream_cut_frame"]
        requested_frame_ids = requested_by_record[record["record_id"]]
        missing = sorted(set(requested_frame_ids) - decoded_frames.keys())
        frame_repair = None
        frame_ids = list(requested_frame_ids)
        if missing:
            frame_repair = describe_terminal_frame_repair(
                record,
                missing_frame_ids=missing,
                declared_frame_count=metadata["num_frames"],
                decoded_frame_count=decoded_frame_count,
            )
            frame_ids = [frame_id for frame_id in frame_ids if frame_id not in missing]
        frame_times = {
            frame_id: decoded_frames[frame_id]["time_s"] for frame_id in frame_ids
        }
        if any(frame_time is None for frame_time in frame_times.values()):
            raise ValueError(f"decoded candidate frame has no presentation timestamp: {video_id}")

        interval = record["request"]["audio_interval_us"]
        cut_time_s = (
            metadata["num_frames"] / metadata["frame_rate"]
            if cut_frame == -1
            else cut_frame / metadata["frame_rate"]
        )
        if interval[1] / 1e6 > cut_time_s:
            raise ValueError(f"audio interval exceeds upstream cut frame: {video_id}")
        if cut_frame >= 0 and any(frame_id >= cut_frame for frame_id in frame_ids):
            raise ValueError(f"candidate frame ID reaches upstream cut frame: {video_id}")
        frame_tolerance_s = 2 / metadata["frame_rate"]
        out_of_interval_frames = [
            frame_id
            for frame_id, frame_time in frame_times.items()
            if not interval[0] / 1e6 - frame_tolerance_s
            <= frame_time
            <= interval[1] / 1e6 + frame_tolerance_s
        ]
        if out_of_interval_frames:
            raise ValueError(
                f"decoded candidate frame timestamps fall outside event interval: "
                f"{video_id} {out_of_interval_frames}"
            )

        coverage = audio_interval_coverage(interval, audio_intervals)
        duration = (interval[1] - interval[0]) / 1e6
        if coverage < duration - 0.001:
            raise ValueError(
                f"decoded audio does not cover full annotated interval: {video_id} "
                f"{coverage:.6f}/{duration:.6f}s"
            )
        frame_shapes = sorted(
            {
                (decoded_frames[frame_id]["height"], decoded_frames[frame_id]["width"])
                for frame_id in frame_ids
            }
        )
        results.append(
            {
                "record_id": record["record_id"],
                "video_id": video_id,
                "source_split": metadata["split"],
                "upstream_cut_frame": cut_frame,
                "none_visible": record["supervision"]["none_visible"],
                "media_bytes": clip["bytes"],
                "media_sha256": actual_hash,
                "candidate_count": len(record["request"]["candidates"]),
                "requested_frame_count": len(requested_frame_ids),
                "decoded_frame_count": len(frame_ids),
                "decoded_frame_ids": frame_ids,
                "requested_frame_ids": requested_frame_ids,
                "media_actual_frame_count": decoded_frame_count if frame_repair else None,
                "terminal_frame_repair": frame_repair,
                "decoded_frame_times_s": {
                    str(frame_id): frame_times[frame_id] for frame_id in sorted(frame_times)
                },
                "decoded_frame_shapes": frame_shapes,
                "audio_interval_us": interval,
                "decoded_audio_coverage_s": coverage,
                "audio_interval_duration_s": duration,
            }
        )
    return results


def preflight(
    corpus_path,
    acquisition_manifest_path,
    media_dir,
    *,
    limit=8,
    seed=17,
    require_none_visible=0,
    require_cut_frame=0,
    record_ids_path=None,
    progress_interval=64,
):
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
    if progress_interval < 1:
        raise ValueError("progress interval must be positive")
    selection = "seeded_one_record_per_video"
    selection_source_jsonl_sha256 = None
    if record_ids_path is None:
        selected = select_smoke_records(
            records,
            clips,
            limit=limit,
            seed=seed,
            require_none_visible=require_none_visible,
            require_cut_frame=require_cut_frame,
        )
        if len(selected) < limit:
            raise ValueError(
                f"only {len(selected)} media groups available for requested smoke size"
            )
    else:
        record_ids_path = Path(record_ids_path)
        requested_ids = [
            json.loads(line)["record_id"]
            for line in record_ids_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        selected = select_records_by_id(records, clips, requested_ids)
        if not selected:
            raise ValueError("explicit record ID list is empty")
        selection = "explicit_ordered_record_ids"
        selection_source_jsonl_sha256 = sha256_file(record_ids_path)

    records_by_video = {}
    for record in selected:
        records_by_video.setdefault(record["group_id"], []).append(record)
    results_by_id = {}
    completed = 0
    for video_id, video_records in records_by_video.items():
        rows = inspect_video_group(video_records, clips[video_id], media_dir)
        results_by_id.update({row["record_id"]: row for row in rows})
        completed += len(rows)
        if completed % progress_interval < len(rows) or completed == len(selected):
            print(f"media preflight {completed}/{len(selected)} records", flush=True)
    results = [results_by_id[record["record_id"]] for record in selected]
    return {
        "status": (
            "PASS_WITH_EXPLICIT_TERMINAL_FRAME_REPAIR"
            if any(row.get("terminal_frame_repair") for row in results)
            else "PASS"
        ),
        "protocol": "perception_test_source_candidate_cpu_media_smoke_v1",
        "source_split": "official train only",
        "selection": selection,
        "seed": seed,
        "selected_record_count": len(results),
        "video_group_count": len({row["video_id"] for row in results}),
        "unique_video_count": len({row["video_id"] for row in results}),
        "selection_source_jsonl_sha256": selection_source_jsonl_sha256,
        "none_visible_event_count": sum(row.get("none_visible", False) for row in results),
        "cut_frame_video_count": sum(row["upstream_cut_frame"] >= 0 for row in results),
        "source_corpus_sha256": sha256_file(corpus_path),
        "acquisition_manifest_sha256": sha256_file(acquisition_manifest_path),
        "gpu_used": False,
        "model_loaded": False,
        "metrics_computed": False,
        "records": results,
        "terminal_frame_repairs": [
            row["terminal_frame_repair"]
            for row in results
            if row.get("terminal_frame_repair") is not None
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--acquisition-manifest", required=True)
    parser.add_argument("--media-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--require-none-visible", type=int, default=0)
    parser.add_argument("--require-cut-frame", type=int, default=0)
    parser.add_argument(
        "--record-ids-jsonl",
        help=(
            "explicit ordered JSONL cohort containing record_id fields; checks every listed record"
        ),
    )
    parser.add_argument("--progress-interval", type=int, default=64)
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
        require_none_visible=args.require_none_visible,
        require_cut_frame=args.require_cut_frame,
        record_ids_path=args.record_ids_jsonl,
        progress_interval=args.progress_interval,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
