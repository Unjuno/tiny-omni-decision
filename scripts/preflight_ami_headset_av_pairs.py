"""Fetch and decode a deterministic train/validation headset-audio AV smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DATASET = "edinburghcstr/ami"
REVISION = "46f28f2503e2ec48f8867a84eef356c70476beab"
MEETINGS = {
    "train": ("TS3009a", "ES2010b", "IS1004b"),
    "validation": ("TS3005b", "ES2008b", "IS1006b"),
}
EXPECTED_METADATA_SHA256 = "3f7a18daaee4864b71279335b263df02919c5912658da97175b68ecdcb6b38e8"
EXPECTED_VIDEO_RECEIPT_SHA256 = "757d9514ae9d2957cd975c2edc3715f24cb1383e04a93f777f26ad5ec41b41b2"
EXPECTED_MEETINGS_XML_SHA256 = "8ab6cdcf03ed863e839418f31c7380392397a2eed892d73b9bc1b438c7ec520a"
MIN_AUDIO_SECONDS = 1.5
MAX_AUDIO_SECONDS = 6.0
VIDEO_END_MARGIN_SECONDS = 2.0
MAX_CLIP_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
ROWS_URL = "https://datasets-server.huggingface.co/rows"


def make_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=2,
        status=3,
        backoff_factor=0.75,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "tiny-omni-decision-ami-av-preflight"})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_one_per_speaker(
    metadata: dict[str, object], video_receipt: dict[str, object]
) -> list[dict[str, object]]:
    rows_by_meeting = metadata["rows_by_meeting"]
    video_rows = video_receipt["results"]
    selected: list[dict[str, object]] = []
    for split, meeting_ids in MEETINGS.items():
        for meeting_id in meeting_ids:
            durations = [
                float(item["duration_s"])
                for item in video_rows
                if item["filename"].startswith(f"{meeting_id}.")
            ]
            if len(durations) != 4:
                raise ValueError(f"expected four verified video durations for {meeting_id}")
            common_end = min(durations)
            # The AMI source quality page reports ES2008b video ending near 2085 s.
            if meeting_id == "ES2008b":
                common_end = min(common_end, 2085.0)
            cutoff = common_end - VIDEO_END_MARGIN_SECONDS

            per_speaker: dict[str, list[dict[str, object]]] = defaultdict(list)
            for row in rows_by_meeting[meeting_id]:
                duration = float(row["duration"])
                if not MIN_AUDIO_SECONDS <= duration <= MAX_AUDIO_SECONDS:
                    continue
                if float(row["end_time"]) > cutoff:
                    continue
                mic = row["microphone_id"]
                if not isinstance(mic, str) or len(mic) != 3 or not mic.startswith("H"):
                    continue
                per_speaker[row["speaker_id"]].append(row)

            if len(per_speaker) != 4:
                raise ValueError(f"expected four eligible headset speakers for {meeting_id}")
            for _speaker_id, choices in sorted(per_speaker.items()):
                row = min(choices, key=lambda x: (float(x["begin_time"]), x["audio_id"]))
                selected.append({**row, "experiment_split": split})

    if len(selected) != 24 or len({r["audio_id"] for r in selected}) != 24:
        raise ValueError("selection must contain 24 unique audio rows")
    return selected


def read_speaker_camera_map(xml_path: Path) -> dict[str, dict[str, dict[str, object]]]:
    result: dict[str, dict[str, dict[str, object]]] = {}
    for meeting in ET.parse(xml_path).getroot().findall(".//meeting"):
        meeting_id = meeting.get("observation")
        if meeting_id not in {m for values in MEETINGS.values() for m in values}:
            continue
        speakers: dict[str, dict[str, object]] = {}
        for speaker in meeting.findall("speaker"):
            speaker_id = speaker.get("global_name")
            if not speaker_id:
                continue
            channel = speaker.get("channel")
            camera = speaker.get("camera")
            if channel not in {"0", "1", "2", "3"} or camera not in {
                "Closeup1", "Closeup2", "Closeup3", "Closeup4"
            }:
                raise ValueError(f"invalid speaker mapping in {meeting_id}: {speaker_id}")
            speakers[speaker_id] = {"channel": int(channel), "camera": camera}
        if len(speakers) != 4:
            raise ValueError(f"expected four mapped speakers for {meeting_id}")
        result[meeting_id] = speakers
    if len(result) != 6:
        raise ValueError("meeting map does not cover the six pinned meetings")
    return result


def current_dataset_revision(session: requests.Session) -> str:
    response = session.get(
        f"https://huggingface.co/api/datasets/{DATASET}", timeout=(15, 45)
    )
    response.raise_for_status()
    current = response.json().get("sha")
    if current != REVISION:
        raise ValueError(f"dataset HEAD changed from pinned revision: {current}")
    return str(current)


def nearest_video_frame(path: Path, target_seconds: float) -> tuple[float, int, int]:
    import av

    with av.open(str(path)) as container:
        stream = next(s for s in container.streams if s.type == "video")
        container.seek(
            int(target_seconds / float(stream.time_base)),
            backward=True,
            any_frame=False,
            stream=stream,
        )
        before = None
        after = None
        dims = None
        for packet in container.demux(stream):
            for frame in packet.decode():
                frame_time = float(frame.time) if frame.time is not None else None
                if frame_time is None:
                    continue
                dims = (frame.width, frame.height)
                if frame_time < target_seconds:
                    before = frame_time
                else:
                    after = frame_time
                    break
            if after is not None:
                break
        if before is None and after is None:
            raise ValueError(f"no video frame near {target_seconds} in {path.name}")
        nearest = min(
            (t for t in (before, after) if t is not None),
            key=lambda t: abs(t - target_seconds),
        )
        if dims is None:
            raise ValueError(f"decoded frame has no dimensions: {path.name}")
        return nearest, dims[0], dims[1]


def fetch_decode_audio(
    session: requests.Session,
    row: dict[str, object],
    output_dir: Path,
    video_paths: dict[tuple[str, str], Path],
    speaker_map: dict[str, dict[str, dict[str, object]]],
) -> dict[str, object]:
    import av

    offset = int(row["source_row_idx"])
    response = session.get(
        ROWS_URL,
        params={
            "dataset": DATASET,
            "config": "ihm",
            "split": "train",
            "offset": offset,
            "length": 1,
        },
        timeout=(15, 45),
    )
    response.raise_for_status()
    payload = response.json()
    if len(payload.get("rows", [])) != 1 or payload["rows"][0].get("row_idx") != offset:
        raise ValueError(f"dataset viewer returned wrong source row at offset {offset}")
    source_row = payload["rows"][0]["row"]
    # The Dataset Viewer response includes a text column. This preflight checks
    # row identity and audio only; it never accesses, stores, or logs that field.
    for key in ("audio_id", "meeting_id", "speaker_id", "microphone_id"):
        if source_row.get(key) != row[key]:
            raise ValueError(f"pinned metadata mismatch for {row['audio_id']}: {key}")
    for key in ("begin_time", "end_time"):
        if abs(float(source_row[key]) - float(row[key])) > 1e-4:
            raise ValueError(f"pinned metadata mismatch for {row['audio_id']}: {key}")

    audio_items = source_row.get("audio")
    if not isinstance(audio_items, list) or len(audio_items) != 1:
        raise ValueError(f"expected one participant audio item for {row['audio_id']}")
    source_url = audio_items[0].get("src")
    if (
        not isinstance(source_url, str)
        or urlparse(source_url).hostname != "datasets-server.huggingface.co"
    ):
        raise ValueError("unexpected audio source host")
    media_response = session.get(source_url, timeout=(15, 90))
    media_response.raise_for_status()
    final_host = urlparse(media_response.url).hostname
    if final_host not in {
        "datasets-server.huggingface.co",
        "huggingface.co",
        "cdn-lfs.huggingface.co",
        "cas-bridge.xethub.hf.co",
    }:
        raise ValueError(f"unexpected media redirect host: {final_host}")
    blob = media_response.content
    if not blob or len(blob) > MAX_CLIP_BYTES:
        raise ValueError(f"audio clip exceeds safety size: {row['audio_id']}")
    content_length = media_response.headers.get("Content-Length")
    if content_length is not None and content_length.isdigit() and len(blob) != int(content_length):
        raise ValueError(f"audio byte count mismatch for {row['audio_id']}")

    with av.open(BytesIO(blob), mode="r") as container:
        if len(container.streams.audio) != 1:
            raise ValueError(f"expected one audio stream for {row['audio_id']}")
        stream = container.streams.audio[0]
        rate = int(stream.rate or 0)
        channels = len(stream.codec_context.layout.channels) if stream.codec_context.layout else 0
        decoded_samples = sum(frame.samples for frame in container.decode(stream))
        decoded_duration = decoded_samples / rate if rate else 0.0
        codec = stream.codec_context.name
        audio_format = container.format.name
    expected_duration = float(row["duration"])
    if rate <= 0 or decoded_samples <= 0 or abs(decoded_duration - expected_duration) > 0.08:
        raise ValueError(f"audio duration/rate mismatch for {row['audio_id']}")

    speaker = speaker_map[str(row["meeting_id"])][str(row["speaker_id"])]
    if speaker["channel"] != int(str(row["microphone_id"])[1:]):
        raise ValueError(f"speaker/channel mapping mismatch for {row['audio_id']}")
    camera = str(speaker["camera"])
    target_seconds = (float(row["begin_time"]) + float(row["end_time"])) / 2
    camera_frames = []
    for camera_index in range(1, 5):
        current_camera = f"Closeup{camera_index}"
        video_path = video_paths[(str(row["meeting_id"]), current_camera)]
        nearest, width, height = nearest_video_frame(video_path, target_seconds)
        camera_frames.append(
            {
                "camera": current_camera,
                "nearest_pts_seconds": nearest,
                "delta_seconds": nearest - target_seconds,
                "width": width,
                "height": height,
            }
        )
    if len(camera_frames) != 4:
        raise ValueError(f"not all four cameras decoded at {row['audio_id']}")
    target_camera_frame = next(x for x in camera_frames if x["camera"] == camera)

    destination = output_dir / f"{row['audio_id']}.audio"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_bytes(blob)
    return {
        "audio_id": row["audio_id"],
        "meeting_id": row["meeting_id"],
        "experiment_split": row["experiment_split"],
        "source_row_idx": offset,
        "speaker_id": row["speaker_id"],
        "microphone_id": row["microphone_id"],
        "camera": camera,
        "begin_time": float(row["begin_time"]),
        "end_time": float(row["end_time"]),
        "duration_metadata": expected_duration,
        "audio_bytes": len(blob),
        "audio_sha256": hashlib.sha256(blob).hexdigest(),
        "audio_format": audio_format,
        "audio_codec": codec,
        "sample_rate": rate,
        "channels": channels,
        "decoded_duration": decoded_duration,
        "decoded_samples": decoded_samples,
        "video_files": [
            video_paths[(str(row["meeting_id"]), f"Closeup{i}")].name
            for i in range(1, 5)
        ],
        "target_video_seconds": target_seconds,
        "target_camera_pts_seconds": target_camera_frame["nearest_pts_seconds"],
        "target_camera_delta_seconds": target_camera_frame["delta_seconds"],
        "four_camera_frames": camera_frames,
        "candidate_frame_decoded": True,
        "audio_path": destination.name,
    }


def run(
    metadata_path: Path,
    video_dir: Path,
    xml_path: Path,
    output_dir: Path,
    plan_path: Path | None = None,
) -> dict[str, object]:
    if sha256_file(metadata_path) != EXPECTED_METADATA_SHA256:
        raise ValueError("selected AMI metadata hash changed")
    if sha256_file(xml_path) != EXPECTED_MEETINGS_XML_SHA256:
        raise ValueError("AMI meeting mapping hash changed")
    video_receipt_path = video_dir / "decode-integrity-receipt.json"
    if sha256_file(video_receipt_path) != EXPECTED_VIDEO_RECEIPT_SHA256:
        raise ValueError("video decode receipt hash changed")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    video_receipt = json.loads(video_receipt_path.read_text(encoding="utf-8"))
    manifest_path = video_dir / "media-manifest.json"
    if sha256_file(manifest_path) != video_receipt["manifest_sha256"]:
        raise ValueError("video manifest hash does not match decode receipt")
    video_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    video_paths: dict[tuple[str, str], Path] = {}
    for item in video_manifest["files"]:
        path = video_dir / item["filename"]
        if (
            path.stat().st_size != item["content_length_bytes"]
            or sha256_file(path) != item["sha256"]
        ):
            raise ValueError(f"video file integrity mismatch: {path.name}")
        video_paths[(item["meeting_id"], f"Closeup{item['camera']}")] = path
    if len(video_paths) != 24:
        raise ValueError("expected 24 intact camera streams")

    speaker_map = read_speaker_camera_map(xml_path)
    selected = select_one_per_speaker(metadata, video_receipt)
    for row in selected:
        mapped = speaker_map[str(row["meeting_id"])].get(str(row["speaker_id"]))
        if mapped is None or mapped["channel"] != int(str(row["microphone_id"])[1:]):
            raise ValueError(f"no pinned speaker/channel map for {row['audio_id']}")
    if plan_path is not None:
        if not plan_path.is_file():
            raise FileNotFoundError(plan_path)
        frozen_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if frozen_plan.get("rows") != selected:
            raise ValueError("selected rows differ from frozen plan")
        expected_plan_hashes = {
            "metadata_sha256": EXPECTED_METADATA_SHA256,
            "video_manifest_sha256": sha256_file(video_dir / "media-manifest.json"),
            "video_decode_receipt_sha256": EXPECTED_VIDEO_RECEIPT_SHA256,
            "meetings_xml_sha256": EXPECTED_MEETINGS_XML_SHA256,
            "revision": REVISION,
        }
        for key, expected in expected_plan_hashes.items():
            if frozen_plan.get(key) != expected:
                raise ValueError(f"selection plan {key} mismatch")

    started = time.perf_counter()
    session = make_session()
    session.headers.update({"User-Agent": "tiny-omni-decision-ami-av-preflight"})
    hub = session.get(f"https://huggingface.co/api/datasets/{DATASET}", timeout=(15, 45))
    hub.raise_for_status()
    current_revision = hub.json().get("sha")
    if current_revision != REVISION:
        raise ValueError(f"dataset HEAD changed from pinned revision: {current_revision}")
    output_dir.mkdir(parents=True, exist_ok=False)
    audio_receipts = []
    total_bytes = 0
    for index, row in enumerate(selected, start=1):
        receipt = fetch_decode_audio(session, row, output_dir, video_paths, speaker_map)
        total_bytes += int(receipt["audio_bytes"])
        if total_bytes > MAX_TOTAL_BYTES:
            raise ValueError("audio acquisition exceeded total safety size")
        audio_receipts.append(receipt)
        print(f"{index:02d}/24 {row['meeting_id']} {row['speaker_id']} decoded", flush=True)

    result = {
        "schema_version": 1,
        "status": "PASS_24_ROW_AUDIO_VIDEO_PREFLIGHT",
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "viewer_current_revision_sha": current_revision,
        "config": "ihm",
        "source_split": "train",
        "experiment_split_by_meeting": MEETINGS,
        "selected_rows": 24,
        "audio_bytes": total_bytes,
        "audio_rows": audio_receipts,
        "text_field_in_api_response": True,
        "text_field_accessed_or_persisted": False,
        "transcripts_used_as_model_input": False,
        "model_inference_or_training": False,
        "sealed_or_unseen_data_accessed": False,
        "gpu_used": False,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "metadata_sha256": EXPECTED_METADATA_SHA256,
        "video_manifest_sha256": video_receipt["manifest_sha256"],
        "video_decode_receipt_sha256": EXPECTED_VIDEO_RECEIPT_SHA256,
        "meetings_xml_sha256": EXPECTED_MEETINGS_XML_SHA256,
        "selection_plan_sha256": sha256_file(plan_path) if plan_path else None,
    }
    receipt_path = output_dir / "av-pair-receipt.json"
    receipt_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    checksum = sha256_file(receipt_path)
    (output_dir / "SHA256SUMS.txt").write_text(
        f"{checksum}  {receipt_path.name}\n", encoding="ascii"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "rows": len(audio_receipts),
                "bytes": total_bytes,
                "receipt_sha256": checksum,
            },
            indent=2,
        )
    )
    return result


def write_selection_plan(
    metadata_path: Path, video_dir: Path, xml_path: Path, output_dir: Path
) -> dict[str, object]:
    if sha256_file(metadata_path) != EXPECTED_METADATA_SHA256:
        raise ValueError("selected AMI metadata hash changed")
    if sha256_file(xml_path) != EXPECTED_MEETINGS_XML_SHA256:
        raise ValueError("AMI meeting mapping hash changed")
    receipt_path = video_dir / "decode-integrity-receipt.json"
    if sha256_file(receipt_path) != EXPECTED_VIDEO_RECEIPT_SHA256:
        raise ValueError("video decode receipt hash changed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest_path = video_dir / "media-manifest.json"
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != receipt["manifest_sha256"]:
        raise ValueError("video manifest hash mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected = select_one_per_speaker(metadata, receipt)
    speaker_map = read_speaker_camera_map(xml_path)
    for row in selected:
        mapped = speaker_map[str(row["meeting_id"])].get(str(row["speaker_id"]))
        if mapped is None or mapped["channel"] != int(str(row["microphone_id"])[1:]):
            raise ValueError(f"speaker/channel mismatch for {row['audio_id']}")
    session = make_session()
    current_revision = current_dataset_revision(session)
    output_dir.mkdir(parents=True, exist_ok=False)
    plan = {
        "schema_version": 1,
        "dataset": DATASET,
        "revision": REVISION,
        "current_hub_head": current_revision,
        "config": "ihm",
        "source_split": "train",
        "experiment_split_by_meeting": MEETINGS,
        "selection_rule": (
            "one earliest row per speaker; duration 1.5–6.0 seconds; ending at least "
            "2 seconds before shortest camera duration; ES2008b end <= 2083 seconds"
        ),
        "metadata_sha256": EXPECTED_METADATA_SHA256,
        "video_manifest_sha256": manifest_hash,
        "video_decode_receipt_sha256": EXPECTED_VIDEO_RECEIPT_SHA256,
        "meetings_xml_sha256": EXPECTED_MEETINGS_XML_SHA256,
        "transcript_text_used": False,
        "rows": selected,
    }
    plan_path = output_dir / "selection-plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    checksum = sha256_file(plan_path)
    (output_dir / "SHA256SUMS.txt").write_text(
        f"{checksum}  {plan_path.name}\n", encoding="ascii"
    )
    print(json.dumps({"rows": len(selected), "selection_plan_sha256": checksum}, indent=2))
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--meetings-xml", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--plan-file", type=Path)
    args = parser.parse_args()
    if args.plan_only:
        write_selection_plan(args.metadata, args.video_dir, args.meetings_xml, args.output_dir)
    elif args.plan_file is None:
        parser.error("--plan-file is required unless --plan-only is set")
    else:
        run(
            args.metadata,
            args.video_dir,
            args.meetings_xml,
            args.output_dir,
            args.plan_file,
        )


if __name__ == "__main__":
    main()
