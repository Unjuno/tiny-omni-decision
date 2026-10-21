"""Create a development-only AMI candidate-camera visibility review."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

DEVELOPMENT_MEETINGS = ("ES2003a", "ES2011a", "IS1008a", "TS3004a", "TS3006a")
AGENTS = ("A", "B", "C", "D")
CAMERAS = {"Closeup1", "Closeup2", "Closeup3", "Closeup4"}
FRAME_FRACTIONS = (0.15, 0.5, 0.85)
ROWS_PER_PAGE = 10
LABEL_WIDTH = 184
CELL_WIDTH = 184
CELL_HEIGHT = 154


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_at(path: Path, timestamp_s: float) -> tuple[Any, float]:
    import av

    container = av.open(str(path))
    try:
        stream = next(item for item in container.streams if item.type == "video")
        container.seek(
            max(0, int(timestamp_s / float(stream.time_base))),
            stream=stream,
            backward=True,
            any_frame=False,
        )
        selected = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            actual_s = float(frame.pts * stream.time_base)
            selected = frame
            if actual_s >= timestamp_s:
                break
        if selected is None:
            raise ValueError(f"no frame found near {timestamp_s:.3f}s in {path.name}")
        return selected.to_image().convert("RGB"), float(selected.pts * stream.time_base)
    finally:
        container.close()


def merge_preflights(paths: list[Path]) -> dict[str, Any]:
    meetings: dict[str, Any] = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = data.get("meetings")
        if entries is None and isinstance(data.get("meeting"), str):
            entry = dict(data)
            entry["partition"] = entry.pop("split", None)
            entries = {entry["meeting"]: entry}
        if not isinstance(entries, dict):
            raise ValueError(f"unsupported preflight manifest shape: {path}")
        for meeting, metadata in entries.items():
            if meeting in meetings:
                raise ValueError(f"duplicate meeting across preflight files: {meeting}")
            meetings[meeting] = metadata
    return {"meetings": meetings}


def select_development_samples(preflight: dict[str, Any]) -> list[dict[str, Any]]:
    meetings = preflight.get("meetings")
    if not isinstance(meetings, dict):
        raise ValueError("preflight meetings must be an object")

    found_development: set[str] = set()
    for meeting, metadata in meetings.items():
        partition = metadata.get("partition")
        if partition not in {"training", "development"}:
            raise ValueError(f"non-development partition in preflight: {meeting}={partition}")
        if partition == "development":
            if meeting not in DEVELOPMENT_MEETINGS:
                raise ValueError(
                    "development meeting not in the pinned development allowlist: "
                    f"{meeting}"
                )
            found_development.add(meeting)
    missing = set(DEVELOPMENT_MEETINGS) - found_development
    if missing:
        raise ValueError(f"missing pinned development meetings: {sorted(missing)}")

    samples: list[dict[str, Any]] = []
    seen_windows: set[tuple[str, str, float, float]] = set()
    for meeting in DEVELOPMENT_MEETINGS:
        metadata = meetings[meeting]
        camera_map = metadata.get("camera_map", {})
        if set(camera_map) != set(AGENTS) or set(camera_map.values()) != CAMERAS:
            raise ValueError(f"invalid camera map for {meeting}")
        windows_by_agent = metadata.get("candidate_windows", {})
        if not isinstance(windows_by_agent, dict) or not set(windows_by_agent) <= set(AGENTS):
            raise ValueError(f"invalid candidate-window agents for {meeting}")
        for agent in AGENTS:
            for index, window in enumerate(windows_by_agent.get(agent, [])):
                start = float(window["start_s"])
                end = float(window["end_s"])
                if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                    raise ValueError(f"invalid candidate window: {meeting}/{agent}/{index}")
                unique_key = (meeting, agent, start, end)
                if unique_key in seen_windows:
                    raise ValueError(f"duplicate candidate window: {meeting}/{agent}/{start}/{end}")
                seen_windows.add(unique_key)
                samples.append(
                    {
                        "sample_id": f"{meeting}:{agent}:{index:04d}:{start:.3f}:{end:.3f}",
                        "meeting": meeting,
                        "parent_set": metadata["parent_set"],
                        "partition": "development",
                        "agent": agent,
                        "target_camera": camera_map[agent],
                        "start_s": start,
                        "end_s": end,
                    }
                )
    return samples


def _render_page(samples: list[dict[str, Any]], output_path: Path, page_no: int) -> None:
    from PIL import Image, ImageDraw, ImageFont

    frame_width = CELL_WIDTH - 8
    frame_height = CELL_HEIGHT - 24
    page = Image.new(
        "RGB",
        (LABEL_WIDTH + len(FRAME_FRACTIONS) * CELL_WIDTH, 32 + len(samples) * CELL_HEIGHT),
        "white",
    )
    draw = ImageDraw.Draw(page)
    font = ImageFont.load_default()
    draw.text(
        (6, 8),
        f"AMI development candidate-camera review / page {page_no}",
        fill="#123",
        font=font,
    )
    for row, sample in enumerate(samples):
        y = 32 + row * CELL_HEIGHT
        draw.text(
            (4, y + 8),
            f"{sample['sample_id']}\n{sample['start_s']:.2f}-{sample['end_s']:.2f}s\n{sample['target_camera']}",
            fill="#123",
            font=font,
        )
        for column, frame_info in enumerate(sample["frame_samples"]):
            x = LABEL_WIDTH + column * CELL_WIDTH
            image = Image.open(output_path.parent / frame_info["image_file"]).convert("RGB")
            image.thumbnail((frame_width, frame_height))
            page.paste(image, (x + 4, y + 20))
            draw.text(
                (x + 4, y + 4),
                f"t={frame_info['actual_pts_s']:.2f}s",
                fill="#345",
                font=font,
            )
    page.save(output_path)


def build_review(
    preflight_paths: list[Path],
    identity_audit_path: Path,
    media_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    preflight = merge_preflights(preflight_paths)
    samples = select_development_samples(preflight)
    identity = json.loads(identity_audit_path.read_text(encoding="utf-8"))
    if identity.get("cross_split_exact_duplicate_group_count") != 0:
        raise ValueError("identity audit reports exact cross-split duplicates")
    media_records = {
        (record["meeting"], record["file"]): record
        for record in identity.get("files", [])
        if record.get("kind") == "video"
    }
    preflight_video_records = {
        (meeting, video["file"]): video
        for meeting, metadata in preflight["meetings"].items()
        for video in metadata.get("videos", [])
    }
    inputs = {
        "preflights": {str(path): sha256_file(path) for path in preflight_paths},
        "identity_audit": sha256_file(identity_audit_path),
    }
    verified_hashes: dict[Path, str] = {}
    render_rows: list[dict[str, Any]] = []
    image_dir = output_dir / "frames"
    image_dir.mkdir(parents=True)

    for sample in samples:
        meeting = sample["meeting"]
        filename = f"{meeting}.{sample['target_camera']}.avi"
        identity_record = media_records.get((meeting, filename))
        preflight_record = preflight_video_records.get((meeting, filename))
        if not identity_record or identity_record.get("split") != "validation":
            raise ValueError(f"non-development or missing video receipt for {sample['sample_id']}")
        if not preflight_record:
            raise ValueError(f"missing pinned preflight video for {sample['sample_id']}")
        if identity_record["sha256"] != preflight_record["sha256"]:
            raise ValueError(f"identity/preflight SHA-256 disagreement for {sample['sample_id']}")
        path = media_root / meeting / "video" / filename
        if path not in verified_hashes:
            if path.stat().st_size != identity_record["bytes"]:
                raise ValueError(f"video size mismatch for {sample['sample_id']}")
            verified_hashes[path] = sha256_file(path)
        if verified_hashes[path] != identity_record["sha256"]:
            raise ValueError(f"video hash mismatch for {sample['sample_id']}")

        frame_records = []
        for frame_index, fraction in enumerate(FRAME_FRACTIONS):
            requested = sample["start_s"] + (sample["end_s"] - sample["start_s"]) * fraction
            frame, actual = _frame_at(path, requested)
            frame_name = f"{sample['sample_id'].replace(':', '_')}_{frame_index:02d}.jpg"
            frame.save(image_dir / frame_name, quality=90)
            frame_records.append(
                {
                    "requested_s": round(requested, 6),
                    "actual_pts_s": round(actual, 6),
                    "delta_s": round(actual - requested, 6),
                    "image_file": str(Path("frames") / frame_name),
                }
            )
        render_rows.append(
            {
                **sample,
                "video_sha256": identity_record["sha256"],
                "frame_samples": frame_records,
            }
        )

    pages = []
    for start in range(0, len(render_rows), ROWS_PER_PAGE):
        page_no = start // ROWS_PER_PAGE + 1
        path = output_dir / f"development-visual-review-{page_no:02d}.jpg"
        group = render_rows[start : start + ROWS_PER_PAGE]
        _render_page(group, path, page_no)
        pages.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})

    result = {
        "schema_version": 1,
        "status": "NOT_ADJUDICATED",
        "purpose": (
            "development-only candidate-camera visibility review; "
            "no model inference or selection"
        ),
        "sample_count": len(render_rows),
        "frame_count_per_sample": len(FRAME_FRACTIONS),
        "source_meetings": list(DEVELOPMENT_MEETINGS),
        "inputs": inputs,
        "verified_video_file_count": len(verified_hashes),
        "pages": pages,
        "samples": render_rows,
        "limitations": [
            "SB development media only; the SC/test partition is not read or included.",
            "Frames and camera metadata do not by themselves establish speaker identity "
            "or precise lip-sync.",
            "This is exploratory review data, not a frozen evaluation battery.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", required=True, action="append", type=Path)
    parser.add_argument("--identity-audit", required=True, type=Path)
    parser.add_argument("--media-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = build_review(
        args.preflight,
        args.identity_audit,
        args.media_root,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "sample_count": result["sample_count"],
                "pages": result["pages"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
