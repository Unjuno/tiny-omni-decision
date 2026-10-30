"""Persistent CPU worker for official DCASE FOA-to-stereo and perspective frames."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

from tiny_omni_decision.dcase2025_media import stereo_from_foa, uniform_clip_frame_indices

GENERATOR_COMMIT = "08e267e5152d37cb7a828f5f52b256f917609132"


class DCASEMediaWorker:
    def __init__(self, media_manifest: Path, official_generator_source: Path):
        commit = subprocess.check_output(
            ["git", "-C", str(official_generator_source), "rev-parse", "HEAD"], text=True
        ).strip()
        if commit != GENERATOR_COMMIT:
            raise ValueError("official DCASE generator source commit mismatch")
        manifest = json.loads(media_manifest.read_text(encoding="utf-8"))
        if manifest.get("status") != "PASS":
            raise ValueError("frozen STARSS23 media manifest is incomplete")
        self.media = {row["clip"]: row for row in manifest["pairs"]}
        sys.path.insert(0, str(official_generator_source.resolve()))
        from utils import E2PFast

        self.E2PFast = E2PFast
        self.audio_parent: str | None = None
        self.foa: np.ndarray | None = None
        self.video_parent: str | None = None
        self.capture: cv2.VideoCapture | None = None
        self.transforms: OrderedDict[tuple[int, int, int], object] = OrderedDict()

    def _load_audio(self, parent: str) -> np.ndarray:
        if parent != self.audio_parent:
            record = self.media[parent]["files"]["audio"]
            values, sample_rate = sf.read(record["path"], dtype="float32", always_2d=True)
            if sample_rate != 24000 or values.shape[1] != 4:
                raise ValueError(f"unexpected FOA source format: {parent}")
            self.audio_parent = parent
            self.foa = values
        assert self.foa is not None
        return self.foa

    def _load_video(self, parent: str) -> cv2.VideoCapture:
        if parent != self.video_parent:
            if self.capture is not None:
                self.capture.release()
            record = self.media[parent]["files"]["video"]
            self.capture = cv2.VideoCapture(record["path"])
            if not self.capture.isOpened():
                raise ValueError(f"could not open source video: {parent}")
            self.video_parent = parent
        assert self.capture is not None
        return self.capture

    def process(self, request: dict[str, object]) -> dict[str, object]:
        started = time.perf_counter()
        queries = request.get("queries")
        output_path = Path(str(request.get("output_npz", "")))
        if not isinstance(queries, list) or not queries or not output_path.parent.is_dir():
            raise ValueError("worker request needs queries and an existing output directory")

        audio_rows = []
        video_rows = []
        valid = []
        clipped_ids = []
        sample_ids = []
        for query in queries:
            if not isinstance(query, dict):
                raise ValueError("worker query must be an object")
            sample_id = str(query["sample_id"])
            parent = str(query["parent_recording"])
            if parent not in self.media:
                raise ValueError(f"parent recording is absent from frozen media: {parent}")
            start_frame = int(query["start_frame"])
            view_degrees = int(query["view_degrees"])
            event_frame = int(query["event_frame"])

            foa = self._load_audio(parent)
            first_sample = int(start_frame * 0.1 * 24000)
            try:
                stereo = stereo_from_foa(
                    foa[first_sample : first_sample + 120000], view_degrees=view_degrees
                )
            except ValueError as error:
                if "clipped stereo sample" not in str(error):
                    raise
                audio_rows.append(np.zeros((2, 120000), dtype=np.float32))
                video_rows.append(np.zeros((8, 360, 640, 3), dtype=np.uint8))
                valid.append(False)
                clipped_ids.append(sample_id)
                sample_ids.append(sample_id)
                continue
            if stereo.shape != (120000, 2):
                raise ValueError(f"short FOA clip: {sample_id}")

            capture = self._load_video(parent)
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_indices = uniform_clip_frame_indices(
                start_frame=start_frame,
                fps=fps,
                event_frame=event_frame,
            )
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            key = (height, width, view_degrees)
            transform = self.transforms.get(key)
            if transform is None:
                horizontal_fov = 100
                vertical_fov = 2 * math.atan(
                    math.tan(math.radians(horizontal_fov / 2)) * 360 / 640
                )
                transform = self.E2PFast(
                    in_hw=(height, width),
                    fov_deg=(horizontal_fov, math.degrees(vertical_fov)),
                    u_deg=view_degrees,
                    v_deg=0,
                    out_hw=(360, 640),
                )
                if len(self.transforms) >= 8:
                    self.transforms.popitem(last=False)
                self.transforms[key] = transform
            else:
                self.transforms.move_to_end(key)

            frames = []
            for frame_index in frame_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                decoded, frame = capture.read()
                if not decoded:
                    raise ValueError(f"source video ended before requested frame: {sample_id}")
                perspective_bgr = transform.e2p(frame)
                frames.append(cv2.cvtColor(perspective_bgr, cv2.COLOR_BGR2RGB))

            audio_rows.append(stereo.T.astype(np.float32, copy=False))
            video_rows.append(np.stack(frames))
            valid.append(True)
            sample_ids.append(sample_id)

        np.savez(
            output_path,
            audio=np.stack(audio_rows),
            video=np.stack(video_rows),
            valid=np.asarray(valid, dtype=np.bool_),
            sample_ids=np.asarray(sample_ids, dtype=np.str_),
        )
        return {
            "status": "PASS",
            "batch_id": request.get("batch_id"),
            "valid_count": int(sum(valid)),
            "clipped_count": len(clipped_ids),
            "clipped_sample_ids": clipped_ids,
            "output_npz": str(output_path.resolve()),
            "wall_seconds": time.perf_counter() - started,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-manifest", type=Path, required=True)
    parser.add_argument("--generator-source", type=Path, required=True)
    args = parser.parse_args()
    worker = DCASEMediaWorker(args.media_manifest, args.generator_source)
    print(
        json.dumps(
            {
                "status": "READY",
                "generator_commit": GENERATOR_COMMIT,
                "media_parent_recordings": len(worker.media),
                "audio_rate_hz": 24000,
                "perspective_frames": 8,
            }
        ),
        flush=True,
    )
    for line in sys.stdin:
        try:
            response = worker.process(json.loads(line))
        except Exception as error:
            response = {"status": "FAIL", "error": f"{type(error).__name__}: {error}"}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
