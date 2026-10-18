"""CPU smoke for actual-PTS sound-event crops through the pinned CLAP audio branch."""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from scripts.audit_perception_av_correspondence import sha256_file
from scripts.audit_perception_av_event_media_pts import decode_timing, event_window_from_pts
from scripts.audit_perception_av_event_task import (
    TRAIN_SOURCE_SHA256,
    load_annotations,
    summarize_split,
)
from scripts.clap_full_clip import duration_weighted_feature, window_spans

PLAN_SHA256 = "9ed4f44f3a0cc85465c5a2955706a957326ef36694311c25b4ee51e3c9f18a25"
WEIGHT_SHA256 = "d190141ffd119b9096d9f3e096b2285f8ce6fa401cdb666d6f53911a2517fee3"
CONFIG_SHA256 = "f7a374a0dc4907b3a30a4e9964286055929c42d932d5abd0619f7b1a20438dd3"
FRONTEND_SHA256 = "9739f58296aa6f9ac18008fd0150fb2649bc554985fbde86d0a4041c882ac753"
MAX_SAMPLES = 480_000
TARGET_RATE = 48_000


def event_sample_bounds(event_start, event_end, audio_start, sample_rate, sample_count):
    """Map an actual-PTS event interval to a half-open decoded waveform slice."""
    if (
        not all(np.isfinite(value) for value in (event_start, event_end, audio_start))
        or event_end <= event_start
        or sample_rate <= 0
        or sample_count < 0
    ):
        raise ValueError("invalid event/audio sample interval")
    start = round((event_start - audio_start) * sample_rate)
    end = round((event_end - audio_start) * sample_rate)
    if not (0 <= start < end <= sample_count):
        raise ValueError("actual-PTS event sample bounds exceed decoded waveform")
    return start, end


def run_smoke(train_source, media_manifest_path, model_root, output_dir):
    import av
    import torch
    import transformers
    from torch.nn import functional as F
    from transformers import ClapAudioConfig, ClapAudioModelWithProjection, ClapFeatureExtractor

    train_source = Path(train_source)
    media_manifest_path = Path(media_manifest_path)
    model_root = Path(model_root)
    audio_root = model_root / "audio-runtime-01"
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("fresh output directory required")
    manifest = json.loads(media_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("split") != "train" or manifest.get("plan_sha256") != PLAN_SHA256:
        raise ValueError("expected pinned TRAIN media manifest")
    clips = {clip["video_id"]: clip for clip in manifest.get("clips", [])}
    video_id, sound_id = "video_11362", 4
    if video_id not in clips:
        raise ValueError("fixed TRAIN smoke video is not acquired")
    clip = clips[video_id]
    media_path = Path(clip["path"])
    if sha256_file(media_path) != clip.get("sha256"):
        raise ValueError("smoke media SHA-256 mismatch")
    for path, expected in (
        (audio_root / "audio.pt", WEIGHT_SHA256),
        (audio_root / "config.json", CONFIG_SHA256),
        (model_root / "model" / "preprocessor_config.json", FRONTEND_SHA256),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"pinned CLAP audio artifact changed: {path.name}")

    data, cuts, annotation_pins = load_annotations(
        train_source / "train_annotations.zip",
        train_source / "cut_frame_mapping_train.json",
        "train",
        {
            "annotations": TRAIN_SOURCE_SHA256["annotations"],
            "cuts": TRAIN_SOURCE_SHA256["cuts"],
        },
    )
    event = next(
        row
        for row in summarize_split(data, cuts, "train")["candidate_records"]
        if row["video_id"] == video_id and row["sound_id"] == sound_id
    )
    timing = decode_timing(media_path)
    event_start, event_end = event_window_from_pts(
        timing["video_frame_times"],
        timing["video_frame_durations"],
        event["frame_interval"],
    )
    audio_start = timing["audio_start_seconds"]
    audio_end = timing["audio_end_seconds"]
    if event_start < audio_start or event_end > audio_end:
        raise ValueError("annotated event is not fully covered by actual audio PTS")

    started = time.perf_counter()
    with av.open(str(media_path)) as container:
        stream = container.streams.audio[0]
        native_rate = stream.codec_context.sample_rate
        if native_rate <= 0:
            raise ValueError("invalid native audio sample rate")
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=TARGET_RATE)
        arrays = []
        resampled_start = None
        for frame in container.decode(audio=0):
            for converted in resampler.resample(frame):
                if resampled_start is None:
                    if converted.pts is None or converted.time_base is None:
                        raise ValueError("resampled audio lacks actual PTS")
                    resampled_start = float(converted.pts * converted.time_base)
                arrays.append(converted.to_ndarray())
        for converted in resampler.resample(None):
            arrays.append(converted.to_ndarray())
    resampled = np.concatenate(arrays, axis=1)
    if resampled_start is None or abs(resampled_start - audio_start) > 1 / TARGET_RATE:
        raise ValueError("resampled audio PTS do not align with decoded audio start")
    event_start_sample, event_end_sample = event_sample_bounds(
        event_start,
        event_end,
        resampled_start,
        TARGET_RATE,
        resampled.shape[1],
    )
    event_wave = resampled[:, event_start_sample:event_end_sample]
    if event_wave.shape[0] != 1 or not np.isfinite(event_wave).all():
        raise ValueError("event waveform must be finite mono audio")
    waveform = event_wave[0]
    if not waveform.size or not np.isfinite(waveform).all():
        raise ValueError("resampled event waveform is empty or non-finite")

    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    model = ClapAudioModelWithProjection(
        ClapAudioConfig.from_pretrained(audio_root, local_files_only=True)
    )
    model.load_state_dict(torch.load(audio_root / "audio.pt", weights_only=True), strict=True)
    model.eval().requires_grad_(False)
    if sum(parameter.numel() for parameter in model.parameters()) != 28_190_872:
        raise ValueError("CLAP audio parameter inventory changed")
    frontend = ClapFeatureExtractor.from_pretrained(
        model_root / "model", local_files_only=True
    )
    if frontend.sampling_rate != TARGET_RATE or frontend.nb_max_samples != MAX_SAMPLES:
        raise ValueError("unexpected CLAP event window contract")

    def encode(window):
        prepared = frontend(window, sampling_rate=TARGET_RATE, return_tensors="pt")
        if prepared["is_longer"].any():
            raise ValueError("CLAP frontend would truncate a fixed event window")
        with torch.inference_mode():
            vector = F.normalize(model(**prepared).audio_embeds, dim=-1)
        return vector.cpu().numpy()[0]

    spans = window_spans(len(waveform), MAX_SAMPLES)
    vectors = []
    for start, stop in spans:
        vector = encode(waveform[start:stop])
        replay = encode(waveform[start:stop])
        if not np.isfinite(vector).all() or not np.array_equal(vector, replay):
            raise ValueError("CLAP event feature is non-finite or nondeterministic")
        vectors.append(vector)
    window_matrix = np.stack(vectors)
    pooled = duration_weighted_feature(window_matrix, [b - a for a, b in spans])

    output_dir.mkdir(parents=True, exist_ok=False)
    feature_path = output_dir / "audio-feature.npy"
    np.save(feature_path, pooled)
    window_path = output_dir / "audio-window-features.npy"
    np.save(window_path, window_matrix)
    result = {
        "status": "PASS_TRAIN_EVENT_AUDIO_ENCODER_SMOKE",
        "scope": "one TRAIN event; audio waveform only; no sound/source label passed to encoder",
        "video_id_for_join_only": video_id,
        "sound_id_for_join_only": sound_id,
        "annotation_source_sha256": annotation_pins,
        "media_sha256": clip["sha256"],
        "frame_interval": event["frame_interval"],
        "actual_video_pts_seconds": [event_start, event_end],
        "actual_audio_pts_seconds": [audio_start, audio_end],
        "native_sample_rate": native_rate,
        "decoded_source_audio_samples": timing["audio_decoded_samples"],
        "resampled_audio_start_seconds": resampled_start,
        "resampled_audio_samples": int(resampled.shape[1]),
        "event_resampled_samples": int(event_wave.shape[1]),
        "event_target_rate": TARGET_RATE,
        "event_target_samples": int(waveform.shape[0]),
        "window_samples": MAX_SAMPLES,
        "window_spans": spans,
        "window_feature_shape": list(window_matrix.shape),
        "pooled_feature_shape": list(pooled.shape),
        "finite": bool(np.isfinite(window_matrix).all() and np.isfinite(pooled).all()),
        "repeat_exact": True,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "model_revision": "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a",
        "audio_weight_sha256": WEIGHT_SHA256,
        "device": "cpu",
        "torch_version": torch.__version__,
        "audio_resampler": "PyAV AudioResampler",
        "transformers_version": transformers.__version__,
        "pyav_version": av.__version__,
        "python_version": platform.python_version(),
        "seconds": time.perf_counter() - started,
        "feature_sha256": sha256_file(feature_path),
        "window_feature_sha256": sha256_file(window_path),
        "training": False,
        "model_selection": False,
        "quality_evaluated": False,
        "gpu_used": False,
        "test_opened": False,
    }
    result["script_sha256"] = sha256_file(Path(__file__))
    output_dir.joinpath("result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--media-manifest", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_smoke(
                args.train_source,
                args.media_manifest,
                args.model_root,
                args.output_dir,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
