"""Pinned frozen audio-only CLAP features with deterministic full-clip coverage."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from scripts.clap_full_clip import duration_weighted_feature, window_spans
from scripts.clotho_acquisition_gate import complete_acquisition

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
MODEL_ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r1/clap-audit-01")
WEIGHT_SHA = "d190141ffd119b9096d9f3e096b2285f8ce6fa401cdb666d6f53911a2517fee3"
CONFIG_SHA = "f7a374a0dc4907b3a30a4e9964286055929c42d932d5abd0619f7b1a20438dd3"
PROCESSOR_SHA = "9739f58296aa6f9ac18008fd0150fb2649bc554985fbde86d0a4041c882ac753"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import av
    import torch
    import torchaudio
    import transformers
    from torch.nn import functional as F
    from transformers import ClapAudioConfig, ClapAudioModelWithProjection, ClapFeatureExtractor

    media = ROOT / "clotho-aqa-development-media-02"
    records = complete_acquisition(media, ROOT / "clotho-aqa-development-split-01")
    if args.scope == "smoke":
        probe = json.loads((ROOT / "clotho-aqa-training-probe-01/result.json").read_text())
        if probe["status"] != "PASS" or len(probe["files"]) != 16:
            raise ValueError("fixed training probe missing")
        ids = {r["source_id"] for r in probe["files"]}
        records = [r for r in records if r["source_id"] in ids]
        if len(records) != 16 or any(r["split"] != "train" for r in records):
            raise ValueError("training smoke identities differ")
    audio_root = MODEL_ROOT / "audio-runtime-01"
    for path, expected in (
        (audio_root / "audio.pt", WEIGHT_SHA),
        (audio_root / "config.json", CONFIG_SHA),
        (MODEL_ROOT / "model/preprocessor_config.json", PROCESSOR_SHA),
    ):
        if digest(path) != expected:
            raise ValueError("pinned encoder/processor differs")
    torch.set_num_threads(4)
    torch.manual_seed(17)
    np.random.seed(17)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 8 * 1024**3:
            raise ValueError("local CUDA unavailable or insufficient free memory")
        torch.cuda.reset_peak_memory_stats()
    model = ClapAudioModelWithProjection(
        ClapAudioConfig.from_pretrained(audio_root, local_files_only=True)
    )
    model.load_state_dict(torch.load(audio_root / "audio.pt", weights_only=True), strict=True)
    model.eval().requires_grad_(False).to(args.device)
    if sum(p.numel() for p in model.parameters()) != 28190872:
        raise ValueError("audio inventory differs")
    frontend = ClapFeatureExtractor.from_pretrained(MODEL_ROOT / "model", local_files_only=True)
    if frontend.sampling_rate != 48000 or frontend.nb_max_samples != 480000:
        raise ValueError("unexpected native window/sample rate")
    output = args.output
    output.mkdir(exist_ok=False)
    report = dict(
        status="RUNNING",
        scope=args.scope,
        device=args.device,
        rows=[],
        weight_sha256=WEIGHT_SHA,
        config_sha256=CONFIG_SHA,
        processor_sha256=PROCESSOR_SHA,
        media_report_sha256=digest(media / "result.json"),
        script_sha256=digest(Path(__file__)),
        helper_sha256=digest(Path(__file__).with_name("clap_full_clip.py")),
        parameters=28190872,
        dtype="float32",
        torch=torch.__version__,
        transformers=transformers.__version__,
        training=False,
        seed=17,
        deterministic_algorithms=True,
        native_padding=frontend.padding,
        window_samples=480000,
        sample_rate=48000,
        tf32=False,
        pooling="true-duration-weighted mean of normalized window features, then L2",
        limitation="pooled vector discards order; ordered window tokens also retained",
    )
    started = time.perf_counter()
    pooled, all_windows = [], []

    def save():
        report["seconds"] = time.perf_counter() - started
        temporary = output / "progress.tmp"
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(output / "progress.json")

    def encode(wave):
        prepared = frontend(wave, sampling_rate=48000, return_tensors="pt")
        if prepared["is_longer"].any():
            raise ValueError("frontend would truncate a coverage window")
        with torch.inference_mode():
            result = F.normalize(model(**prepared.to(args.device)).audio_embeds, dim=-1)
        return result.cpu().numpy()[0]

    save()
    try:
        for row in records:
            with av.open(str(media / (row["source_id"] + ".wav"))) as container:
                stream = container.streams.audio[0]
                rate = stream.codec_context.sample_rate
                converter = av.AudioResampler(format="fltp", layout="mono", rate=rate)
                arrays = []
                for frame in container.decode(stream):
                    arrays.extend(f.to_ndarray() for f in converter.resample(frame))
                arrays.extend(f.to_ndarray() for f in converter.resample(None))
            native = np.concatenate(arrays, axis=1)
            wave = torchaudio.functional.resample(torch.from_numpy(native), rate, 48000).numpy()[0]
            if not wave.size or not np.isfinite(wave).all():
                raise ValueError("invalid whole-clip input")
            spans = window_spans(len(wave), 480000)
            vectors = []
            for start, stop in spans:
                vector = encode(wave[start:stop])
                replay = encode(wave[start:stop])
                if not np.isfinite(vector).all() or not np.array_equal(vector, replay):
                    raise ValueError("nonfinite or nonrepeatable window feature")
                vectors.append(vector)
            matrix = np.stack(vectors)
            feature = duration_weighted_feature(matrix, [b - a for a, b in spans])
            offset = sum(len(v) for v in all_windows)
            all_windows.append(matrix)
            pooled.append(feature)
            report["rows"].append(
                dict(
                    id=row["source_id"],
                    split=row["split"],
                    samples=len(wave),
                    native_samples=native.shape[1],
                    native_rate=rate,
                    covered_samples=sum(b - a for a, b in spans),
                    spans=spans,
                    window_offset=offset,
                    windows=len(spans),
                )
            )
            save()
            if len(pooled) % 16 == 0:
                print(f"verified full-clip features {len(pooled)}/{len(records)}", flush=True)
        np.save(output / "features.npy", np.stack(pooled))
        np.save(output / "window_features.npy", np.concatenate(all_windows))
        report.update(
            status="PASS",
            feature_sha256=digest(output / "features.npy"),
            window_feature_sha256=digest(output / "window_features.npy"),
            repeat_bitwise_equal=True,
        )
        if args.device == "cuda":
            report.update(
                gpu=torch.cuda.get_device_name(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
    except Exception as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        save()
        (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "rows"}))


if __name__ == "__main__":
    main()
