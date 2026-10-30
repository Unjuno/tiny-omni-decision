"""Bounded full-recording waveform overlap screen; no learning or label tuning."""

import time
import wave
from pathlib import Path

import numpy as np

from scripts.audio_overlap_probe import peak_normalized_match
from scripts.extract_frozen_videoprism_features import sha256_file
from scripts.extract_starss23_corpus_features import read_manifest
from scripts.fetch_starss23_frozen_media import write_json
from scripts.fsd_acquisition_gate import complete_acquisition

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")


def fsd_waveform(path):
    import av
    import torch
    import torchaudio

    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        rate = stream.codec_context.sample_rate
        if stream.codec_context.channels != 1:
            raise ValueError("expected mono FSD input")
        converter = av.AudioResampler(format="fltp", layout="mono", rate=rate)
        arrays = []
        for frame in container.decode(stream):
            arrays.extend(f.to_ndarray() for f in converter.resample(frame))
        arrays.extend(f.to_ndarray() for f in converter.resample(None))
    waveform = torch.from_numpy(np.concatenate(arrays, axis=1))
    return torchaudio.functional.resample(waveform, rate, 2000).numpy().ravel()


def main():
    media = ROOT / "fsd50k-training-128-02"
    output = ROOT / "fsd-starss-waveform-screen-128-01"
    report = complete_acquisition(media)
    ids = [r["id"] for r in report["files"]]
    import torch
    import torchaudio

    windows = read_manifest(ROOT / "starss23-all-windows-v1")
    recordings = {}
    for row in windows:
        previous = recordings.setdefault(row["recording"], row["media"]["audio"])
        if previous != row["media"]["audio"]:
            raise ValueError("inconsistent reference identity")
    output.mkdir(exist_ok=False)
    result = {"status": "RUNNING", "rate": 2000, "review_threshold": 0.99,
              "records": [], "training": False, "device": "cpu",
              "media_report_sha256": sha256_file(media / "result.json"),
              "runner_sha256": sha256_file(Path(__file__)),
              "limitations": "Whole-template, gain/offset/polarity invariance only; "
                             "not proof against mixing, edits or shorter reused excerpts."}
    started = time.perf_counter()
    torch.set_num_threads(4)
    try:
        templates = []
        for row in report["files"]:
            path = media / f"{row['id']}.wav"
            if sha256_file(path) != row["sha256"]:
                raise ValueError("FSD media hash mismatch")
            data = fsd_waveform(path)
            if not np.isfinite(data).all() or data.size < 2:
                raise ValueError("invalid decoded FSD signal")
            templates.append(data)
        result["exact_resampled_duplicates"] = []
        import hashlib

        hashes = {}
        for clip_id, data in zip(ids, templates, strict=True):
            digest = hashlib.sha256(data.astype("<f4").tobytes()).hexdigest()
            if digest in hashes:
                result["exact_resampled_duplicates"].append([hashes[digest], clip_id])
            hashes[digest] = clip_id
        for name, metadata in recordings.items():
            path = Path(metadata["path"])
            if sha256_file(path) != metadata["sha256"]:
                raise ValueError("STARSS media hash mismatch")
            with wave.open(str(path)) as audio:
                fmt = (audio.getframerate(), audio.getnchannels(), audio.getsampwidth())
                if fmt != (24000, 4, 2):
                    raise ValueError("unexpected reference PCM")
                samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
                signal = samples.reshape(-1, 4)[:, 0].astype(np.float32) / 32768
            signal = torchaudio.functional.resample(torch.from_numpy(signal), 24000, 2000).numpy()
            scores = []
            for clip_id, template in zip(ids, templates, strict=True):
                if len(template) > len(signal):
                    raise ValueError("template exceeds reference; incomplete screen")
                scores.append({"clip_id": clip_id, **peak_normalized_match(template, signal)})
            result["records"].append({"recording": name, "sha256": metadata["sha256"],
                                      "scores": scores})
            write_json(output / "progress.json", result)
        values = [s["absolute_correlation"] for r in result["records"] for s in r["scores"]]
        result["pairs"] = len(values)
        result["flagged_pairs"] = sum(v is None or v >= 0.99 for v in values)
        result["maximum_correlation"] = max(v for v in values if v is not None)
        result["status"] = "COMPLETE"
    except Exception as error:
        result["status"], result["error"] = "FAIL", repr(error)
        raise
    finally:
        result["seconds"] = time.perf_counter() - started
        write_json(output / "result.json", result)


if __name__ == "__main__":
    main()
