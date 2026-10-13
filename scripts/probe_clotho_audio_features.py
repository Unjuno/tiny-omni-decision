"""Frozen whole-clip features and cross-source similarity triage, no fitting."""

import argparse
import contextlib
import hashlib
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import av
import numpy as np
import torch

from scripts.extract_frozen_videoprism_features import sha256_file
from scripts.extract_starss23_features import AUDIO_SHA, AUDIO_SOURCE


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media", type=Path, default=root / "clotho-aqa-training-probe-01")
    parser.add_argument("--output", type=Path, default=root / "clotho-aqa-frozen-feature-probe-01")
    args = parser.parse_args()
    media, output = args.media, args.output
    audio_root = Path("C:/CodexArtifacts/pretrained-reuse-r1/efficientat-audit-v1")
    source = audio_root / "source"
    checkpoint = audio_root / "dymn04_as.pt"
    if sha256_file(checkpoint) != AUDIO_SHA:
        raise ValueError("checkpoint mismatch")
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                                       text=True).strip()
    if revision != AUDIO_SOURCE:
        raise ValueError("source mismatch")
    acquisition = json.loads((media / "result.json").read_text())
    if acquisition["status"] != "PASS":
        raise ValueError("incomplete media")
    for row in acquisition["files"]:
        if sha256_file(media / f"{row['source_id']}.wav") != row["sha256"]:
            raise ValueError("media mismatch")
    old_root = root / "starss23-audio-corpus-features-v1"
    old_report = json.loads((old_root / "result.json").read_text())
    if sha256_file(old_root / "features.npy") != old_report["feature_sha256"]:
        raise ValueError("reference features mismatch")
    output.mkdir(exist_ok=False)
    sys.path.insert(0, str(source))
    from models.dymn.model import get_model
    from models.preprocess import AugmentMelSTFT

    torch.set_num_threads(4)
    with contextlib.redirect_stdout(io.StringIO()):
        model = get_model(width_mult=0.4, pretrained_name=None, T_max=1.0)
        mel = AugmentMelSTFT().eval()
    model.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    model.eval().requires_grad_(False)
    results, features = [], []
    started = time.perf_counter()
    for row in acquisition["files"]:
        # Decode native PCM with PyAV, then use the same torchaudio resampler as STARSS.
        import torchaudio

        with av.open(str(media / f"{row['source_id']}.wav")) as container:
            stream = container.streams.audio[0]
            rate = stream.codec_context.sample_rate
            converter = av.AudioResampler(format="fltp", layout="mono", rate=rate)
            arrays = []
            for frame in container.decode(stream):
                arrays.extend(f.to_ndarray() for f in converter.resample(frame))
            arrays.extend(f.to_ndarray() for f in converter.resample(None))
        waveform = torch.from_numpy(np.concatenate(arrays, axis=1))
        waveform = torchaudio.functional.resample(waveform, rate, 32000)
        with torch.inference_mode():
            _, feature = model(mel(waveform).unsqueeze(1))
            _, replay = model(mel(waveform).unsqueeze(1))
        if not torch.isfinite(feature).all() or not torch.equal(feature, replay):
            raise ValueError("nonfinite or nonrepeatable features")
        features.append(feature.squeeze(0).numpy())
        results.append({"id": row["source_id"], "samples_32k": waveform.shape[1]})
    features = np.stack(features)
    np.save(output / "features.npy", features)
    old = np.load(old_root / "features.npy", allow_pickle=False)
    similarity = (features / np.linalg.norm(features, axis=1, keepdims=True)) @ (
        old / np.linalg.norm(old, axis=1, keepdims=True)).T
    for row, scores in zip(results, similarity, strict=True):
        i = int(scores.argmax())
        row["nearest_starss"] = {"id": old_report["features"][i]["id"],
                                 "split": old_report["features"][i]["split"],
                                 "cosine": float(scores[i])}
    report = {"status": "PASS", "rows": results, "seconds": time.perf_counter() - started,
              "checkpoint_sha256": AUDIO_SHA, "source_commit": AUDIO_SOURCE,
              "feature_sha256": sha256_file(output / "features.npy"),
              "media_report_sha256": sha256_file(media / "result.json"),
              "reference_features_sha256": old_report["feature_sha256"],
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "parameters": sum(p.numel() for p in model.parameters()),
              "device": "cpu", "training": False, "torch": torch.__version__,
              "preprocessing": "native mono float; torchaudio native->32k; official eval mel",
              "limitation": "Whole clips versus 5s windows. Cosine is not a duplication test."}
    (output / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
