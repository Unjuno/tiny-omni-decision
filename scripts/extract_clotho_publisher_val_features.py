"""Extract pinned frozen MiniLM and EfficientAT features for gated Clotho Val."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import shutil
import subprocess
import time
from pathlib import Path

import av
import numpy as np
import torch
import torchaudio

from scripts.extract_starss23_features import AUDIO_SHA, AUDIO_SOURCE
from scripts.fetch_clotho_publisher_val_media import (
    ATTEMPT,
    MANIFEST_SHA256,
    load_manifest,
    unique_media_rows,
)

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
OUTPUT = ATTEMPT / "features"
MEDIA = ATTEMPT / "media"
AUDIO_ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r1/efficientat-audit-v1")
AUDIO_PARITY_SOURCE_IDS = ("100507", "101390")
PREVIOUS_GENERATION_SCRIPT_SHA256 = (
    "40046f1288334a8bf34a58a504ca0fe3bb36f23e160538bd297a9c2814f5f1dd"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_audio(path: Path, model, mel) -> tuple[np.ndarray, int]:
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        rate = stream.codec_context.sample_rate
        converter = av.AudioResampler(format="fltp", layout="mono", rate=rate)
        arrays = []
        for frame in container.decode(stream):
            arrays.extend(frame.to_ndarray() for frame in converter.resample(frame))
        arrays.extend(frame.to_ndarray() for frame in converter.resample(None))
    if not arrays:
        raise ValueError(f"no decoded PCM: {path.name}")
    waveform = torch.from_numpy(np.concatenate(arrays, axis=1))
    waveform = torchaudio.functional.resample(waveform, rate, 32000)
    with torch.inference_mode():
        _, feature = model(mel(waveform).unsqueeze(1))
    if feature.shape != (1, 384) or not torch.isfinite(feature).all():
        raise ValueError(f"invalid EfficientAT output: {path.name}")
    return feature.squeeze(0).cpu().numpy(), int(waveform.shape[1])


def run(output: Path = OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    questions = load_manifest()
    media_rows = unique_media_rows(questions)
    media_report_path = MEDIA / "result.json"
    media_report_bytes = media_report_path.read_bytes()
    media_report = json.loads(media_report_bytes)
    media_ids = {row["source_id"] for row in media_report.get("files", [])}
    expected_ids = {row["source_id"] for row in media_rows}
    if media_report.get("status") != "PASS" or media_ids != expected_ids:
        raise ValueError("publisher validation media differs from frozen cohort")

    text_dir = ATTEMPT / "text-features"
    text_report_path = text_dir / "result.json"
    text_report_bytes = text_report_path.read_bytes()
    text_report = json.loads(text_report_bytes)
    text_features_path = text_dir / "question-features.npy"
    text_ids_path = text_dir / "question-ids.json"
    if (
        text_report.get("status") != "PASS"
        or text_report.get("manifest_sha256") != MANIFEST_SHA256
        or sha256(text_features_path) != text_report["feature_sha256"]
        or sha256(text_ids_path) != text_report["ids_sha256"]
    ):
        raise ValueError("staged MiniLM features differ from frozen cohort")
    question_features = np.load(text_features_path, allow_pickle=False)
    question_ids = json.loads(text_ids_path.read_text(encoding="utf-8"))
    expected_question_ids = [
        [row["source_id"], row["normalized_question"]] for row in questions
    ]
    if (
        question_features.shape != (len(questions), 384)
        or not np.isfinite(question_features).all()
        or question_ids != expected_question_ids
    ):
        raise ValueError("staged text IDs/order/shape differ from frozen manifest")

    audio_checkpoint = AUDIO_ROOT / "dymn04_as.pt"
    audio_source = AUDIO_ROOT / "source"
    if sha256(audio_checkpoint) != AUDIO_SHA:
        raise ValueError("EfficientAT checkpoint hash mismatch")
    audio_revision = subprocess.check_output(
        ["git", "-C", str(audio_source), "rev-parse", "HEAD"], text=True
    ).strip()
    if audio_revision != AUDIO_SOURCE:
        raise ValueError("EfficientAT implementation revision mismatch")

    sys_path = str(audio_source)
    import sys

    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    with contextlib.redirect_stdout(io.StringIO()):
        from models.dymn.model import get_model
        from models.preprocess import AugmentMelSTFT

        audio_model = get_model(width_mult=0.4, pretrained_name=None, T_max=1.0)
        mel = AugmentMelSTFT().eval()
    audio_model.load_state_dict(
        torch.load(audio_checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    audio_model.eval().requires_grad_(False)
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    shutil.copyfile(text_features_path, output / "question-features.npy")
    shutil.copyfile(text_ids_path, output / "question-ids.json")

    audio_rows = []
    audio_features = []
    for row in media_rows:
        path = MEDIA / f"{row['source_id']}.wav"
        feature, samples_32k = decode_audio(path, audio_model, mel)
        audio_features.append(feature)
        audio_rows.append({"source_id": row["source_id"], "samples_32k": samples_32k})
    audio_features = np.stack(audio_features).astype(np.float32, copy=False)
    np.save(output / "audio-features.npy", audio_features)
    (output / "audio-ids.json").write_text(
        json.dumps([row["source_id"] for row in audio_rows]), encoding="utf-8"
    )

    # Re-extract two fixed, existing Clotho assets and verify the previous frozen
    # EfficientAT feature cache, so preprocessing is checked before scoring.
    old_media = ROOT / "clotho-aqa-development-media-02"
    old_feature_dir = ROOT / "clotho-aqa-audio-features-01"
    old_feature_report = json.loads((old_feature_dir / "result.json").read_text())
    old_feature_path = old_feature_dir / "features.npy"
    if sha256(old_feature_path) != old_feature_report["feature_sha256"]:
        raise ValueError("saved EfficientAT feature cache hash mismatch")
    old_features = np.load(old_feature_path, allow_pickle=False)
    old_feature_ids = [row["id"] for row in old_feature_report["rows"]]
    audio_parity = []
    for source_id in AUDIO_PARITY_SOURCE_IDS:
        old_index = old_feature_ids.index(source_id)
        feature, _ = decode_audio(old_media / f"{source_id}.wav", audio_model, mel)
        max_error = float(np.max(np.abs(feature - old_features[old_index])))
        if max_error > 1e-5:
            raise ValueError("EfficientAT feature preprocessing differs from frozen reference")
        audio_parity.append({"source_id": source_id, "max_abs_error": max_error})

    question_ids_bytes = (output / "question-ids.json").read_bytes()
    report = {
        "status": "PASS",
        "scope": "frozen features only; no fitting or calibration",
        "device": "cpu",
        "torch": torch.__version__,
        "audio_encoder": {
            "source_revision": audio_revision,
            "checkpoint_sha256": AUDIO_SHA,
            "parameters": sum(parameter.numel() for parameter in audio_model.parameters()),
            "features_shape": list(audio_features.shape),
            "feature_sha256": sha256(output / "audio-features.npy"),
            "preprocessing": "native mono float; torchaudio native->32k; official eval mel",
            "parity": audio_parity,
        },
        "text_features_report_sha256": hashlib.sha256(text_report_bytes).hexdigest(),
        "text_encoder": text_report["encoder"],
        "text_feature_sha256": sha256(output / "question-features.npy"),
        "media_report_sha256": hashlib.sha256(media_report_bytes).hexdigest(),
        "question_count": len(questions),
        "audio_source_count": len(media_rows),
        "question_ids_sha256": hashlib.sha256(question_ids_bytes).hexdigest(),
        "generation_script_sha256": sha256(Path(__file__)),
        "seconds": time.perf_counter() - start,
        "script_sha256": sha256(Path(__file__)),
    }
    (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def finalize_existing(output: Path = OUTPUT) -> dict:
    """Finish provenance checks after feature arrays were saved before an error."""
    result_path = output / "result.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")
    questions = load_manifest()
    media_rows = unique_media_rows(questions)
    text_dir = ATTEMPT / "text-features"
    text_report_bytes = (text_dir / "result.json").read_bytes()
    text_report = json.loads(text_report_bytes)
    media_report_bytes = (MEDIA / "result.json").read_bytes()
    media_report = json.loads(media_report_bytes)
    audio_features_path = output / "audio-features.npy"
    audio_ids_path = output / "audio-ids.json"
    question_features_path = output / "question-features.npy"
    question_ids_path = output / "question-ids.json"
    for path in (audio_features_path, audio_ids_path, question_features_path, question_ids_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    audio_features = np.load(audio_features_path, allow_pickle=False)
    question_features = np.load(question_features_path, allow_pickle=False)
    audio_ids = json.loads(audio_ids_path.read_text(encoding="utf-8"))
    question_ids = json.loads(question_ids_path.read_text(encoding="utf-8"))
    expected_audio_ids = [row["source_id"] for row in media_rows]
    expected_question_ids = [
        [row["source_id"], row["normalized_question"]] for row in questions
    ]
    if (
        media_report.get("status") != "PASS"
        or text_report.get("status") != "PASS"
        or text_report.get("manifest_sha256") != MANIFEST_SHA256
        or audio_ids != expected_audio_ids
        or question_ids != expected_question_ids
        or audio_features.shape != (len(media_rows), 384)
        or question_features.shape != (len(questions), 384)
        or not np.isfinite(audio_features).all()
        or not np.isfinite(question_features).all()
    ):
        raise ValueError("saved feature arrays or IDs do not match the frozen cohort")
    if sha256(question_features_path) != text_report["feature_sha256"]:
        raise ValueError("saved MiniLM feature hash mismatch")

    audio_checkpoint = AUDIO_ROOT / "dymn04_as.pt"
    audio_source = AUDIO_ROOT / "source"
    if sha256(audio_checkpoint) != AUDIO_SHA:
        raise ValueError("EfficientAT checkpoint hash mismatch")
    audio_revision = subprocess.check_output(
        ["git", "-C", str(audio_source), "rev-parse", "HEAD"], text=True
    ).strip()
    if audio_revision != AUDIO_SOURCE:
        raise ValueError("EfficientAT implementation revision mismatch")
    import sys

    if str(audio_source) not in sys.path:
        sys.path.insert(0, str(audio_source))
    with contextlib.redirect_stdout(io.StringIO()):
        from models.dymn.model import get_model
        from models.preprocess import AugmentMelSTFT

        model = get_model(width_mult=0.4, pretrained_name=None, T_max=1.0)
        mel = AugmentMelSTFT().eval()
    model.load_state_dict(
        torch.load(audio_checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    model.eval().requires_grad_(False)
    old_media = ROOT / "clotho-aqa-development-media-02"
    old_feature_dir = ROOT / "clotho-aqa-audio-features-01"
    old_report = json.loads((old_feature_dir / "result.json").read_text())
    old_path = old_feature_dir / "features.npy"
    if sha256(old_path) != old_report["feature_sha256"]:
        raise ValueError("saved EfficientAT feature cache hash mismatch")
    old_features = np.load(old_path, allow_pickle=False)
    old_ids = [row["id"] for row in old_report["rows"]]
    parity = []
    for source_id in AUDIO_PARITY_SOURCE_IDS:
        feature, _ = decode_audio(old_media / f"{source_id}.wav", model, mel)
        error = float(np.max(np.abs(feature - old_features[old_ids.index(source_id)])))
        if error > 1e-5:
            raise ValueError("EfficientAT feature preprocessing differs from frozen reference")
        parity.append({"source_id": source_id, "max_abs_error": error})

    report = {
        "status": "PASS",
        "scope": "frozen features only; no fitting or calibration",
        "device": "cpu",
        "torch": torch.__version__,
        "audio_encoder": {
            "source_revision": audio_revision,
            "checkpoint_sha256": AUDIO_SHA,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "features_shape": list(audio_features.shape),
            "feature_sha256": sha256(audio_features_path),
            "preprocessing": "native mono float; torchaudio native->32k; official eval mel",
            "parity": parity,
        },
        "text_features_report_sha256": hashlib.sha256(text_report_bytes).hexdigest(),
        "text_encoder": text_report["encoder"],
        "text_feature_sha256": sha256(question_features_path),
        "media_report_sha256": hashlib.sha256(media_report_bytes).hexdigest(),
        "question_count": len(questions),
        "audio_source_count": len(media_rows),
        "question_ids_sha256": sha256(question_ids_path),
        "audio_ids_sha256": sha256(audio_ids_path),
        "generation_script_sha256": PREVIOUS_GENERATION_SCRIPT_SHA256,
        "finalizer_script_sha256": sha256(Path(__file__)),
        "feature_extraction_duration_note": (
            "Audio array was written at 2026-11-26 08:47:29 local time; runner started at "
            "08:44:46. Approx. 163 seconds from runner start to saved feature array, "
            "including model setup."
        ),
    }
    result_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    report = finalize_existing() if args.finalize_existing else run()
    print(json.dumps(report, indent=2))
