"""CPU runtime for the fixed Clotho audio-question binary Decision candidate."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
AUDIO_ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r1/efficientat-audit-v1")
TEXT_ROOT = ROOT / "models/text-minilm-en"
MODEL_DIR = ROOT / "clotho-aqa-joint-control-01"
EXPECTED = {
    "checkpoint": "ac3fc2012c146538537a293e007a386fd1c738ba6ee0a7452cc89ccf05daab4e",
    "normalization": "f152042f1f1ff03a4affb3381b5419dfe390edf64420dbd1ed178e835bbe1791",
    "audio_checkpoint": "49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e",
    "audio_source": "a425fdce92572e602a1d5634799bd9f1f2efa806",
    "text_weights": "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def option_order(option_ids: list[str], option_texts: list[str]) -> list[int]:
    """Return canonical class indices in caller order: class 0=no, class 1=yes."""
    if len(option_ids) != 2 or len(option_texts) != 2 or len(set(option_ids)) != 2:
        raise ValueError("exactly two uniquely identified binary options are required")
    labels = [text.strip().casefold() for text in option_texts]
    if sorted(labels) != ["no", "yes"]:
        raise ValueError("the fixed readout supports only supplied No/Yes options")
    return [0 if label == "no" else 1 for label in labels]


def candidate_features(audio: np.ndarray, question: np.ndarray) -> np.ndarray:
    if audio.shape != (384,) or question.shape != (384,):
        raise ValueError("audio and question encoders must each return 384 values")
    if not np.isfinite(audio).all() or not np.isfinite(question).all():
        raise ValueError("audio and question features must be finite")
    shared = np.concatenate((audio, question)).astype(np.float32, copy=False)
    expanded = np.repeat(shared[None, :], 2, axis=0)
    return np.concatenate((expanded, np.eye(2, dtype=np.float32)), axis=1)


class ClothoAqaPredictor:
    """Load pinned frozen encoders and one immutable audio+question readout."""

    def __init__(self) -> None:
        import av
        import torch
        import torchaudio
        from transformers import AutoModel, AutoTokenizer

        from scripts.extract_starss23_features import AUDIO_SHA, AUDIO_SOURCE
        from tiny_omni_decision.decision import FrozenFeatureCandidateScorer

        checkpoint = MODEL_DIR / "best.pt"
        normalization_path = MODEL_DIR / "normalization.npz"
        audio_checkpoint = AUDIO_ROOT / "dymn04_as.pt"
        audio_source = AUDIO_ROOT / "source"
        if sha256(checkpoint) != EXPECTED["checkpoint"]:
            raise ValueError("frozen Decision checkpoint hash mismatch")
        if sha256(normalization_path) != EXPECTED["normalization"]:
            raise ValueError("frozen normalization hash mismatch")
        if sha256(audio_checkpoint) != EXPECTED["audio_checkpoint"] or AUDIO_SHA != EXPECTED[
            "audio_checkpoint"
        ]:
            raise ValueError("EfficientAT checkpoint hash mismatch")
        if AUDIO_SOURCE != EXPECTED["audio_source"]:
            raise ValueError("EfficientAT source pin differs")
        revision = subprocess.check_output(
            ["git", "-C", str(audio_source), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != EXPECTED["audio_source"]:
            raise ValueError("EfficientAT source revision mismatch")
        if sha256(TEXT_ROOT / "model.safetensors") != EXPECTED["text_weights"]:
            raise ValueError("MiniLM checkpoint hash mismatch")

        self.torch = torch
        self.torchaudio = torchaudio
        self.av = av
        torch.set_num_threads(4)
        self.text_model = AutoModel.from_pretrained(
            TEXT_ROOT, local_files_only=True
        ).eval().requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_ROOT, local_files_only=True)

        import sys

        if str(audio_source) not in sys.path:
            sys.path.insert(0, str(audio_source))
        from models.dymn.model import get_model
        from models.preprocess import AugmentMelSTFT

        with contextlib.redirect_stdout(io.StringIO()):
            self.audio_model = get_model(width_mult=0.4, pretrained_name=None, T_max=1.0)
        self.audio_model.load_state_dict(
            torch.load(audio_checkpoint, map_location="cpu", weights_only=True), strict=True
        )
        self.audio_model.eval().requires_grad_(False)
        self.mel = AugmentMelSTFT().eval()
        self.normalization = np.load(normalization_path, allow_pickle=False)
        self.scorer = FrozenFeatureCandidateScorer(770, 64)
        self.scorer.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
        )
        self.scorer.eval().requires_grad_(False)
        self.temperature = float(
            json.loads((MODEL_DIR / "result.json").read_text(encoding="utf-8"))["temperature"]
        )
        if self.temperature != 1.4142135623730951:
            raise ValueError("frozen temperature differs from the audited checkpoint")

    def encode_question(self, question: str) -> np.ndarray:
        if not question.strip():
            raise ValueError("question must be nonempty")
        tokens = self.tokenizer(question, padding=True, truncation=False, return_tensors="pt")
        if int(tokens["input_ids"].shape[1]) > 128:
            raise ValueError("question exceeds the fixed 128-token feature limit")
        with self.torch.inference_mode():
            hidden = self.text_model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            feature = (hidden * mask).sum(1) / mask.sum(1)
        return feature.squeeze(0).cpu().numpy().astype(np.float32, copy=False)

    def encode_audio(self, audio_path: Path) -> np.ndarray:
        with self.av.open(str(audio_path)) as container:
            stream = container.streams.audio[0]
            rate = stream.codec_context.sample_rate
            converter = self.av.AudioResampler(format="fltp", layout="mono", rate=rate)
            frames = []
            for frame in container.decode(stream):
                frames.extend(frame.to_ndarray() for frame in converter.resample(frame))
            frames.extend(frame.to_ndarray() for frame in converter.resample(None))
        if not frames:
            raise ValueError("audio file has no decodable samples")
        waveform = self.torch.from_numpy(np.concatenate(frames, axis=1))
        waveform = self.torchaudio.functional.resample(waveform, rate, 32000)
        with self.torch.inference_mode():
            _, feature = self.audio_model(self.mel(waveform).unsqueeze(1))
        if feature.shape != (1, 384) or not self.torch.isfinite(feature).all():
            raise ValueError("EfficientAT returned invalid audio features")
        return feature.squeeze(0).cpu().numpy().astype(np.float32, copy=False)

    def predict(
        self,
        audio_path: Path,
        question: str,
        option_ids: list[str],
        option_texts: list[str],
        *,
        request_id: str = "clotho-aqa",
    ) -> str:
        order = option_order(option_ids, option_texts)
        audio = self.encode_audio(audio_path)
        question_vector = self.encode_question(question)
        norm = self.normalization
        audio = (audio - norm["audio_mean"]) / norm["audio_std"]
        question_vector = (question_vector - norm["question_mean"]) / norm["question_std"]
        features = self.torch.from_numpy(candidate_features(audio, question_vector))
        with self.torch.inference_mode():
            raw_logits = self.scorer(features).float()
            canonical_probabilities = (raw_logits / self.temperature).softmax(-1)
        index = self.torch.tensor(order, dtype=self.torch.long)
        logits = raw_logits[index]
        probabilities = canonical_probabilities[index]
        from tiny_omni_decision.decision import DecisionOutput

        output = DecisionOutput(
            option_logits=logits,
            probabilities=probabilities,
            predicted_index=int(probabilities.argmax().item()),
            confidence=float(probabilities.max().item()),
        )
        return output.to_decision_json(
            request_id=request_id,
            decision_type="yes_no",
            option_ids=option_ids,
            option_texts=option_texts,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--option-ids", nargs=2, required=True)
    parser.add_argument("--option-texts", nargs=2, required=True)
    parser.add_argument("--request-id", default="clotho-aqa")
    args = parser.parse_args()
    if not args.audio.is_file():
        raise FileNotFoundError(args.audio)
    print(
        ClothoAqaPredictor().predict(
            args.audio,
            args.question,
            args.option_ids,
            args.option_texts,
            request_id=args.request_id,
        )
    )


if __name__ == "__main__":
    main()
